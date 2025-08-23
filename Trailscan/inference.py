"""
***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from typing import Any, Optional
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingParameterFile,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    Qgis,
    QgsRasterLayer,
    QgsProcessingMultiStepFeedback,
)
from qgis import processing
import numpy as np
import rasterio
import onnxruntime as ort
import itertools
import os
from qgis.PyQt.QtGui import QIcon
import time
from sys import stdout

PIXEL_SIZE = 0.38  # Example pixel size, adjust as needed
MODEL_CONFIG = {
    'in_shape': (4, 448, 448),  # Channels, Height, Width (match ONNX model)
    'out_bands': 1,
    'stride': 112,
    'augmentation': True,
    'batch_size': 1,  # ONNX model expects static batch size = 1
    'tile_size': 448,
    'overlap': 112
}


class TrailscanInferenceProcessingAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    MODEL_FILE = "MODEL_FILE"

    def name(self) -> str:
        return "inference"

    def displayName(self) -> str:
        return "02 Inference"

    def group(self) -> str:
        return ""

    def groupId(self) -> str:
        return ""

    def shortHelpString(self) -> str:
        return "Trailscan segmentation"

    def icon(self):
        return QIcon(os.path.join(os.path.dirname(__file__), 'TrailScan_Logo.svg'))

    def initAlgorithm(self, config: Optional[dict[str, Any]] = None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                name=self.INPUT,
                description="Input preprocessed point cloud data",
                defaultValue="Normalized",
                optional=False,
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                name=self.MODEL_FILE,
                description="Path to the ONNX model file",
                behavior=QgsProcessingParameterFile.Behavior.File,
                fileFilter="ONNX model files (*.onnx);;All files (*)",
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT,
                description="Trailmap"
            )
        )

    # ---------- Prediction Helper Methods ----------
    def predict_batch_onnx(self, session, input_name, batch: np.ndarray) -> np.ndarray:
        inputs = {input_name: batch.astype(np.float32)}
        outputs = session.run(None, inputs)
        return outputs[0]

    def compute_pyramid_patch_weight_loss(self, width: int, height: int) -> np.ndarray:
        xc = width * 0.5
        yc = height * 0.5
        xl = 0
        xr = width
        yb = 0
        yt = height

        Dcx = np.square(np.arange(width) - xc + 0.5)
        Dcy = np.square(np.arange(height) - yc + 0.5)
        Dc = np.sqrt(Dcx[np.newaxis].transpose() + Dcy)

        De_l = np.square(np.arange(width) - xl + 0.5) + np.square(0.5)
        De_r = np.square(np.arange(width) - xr + 0.5) + np.square(0.5)
        De_b = np.square(0.5) + np.square(np.arange(height) - yb + 0.5)
        De_t = np.square(0.5) + np.square(np.arange(height) - yt + 0.5)

        De_x = np.sqrt(np.minimum(De_l, De_r))
        De_y = np.sqrt(np.minimum(De_b, De_t))
        De = np.minimum(De_x[np.newaxis].transpose(), De_y)

        alpha = (width * height) / np.sum(np.divide(De, np.add(Dc, De)))
        W = alpha * np.divide(De, np.add(Dc, De))
        return W

    def predict_on_array_cf(self, model,
                            arr,
                            in_shape,
                            out_bands,
                            stride=None,
                            drop_border=0,
                            batchsize=64,
                            dtype="float32",
                            device="cpu",
                            augmentation=False,
                            no_data=None,
                            verbose=False,
                            aggregate_metric=False,
                            input_name=None):
        t0 = time.time()
        metric = 0

        operations, inverse = (lambda x: x,), (lambda x: x,)
        if augmentation:
            operations = (lambda x: x,
                          lambda x: np.rot90(x, 1, axes=(1, 2)),
                          lambda x: np.flip(x, 1))
            inverse = (lambda x: x,
                       lambda x: np.rot90(x, -1, axes=(1, 2)),
                       lambda x: np.flip(x, 1))

        assert in_shape[1] == in_shape[2], "Input shape must be square."
        out_shape = (out_bands, in_shape[1] - 2 * drop_border, in_shape[2] - 2 * drop_border)
        in_size = in_shape[1]
        out_size = out_shape[1]
        stride = stride or out_size
        pad = (in_size - out_size) // 2
        assert pad % 2 == 0, "Model input and output shapes must have pad divisible by 2."

        original_size = arr.shape
        ymin, xmin = 0, 0
        ymax, xmax = arr.shape[1], arr.shape[2]

        if no_data is not None:
            nonzero = np.nonzero(arr[0, :, :] != no_data)
            if len(nonzero[0]) == 0:
                return {"prediction": None,
                        "time": time.time() - t0,
                        "nodata_region": (0, 0, 0, 0),
                        "metric": metric}
            ymin, ymax = int(np.min(nonzero[0])), int(np.max(nonzero[0]))
            xmin, xmax = int(np.min(nonzero[1])), int(np.max(nonzero[1]))
            img = arr[:, ymin:ymax, xmin:xmax]
        else:
            img = arr

        weight_mask = self.compute_pyramid_patch_weight_loss(out_size, out_size)
        final_output = np.zeros((out_bands,) + img.shape[1:], dtype=dtype)

        for op_cnt, (op, inv) in enumerate(zip(operations, inverse)):
            img_aug = op(img)
            img_shape = img_aug.shape

            x_tiles = int(np.ceil(img_shape[2] / stride))
            y_tiles = int(np.ceil(img_shape[1] / stride))

            y_range = range(0, (y_tiles + 1) * stride - out_size, stride)
            x_range = range(0, (x_tiles + 1) * stride - out_size, stride)

            y_pad_after = y_range[-1] + in_size - img_shape[1] - pad
            x_pad_after = x_range[-1] + in_size - img_shape[2] - pad

            output = np.zeros((out_bands,) + (img_shape[1] + y_pad_after - pad, img_shape[2] + x_pad_after - pad),
                              dtype=dtype)
            division_mask = np.zeros(output.shape[1:], dtype=dtype) + 1E-7
            img_padded = np.pad(img_aug, ((0, 0), (pad, y_pad_after), (pad, x_pad_after)), mode='reflect')

            patches = len(y_range) * len(x_range)
            patch_gen = (img_padded[:, y:y + in_size, x:x + in_size] for y in y_range for x in x_range)
            y_idx = x_idx = 0
            patch_idx = 0

            while patch_idx < patches:
                bsize = min(batchsize, patches - patch_idx)
                batch_idx_range = range(patch_idx, patch_idx + bsize)
                batch = np.stack([next(patch_gen) for _ in batch_idx_range], axis=0)
                prediction = self.predict_batch_onnx(model, input_name, batch)
                if drop_border > 0:
                    prediction = prediction[:, :, drop_border:-drop_border, drop_border:-drop_border]
                for j in range(bsize):
                    output[:, y_idx:y_idx + out_size, x_idx:x_idx + out_size] += prediction[j] * weight_mask[None, ...]
                    division_mask[y_idx:y_idx + out_size, x_idx:x_idx + out_size] += weight_mask
                    x_idx += stride
                    if x_idx + out_size > output.shape[2]:
                        x_idx = 0
                        y_idx += stride
                patch_idx += bsize

            output /= division_mask[None, ...]
            output = inv(output[:, :img_shape[1], :img_shape[2]])
            final_output += output

        final_output /= len(operations)
        if no_data is not None:
            final_output = np.pad(final_output,
                                  ((0, 0), (ymin, original_size[1] - ymax), (xmin, original_size[2] - xmax)),
                                  mode='constant', constant_values=0)

        return {"prediction": final_output,
                "time": time.time() - t0,
                "nodata_region": (ymin, ymax, xmin, xmax),
                "metric": metric}

    # ---------- Main Processing ----------
    def processAlgorithm(
            self,
            parameters: dict[str, Any],
            context: QgsProcessingContext,
            feedback: QgsProcessingMultiStepFeedback,
    ) -> dict[str, Any]:

        feedback.pushInfo("Starting Trailscan inference...")

        source = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        model_path = self.parameterAsFile(parameters, self.MODEL_FILE, context)
        output_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        feedback.pushInfo(f"Loading ONNX model from {model_path}...")
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name

        feedback.pushInfo(f"Reading raster data from {source.source()}...")
        with rasterio.open(source.source()) as src:
            arr = src.read().astype(np.float32)

        feedback.pushInfo("Running prediction...")
        result_dict = self.predict_on_array_cf(
            model=session,
            arr=arr,
            in_shape=MODEL_CONFIG['in_shape'],
            out_bands=MODEL_CONFIG['out_bands'],
            stride=MODEL_CONFIG['stride'],
            batchsize=MODEL_CONFIG['batch_size'],
            augmentation=MODEL_CONFIG['augmentation'],
            input_name=input_name
        )

        prediction = result_dict["prediction"]
        if prediction is None:
            feedback.reportError("No prediction could be made.")
            return {}

        feedback.pushInfo("Writing prediction to output raster...")
        meta = src.meta.copy()
        meta.update({
            "driver": "GTiff",
            "count": prediction.shape[0],
            "dtype": prediction.dtype
        })

        with rasterio.open(output_raster, "w", **meta) as dst:
            dst.write(prediction)

        feedback.pushInfo("Inference completed successfully.")
        return {self.OUTPUT: output_raster}
