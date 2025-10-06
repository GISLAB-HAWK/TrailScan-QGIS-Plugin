"""
#-----------------------------------------------------------
# Copyright (C) 2025 Tanja Kempen, Mathias Gröbe
#-----------------------------------------------------------
# Licensed under the terms of GNU GPL 2
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

#---------------------------------------------------------------------
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

PIXEL_SIZE = 0.25  # Example pixel size, adjust as needed
MODEL_CONFIG = {
    'in_shape': (4, 224, 224),  # Channels, Height, Width (match ONNX model)
    'out_bands': 1,
    'stride': 112,
    'augmentation': True,
    'batch_size': 1,  # ONNX model expects static batch size = 1
    'tile_size': 224,
    'overlap': 112
}


class TrailscanInferenceProcessingAlgorithm(QgsProcessingAlgorithm):
    """

    """

    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    MODEL_FILE = "MODEL_FILE"

    def name(self) -> str:
        """
        Returns the algorithm name
        """
        return "inference"

    def displayName(self) -> str:
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return "02 Inference"

    def group(self) -> str:
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return ""

    def groupId(self) -> str:
        """
        Returns the unique ID of the group this algorithm belongs to.
        """
        return ""

    def shortHelpString(self) -> str:
        """
        Returns a localised short helper string for the algorithm.
        """

        help_string = (
        "The normalized raster is processed with the TrailScan Model.\n\n"
        "Download the trained TrailScan Model here:\n"
        "https://doi.org/10.25625/GEIP6T\n\n"
        "Save the TrailScan Model on your local drive\n"
        "Output:\n"
        "- Trailmap: A probability raster with values between 0 and 1.\n"
        "- Pixels with a value near 0 indicate areas that are not skid trails.\n"
        "- Higher values indicate a higher probability that a skid trail was detected."
    )
        return help_string

    def icon(self):
        return QIcon(os.path.join(os.path.dirname(__file__), 'TrailScan_Inference.svg'))

    def initAlgorithm(self, config: Optional[dict[str, Any]] = None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                name=self.INPUT,
                description="Input normalized raster from TrailScan Preprocessing tool",
                defaultValue="Normalized",
                optional=False,
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                name=self.MODEL_FILE,
                description="Path to the TrailScan model file",
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

    def predict_batch_onnx(self, session, input_name, batch: np.ndarray) -> np.ndarray:
        inputs = {input_name: batch.astype(np.float32)}
        outputs = session.run(None, inputs)
        return outputs[0]  # Shape: (N, C, H, W)

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
        """
        Applies a ONNX segmentation model to an array in a strided manner.
        Channels first version.

        Args:
            model: ONNX InferenceSession
            arr: CHW array for which the segmentation should be created
            stride: stride with which the model should be applied. Default: output size
            batchsize: number of images to process in parallel
            dtype: desired output type (default: float32)
            augmentation: whether to average over rotations and mirrorings of the image or not.
            no_data: a no-data value. It's used to compute the area containing data via the first input image channel.
            verbose: whether or not to display progress
            input_name: ONNX input tensor name

        Returns:
            dict with prediction, time, nodata_region and metric
        """
        t0 = time.time()
        metric = 0

        if augmentation:
            operations = (lambda x: x,
                          lambda x: np.rot90(x, 1, axes=(1, 2)),
                          lambda x: np.flip(x, 1))
            inverse = (lambda x: x,
                       lambda x: np.rot90(x, -1, axes=(1, 2)),
                       lambda x: np.flip(x, 1))
        else:
            operations = (lambda x: x,)
            inverse = (lambda x: x,)

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
            ymin = int(np.min(nonzero[0]))
            ymax = int(np.max(nonzero[0]))
            xmin = int(np.min(nonzero[1]))
            xmax = int(np.max(nonzero[1]))
            img = arr[:, ymin:ymax, xmin:xmax]
        else:
            img = arr

        weight_mask = self.compute_pyramid_patch_weight_loss(out_size, out_size)
        final_output = np.zeros((out_bands,) + img.shape[1:], dtype=dtype)

        op_cnt = 0
        for op, inv in zip(operations, inverse):
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

            def patch_generator():
                for y in y_range:
                    for x in x_range:
                        yield img_padded[:, y:y + in_size, x:x + in_size]

            patch_gen = patch_generator()

            y = 0
            x = 0
            patch_idx = 0
            batchsize_ = batchsize

            while patch_idx < patches:
                batchsize_ = min(batchsize_, patches, patches - patch_idx)
                patch_idx += batchsize_
                if verbose:
                    stdout.write("\r%.2f%%" % (100 * (patch_idx + op_cnt * patches) / (len(operations) * patches)))

                batch = np.zeros((batchsize_,) + in_shape, dtype=dtype)
                for j in range(batchsize_):
                    batch[j] = next(patch_gen)

                prediction = self.predict_batch_onnx(model, input_name, batch)
                if drop_border > 0:
                    prediction = prediction[:, :, drop_border:-drop_border, drop_border:-drop_border]

                for j in range(batchsize_):
                    output[:, y:y + out_size, x:x + out_size] += prediction[j] * weight_mask[None, ...]
                    division_mask[y:y + out_size, x:x + out_size] += weight_mask
                    x += stride
                    if x + out_size > output.shape[2]:
                        x = 0
                        y += stride

            output = output / division_mask[None, ...]
            output = inv(output[:, :img_shape[1], :img_shape[2]])
            final_output += output
            op_cnt += 1
            if verbose:
                stdout.write("\rAugmentation step %d/%d done.\n" % (op_cnt, len(operations)))

        if verbose:
            stdout.flush()

        final_output = final_output / len(operations)

        if no_data is not None:
            final_output = np.pad(final_output,
                                  ((0, 0), (ymin, original_size[1] - ymax), (xmin, original_size[2] - xmax)),
                                  mode='constant',
                                  constant_values=0)

        return {"prediction": final_output,
                "time": time.time() - t0,
                "nodata_region": (ymin, ymax, xmin, xmax),
                "metric": metric}

    def compute_pyramid_patch_weight_loss(self, width: int, height: int) -> np.ndarray:
        """Compute a weight matrix that assigns bigger weight on pixels in center and
        less weight to pixels on image boundary.
        This weight matrix is then used for merging individual tile predictions and helps dealing
        with prediction artifacts on tile boundaries.

        Taken from & credit to:
            https://github.com/BloodAxe/pytorch-toolbelt/blob/f3acfca5da05cd7ccdd85e8d343d75fa40fb44d9/pytorch_toolbelt/inference/tiles.py#L16-L50

        Args:
            width: Tile width
            height: Tile height
        Returns:
            The weight mask as ndarray
        """
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

    def processAlgorithm(
            self,
            parameters: dict[str, Any],
            context: QgsProcessingContext,
            feedback: QgsProcessingMultiStepFeedback,
    ) -> dict[str, Any]:
        """
        Here is where the processing itself takes place.

        """

        counter = itertools.count(1)
        count_max = 12
        feedback = QgsProcessingMultiStepFeedback(count_max, feedback)

        source = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        model_path = self.parameterAsFile(parameters, self.MODEL_FILE, context)
        output_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        if source is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.INPUT)
            )

        if not os.path.isfile(model_path):
            raise QgsProcessingException(f"Model file does not exist: {model_path}")

        crs = source.crs().horizontalCrs()
        if not crs.isValid():
            raise QgsProcessingException("Invalid CRS in input raster")

        file_path = source.source()

        feedback.pushInfo(f"Using CRS: {crs.description()}")

        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_info = session.get_inputs()[0]
        input_name = input_info.name
        inferred_in_shape = MODEL_CONFIG['in_shape']
        inferred_stride = MODEL_CONFIG['stride']
        inferred_batch_size = MODEL_CONFIG['batch_size']

        with rasterio.open(file_path) as src:
            img = src.read()  # CHW (count, height, width)
            meta = src.meta.copy()

        pred = self.predict_on_array_cf(
            session,
            img,
            in_shape=inferred_in_shape,
            out_bands=MODEL_CONFIG['out_bands'],
            stride=inferred_stride,
            augmentation=MODEL_CONFIG['augmentation'],
            batchsize=inferred_batch_size,
            input_name=input_name
        )

        meta.update({
            'count': 1,
            'dtype': 'float32'
        })

        with rasterio.open(output_raster, 'w', **meta) as dst:
            dst.write(pred["prediction"][0].astype(np.float32), 1)

        raster_out = {'OUTPUT': output_raster}

        feedback.pushInfo(f"Output raster created: {output_raster}")

        return {self.OUTPUT: raster_out['OUTPUT']}

    def createInstance(self):
        return self.__class__()