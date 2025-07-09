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

PIXEL_SIZE = 0.38  # Example pixel size, adjust as needed
MODEL_NAME = "TrailScan.onnx"
MODEL_CONFIG = {
    'in_shape': (4, 448, 448),  # Channels, Height, Width
    'out_bands': 1,
    'stride': 224,
    'augmentation': True,
    'batch_size': 4,
    'tile_size': 256,
    'overlap': 32
}


class TrailscanInferenceProcessingAlgorithm(QgsProcessingAlgorithm):
    """

    """


    INPUT = "INPUT"
    OUTPUT = "OUTPUT"

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
        return "Trailscan segmentation"

    def initAlgorithm(self, config: Optional[dict[str, Any]] = None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                name=self.INPUT,
                description="Input preprocessed point cloud data",
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


    def augmentations_forward(self, batch: np.ndarray) -> list[np.ndarray]:
        # Nur 4 Varianten: original + 3 Rotationen
        aug_batches = []
        aug_batches.append(batch)  # original
        aug_batches.append(np.rot90(batch, k=1, axes=(2, 3)).copy())
        aug_batches.append(np.rot90(batch, k=2, axes=(2, 3)).copy())
        aug_batches.append(np.rot90(batch, k=3, axes=(2, 3)).copy())
        return aug_batches


    def reverse_augmentations(self, preds: list[np.ndarray]) -> np.ndarray:
        # weights: original 70%, augmentations each 10%
        weights = np.array([0.7, 0.1, 0.1, 0.1])

        restored = []
        restored.append(preds[0])  # original
        restored.append(np.rot90(preds[1], k=3, axes=(2, 3)))
        restored.append(np.rot90(preds[2], k=2, axes=(2, 3)))
        restored.append(np.rot90(preds[3], k=1, axes=(2, 3)))

        restored = np.stack(restored)
        weighted_mean = np.tensordot(weights, restored, axes=1)
        return weighted_mean


    def predict_on_array_cf(self, model, arr, in_shape, out_bands, stride=None,
                            batchsize=64, dtype="float32", augmentation=False,
                            input_name=None):
        C, H, W = in_shape
        stride = stride or H
        arr = arr.astype(dtype)
        height, width, channels = arr.shape

        pad_h = (H - (height - H) % stride) % stride
        pad_w = (W - (width - W) % stride) % stride
        arr_padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

        out_h = (arr_padded.shape[0] - H) // stride + 1
        out_w = (arr_padded.shape[1] - W) // stride + 1

        prediction = np.zeros((out_h * out_w, out_bands, H, W), dtype=dtype)

        batch = []
        coords = []
        k = 0

        for i in range(0, arr_padded.shape[0] - H + 1, stride):
            for j in range(0, arr_padded.shape[1] - W + 1, stride):
                patch = arr_padded[i:i + H, j:j + W, :].transpose(2, 0, 1)  # HWC to CHW
                batch.append(patch)
                coords.append((i, j))
                if len(batch) == batchsize or (i == arr_padded.shape[0] - H and j == arr_padded.shape[1] - W):
                    batch_np = np.stack(batch)
                    if augmentation:
                        aug_batches = self.augmentations_forward(batch_np)
                        aug_preds = []
                        for aug_batch in aug_batches:
                            pred = self.predict_batch_onnx(model, input_name, aug_batch)
                            aug_preds.append(pred)
                        pred = self.reverse_augmentations(aug_preds)
                    else:
                        pred = self.predict_batch_onnx(model, input_name, batch_np)

                    prediction[k:k + len(batch)] = pred
                    k += len(batch)
                    batch = []

        result = np.zeros((arr_padded.shape[0], arr_padded.shape[1], out_bands), dtype=dtype)
        count = np.zeros_like(result)
        k = 0
        for i in range(0, arr_padded.shape[0] - H + 1, stride):
            for j in range(0, arr_padded.shape[1] - W + 1, stride):
                result[i:i + H, j:j + W, :] += prediction[k].transpose(1, 2, 0)
                count[i:i + H, j:j + W, :] += 1
                k += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(result, count, out=np.zeros_like(result), where=(count != 0))

        result = result[:height, :width, :]
        return result.squeeze()
                         

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

        output_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        if source is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.INPUT)
            )

        crs = source.crs().horizontalCrs()
        if not crs.isValid():
            raise QgsProcessingException("Invalid CRS in input raster")

        file_path = source.source()

        feedback.pushInfo(f"Using CRS: {crs.description()}")

        model_path = os.path.join(os.path.dirname(__file__), MODEL_NAME)

        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name

        with rasterio.open(file_path) as src:
            img = src.read().transpose(1, 2, 0)  # HWC
            meta = src.meta.copy()

        pred = self.predict_on_array_cf(
            session,
            img,
            in_shape=MODEL_CONFIG['in_shape'],
            out_bands=MODEL_CONFIG['out_bands'],
            stride=MODEL_CONFIG['stride'],
            augmentation=MODEL_CONFIG['augmentation'],
            batchsize=MODEL_CONFIG['batch_size'],
            input_name=input_name
        )

        meta.update({
            'count': 1,
            'dtype': 'float32'
        })

        
        with rasterio.open(output_raster, 'w', **meta) as dst:
            dst.write(pred.astype(np.float32), 1)

        raster_out = {'OUTPUT': output_raster}

        feedback.pushInfo(f"Output raster created: {output_raster}")

        return {self.OUTPUT: raster_out['OUTPUT']}

    def createInstance(self):
        return self.__class__()