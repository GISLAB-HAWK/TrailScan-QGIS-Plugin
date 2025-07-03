import numpy as np
import rasterio
import onnxruntime as ort
from pathlib import Path

MODEL_PATH = r"F:\TrailScan\08_Plugin\Models\TrailScan.onnx"
MODEL_CONFIG = {
    'in_shape': (4, 448, 448),  # Channels, Height, Width
    'out_bands': 1,
    'stride': 224,
    'augmentation': True,
    'batch_size': 4,
    'tile_size': 256,
    'overlap': 32
}


def predict_batch_onnx(session, input_name, batch: np.ndarray) -> np.ndarray:
    inputs = {input_name: batch.astype(np.float32)}
    outputs = session.run(None, inputs)
    return outputs[0]  # Shape: (N, C, H, W)


def augmentations_forward(batch: np.ndarray) -> list[np.ndarray]:
    # Nur 4 Varianten: original + 3 Rotationen
    aug_batches = []
    aug_batches.append(batch)  # original
    aug_batches.append(np.rot90(batch, k=1, axes=(2, 3)).copy())
    aug_batches.append(np.rot90(batch, k=2, axes=(2, 3)).copy())
    aug_batches.append(np.rot90(batch, k=3, axes=(2, 3)).copy())
    return aug_batches


def reverse_augmentations(preds: list[np.ndarray]) -> np.ndarray:
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


def predict_on_array_cf(model, arr, in_shape, out_bands, stride=None,
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
                    aug_batches = augmentations_forward(batch_np)
                    aug_preds = []
                    for aug_batch in aug_batches:
                        pred = predict_batch_onnx(model, input_name, aug_batch)
                        aug_preds.append(pred)
                    pred = reverse_augmentations(aug_preds)
                else:
                    pred = predict_batch_onnx(model, input_name, batch_np)

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


def main():
    session = ort.InferenceSession(str(MODEL_PATH), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    img_path = Path(r"F:\TrailScan\02_Daten\04_NS_Solling\2Solling_normalized.tif")
    with rasterio.open(img_path) as src:
        img = src.read().transpose(1, 2, 0)  # HWC
        meta = src.meta.copy()

    pred = predict_on_array_cf(
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
    out_path = Path(r"F:\TrailScan\02_Daten\04_NS_Solling\2Solling_result.tif")
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(pred.astype(np.float32), 1)


if __name__ == "__main__":
    main()

#end