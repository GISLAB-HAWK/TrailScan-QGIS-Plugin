import os
import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import time
from sys import stdout
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
MODEL_PATH = Path(r"F:\TrailScan\04_Models\TrailScan.pt")
INPUT_PATH = Path(r"F:\TrailScan\02_Daten\04_NS_Solling\2Solling_normalized.tif")
OUTPUT_PATH = Path(r"F:\TrailScan\02_Daten\04_NS_Solling\2Solling_result.tif")

# Model configuration
MODEL_CONFIG = {
    'in_shape': (4, 448, 448),  # Number of input bands and input size
    'out_bands': 1,  # Number of output bands
    'stride': 224,  # Stride for prediction
    'augmentation': True,  # Whether to use augmentation during prediction
    'batch_size': 4,
    'tile_size': 256,
    'overlap': 32
}

def read_img(input_file, dim_ordering="HWC", dtype='float32', band_mapping=None, return_extent=False):
    """Reads an image from disk and returns it as numpy array.

    Args:
        input_file: Path to the input file.
        dim_ordering: One of HWC or CHW, C=Channels, H=Height, W=Width
        dtype: Desired data type for loading, e.g. np.uint8, np.float32...
        band_mapping: Dictionary of which image band to load into which array band. E.g. {1:0, 3:1}
        return_extent: Whether or not to return the raster extent in the form (ymin, ymax, xmin, xmax). Defaults to False.

    Returns:
        Numpy array containing the image and optionally the extent.
    """
    if not os.path.isfile(input_file):
        raise RuntimeError(f"Input file does not exist. Given path: {input_file}")

    with rasterio.open(input_file) as src:
        if band_mapping is None:
            num_bands = src.count
            band_mapping = {i + 1: i for i in range(num_bands)}
        elif isinstance(band_mapping, dict):
            num_bands = len(band_mapping)
        else:
            raise TypeError(f"band_mapping must be a dict, not {type(band_mapping)}.")

        arr = np.empty((num_bands, src.height, src.width), dtype=dtype)

        for source_layer, target_layer in band_mapping.items():
            arr[target_layer] = src.read(source_layer)

        if dim_ordering == "HWC":
            arr = np.transpose(arr, (1, 2, 0))  # Reorders dimensions, so that channels are last
        elif dim_ordering == "CHW":
            pass
        else:
            raise ValueError(f"Dim ordering {dim_ordering} not supported. Choose one of 'HWC' or 'CHW'.")

        if return_extent:
            bounds = src.bounds
            extent = {
                "xmin": bounds.left,
                "xmax": bounds.right,
                "ymin": bounds.bottom,
                "ymax": bounds.top,
                "xres": src.res[0],
                "yres": src.res[1]
            }
            return arr, extent
        else:
            return arr

def array_to_tif(array, dst_filename, num_bands='multi', save_background=True, src_raster: str = "", transform=None,
                 crs=None):
    """Takes a numpy array and writes a tif. Uses deflate compression.

    Args:
        array: numpy array
        dst_filename (str): Destination file name/path
        num_bands (str): 'single' or 'multi'. If 'single' is chosen, everything is saved into one layer.
        save_background (bool): Whether or not to save the last layer, which is often the background class.
        src_raster (str): Raster file used to determine the corner coords.
        transform: A geotransform in the gdal format
        crs: A coordinate reference system as proj4 string
    """
    if src_raster:
        with rasterio.open(src_raster) as src:
            profile = src.profile
            transform = src.transform
            crs = src.crs
    elif transform is not None and crs is not None:
        profile = {
            'driver': 'GTiff',
            'dtype': str(array.dtype),
            'count': 1 if num_bands == 'single' else (min(array.shape) if array.ndim == 3 else 1),
            'height': array.shape[0],
            'width': array.shape[1],
            'transform': transform,
            'crs': crs,
            'compress': 'deflate'
        }
    else:
        raise RuntimeError("Please provide either a source raster file or geotransform and coordinate reference system.")

    if array.ndim == 2:
        profile['count'] = 1
        with rasterio.open(dst_filename, 'w', **profile) as dst:
            dst.write(array, 1)
            dst.nodata = 0
    else:
        if num_bands == 'single':
            singleband = np.zeros(array.shape[:2], dtype=array.dtype)
            for i in range(min(array.shape)):
                singleband += (i + 1) * array[:, :, i]
            profile['count'] = 1
            with rasterio.open(dst_filename, 'w', **profile) as dst:
                dst.write(singleband, 1)
                dst.nodata = 0
        elif num_bands == 'multi':
            bands = min(array.shape) if array.ndim == 3 else 1
            if not save_background and array.ndim == 3:
                bands -= 1
            profile['count'] = bands
            with rasterio.open(dst_filename, 'w', **profile) as dst:
                for i in range(bands):
                    dst.write(array[:, :, i], i + 1)
                    dst.nodata = 0

def compute_pyramid_patch_weight_loss(width: int, height: int) -> np.ndarray:
    """Compute a weight matrix that assigns bigger weight on pixels in center and
    less weight to pixels on image boundary.
    This weight matrix is then used for merging individual tile predictions and helps dealing
    with prediction artifacts on tile boundaries.

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

def predict_on_array_cf(model,
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
                       report_time=False,
                       return_data_region=False):
    """
    Applies a pytorch segmentation model to an array in a strided manner.

    Channels first version.

    Call model.eval() before use!

    Args:
        model: pytorch model - make sure to call model.eval() before using this function!
        arr: CHW array for which the segmentation should be created
        stride: stride with which the model should be applied. Default: output size
        batchsize: number of images to process in parallel
        dtype: desired output type (default: float32)
        augmentation: whether to average over rotations and mirrorings of the image or not. triples computation time.
        no_data: a no-data vector. its length must match the number of layers in the input array.
        verbose: whether or not to display progress
        report_time: if true, returns (result, execution time)

    Returns:
        An array containing the segmentation.
    """
    t0 = None

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

    assert in_shape[1] == in_shape[2], "Input shape must be equal in last two dims."
    out_shape = (out_bands, in_shape[1] - 2 * drop_border, in_shape[2] - 2 * drop_border)
    in_size = in_shape[1]
    out_size = out_shape[1]
    stride = stride or out_size
    pad = (in_size - out_size) // 2
    assert pad % 2 == 0, "Model input and output shapes have to be divisible by 2."

    original_size = arr.shape
    ymin = 0
    xmin = 0

    if no_data is not None:
        nonzero = np.nonzero(arr[0, :, :] - no_data)
        if len(nonzero[0]) == 0:
            if return_data_region:
                return None, (0, 0, 0, 0)
            else:
                return None
        ymin = np.min(nonzero[0])
        ymax = np.max(nonzero[0])
        xmin = np.min(nonzero[1])
        xmax = np.max(nonzero[1])
        img = arr[:, ymin:ymax, xmin:xmax]
    else:
        img = arr

    weight_mask = compute_pyramid_patch_weight_loss(out_size, out_size)
    final_output = np.zeros((out_bands,) + img.shape[1:], dtype=dtype)

    op_cnt = 0
    for op, inv in zip(operations, inverse):
        img = op(img)
        img_shape = img.shape
        x_tiles = int(np.ceil(img.shape[2] / stride))
        y_tiles = int(np.ceil(img.shape[1] / stride))

        y_range = range(0, (y_tiles + 1) * stride - out_size, stride)
        x_range = range(0, (x_tiles + 1) * stride - out_size, stride)

        y_pad_after = y_range[-1] + in_size - img.shape[1] - pad
        x_pad_after = x_range[-1] + in_size - img.shape[2] - pad

        output = np.zeros((out_bands,) + (img.shape[1] + y_pad_after - pad, img.shape[2] + x_pad_after - pad),
                         dtype=dtype)
        division_mask = np.zeros(output.shape[1:], dtype=dtype) + 1E-7
        img = np.pad(img, ((0, 0), (pad, y_pad_after), (pad, x_pad_after)), mode='reflect')

        patches = len(y_range) * len(x_range)

        def patch_generator():
            for y in y_range:
                for x in x_range:
                    yield img[:, y:y + in_size, x:x + in_size]

        patch_gen = patch_generator()

        y = 0
        x = 0
        patch_idx = 0
        batchsize_ = batchsize

        t0 = time.time()

        while patch_idx < patches:
            batchsize_ = min(batchsize_, patches, patches - patch_idx)
            patch_idx += batchsize_
            if verbose: stdout.write("\r%.2f%%" % (100 * (patch_idx + op_cnt * patches) / (len(operations) * patches)))

            batch = np.zeros((batchsize_,) + in_shape, dtype=dtype)

            for j in range(batchsize_):
                batch[j] = next(patch_gen)

            with torch.no_grad():
                prediction = model(torch.from_numpy(batch).to(device=device, dtype=torch.float32))
                prediction = prediction.detach().cpu().numpy()
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
        img = arr[:, ymin:ymax, xmin:xmax] if no_data is not None else arr
        op_cnt += 1
        if verbose: stdout.write("\rAugmentation step %d/%d done.\n" % (op_cnt, len(operations)))

    if verbose: stdout.flush()

    final_output = final_output / len(operations)

    if no_data is not None:
        final_output = np.pad(final_output,
                             ((0, 0), (ymin, original_size[1] - ymax), (xmin, original_size[2] - xmax)),
                             mode='constant',
                             constant_values=0)

    if report_time:
        return final_output, time.time() - t0
    elif return_data_region:
        return final_output, (ymin, ymax, xmin, xmax)
    else:
        return final_output

def main():
    """Main function to run the inference pipeline."""
    try:
        logger.info("Starting inference pipeline")
        
        # Load model
        model = torch.jit.load(str(MODEL_PATH))
        model.to("cpu")
        model.eval()
        
        # Load input file
        img = read_img(str(INPUT_PATH), dim_ordering="CHW")[[0, 1, 2, 3]]
        
        # Run prediction
        pred = predict_on_array_cf(
            model, 
            img, 
            in_shape=MODEL_CONFIG['in_shape'], 
            out_bands=MODEL_CONFIG['out_bands'], 
            stride=MODEL_CONFIG['stride'], 
            augmentation=MODEL_CONFIG['augmentation']
        )
        
        # Visualize result
        plt.clf()
        plt.imshow(pred[0])
        
        # Save result
        array_to_tif(
            pred.transpose(1,2,0),
            str(OUTPUT_PATH),
            src_raster=str(INPUT_PATH)
        )
        
        logger.info("Inference pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()


