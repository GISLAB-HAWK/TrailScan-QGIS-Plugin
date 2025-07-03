import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import pickle
import os
import osgeo.gdal as gdal
import osgeo.gdal_array as gdn
from osgeo import osr


def get_map_extent(gdal_raster):
    """Returns a dict of {xmin, xmax, ymin, ymax, xres, yres} of a given GDAL raster file.
    Returns None if no geo reference was found.
    Args:
        gdal_raster: File opened via gdal.Open().
    """
    xmin, xres, xskew, ymax, yskew, yres = gdal_raster.GetGeoTransform()
    xmax = xmin + (gdal_raster.RasterXSize * xres)
    ymin = ymax + (gdal_raster.RasterYSize * yres)
    # ret = ( (ymin, ymax), (xmin, xmax) )
    ret = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "xres": xres, "yres": yres}
    if 0. in (ymin, ymax, xmin, xmax): return None  # This is true if no real geodata is referenced.
    return ret


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
        raise RuntimeError("Input file does not exist. Given path: {}".format(input_file))

    ds = gdal.Open(input_file)
    extent = get_map_extent(ds)

    if band_mapping is None:
        num_bands = ds.RasterCount
        band_mapping = {i + 1: i for i in range(num_bands)}
    elif isinstance(band_mapping, dict):
        num_bands = len(band_mapping)
    else:
        raise TypeError("band_mapping must be a dict, not {}.".format(type(band_mapping)))

    arr = np.empty((num_bands, ds.RasterYSize, ds.RasterXSize), dtype=dtype)

    for source_layer, target_layer in band_mapping.items():
        arr[target_layer] = gdn.BandReadAsArray(ds.GetRasterBand(source_layer))

    if dim_ordering == "HWC":
        arr = np.transpose(arr, (1, 2, 0))  # Reorders dimensions, so that channels are last
    elif dim_ordering == "CHW":
        pass
    else:
        raise ValueError("Dim ordering {} not supported. Choose one of 'HWC' or 'CHW'.".format(dim_ordering))

    if return_extent:
        return arr, extent
    else:
        return arr


def normalize_percentile(data, low=1, high=99, nodata_value=0):
    """Normalizes data by applying a percentile cut stretch bandwise.

    Args:
        data (np.ndarray): The data to be normalized
        low: The low cut value
        high: The high cut value
        nodata_value: The no data value
    """
    # first, find areas where we have data
    datamask = data != nodata_value
    # calculate the low and high percentile values bandwise within the valid data area
    pmin = np.array([np.percentile(data[:, :, i][datamask[:, :, i]], q=low) for i in range(data.shape[-1])])
    pmax = np.array([np.percentile(data[:, :, i][datamask[:, :, i]], q=high) for i in range(data.shape[-1])])
    # stretch and clip
    res = np.clip((data - pmin) / (pmax - pmin + 1E-10), 0, 1)
    return res


# %%
# the data_types define the 4 types of terrain- and vegetation models derived from ALS data
data_types = ["DTM", "CHM", "LRM", "VDI"]
# if the data naming is DTM_locationname, CHM_locationname, LRM_locationname, VDI_locationname, enter location_names in
## brackets:
location_names = []
# read all data into a list of lists
# the inner lists contain the different data types and the outer list runs over different locations
data = [[np.nan_to_num(read_img("data/raw_data/{}_{}.tif".format(loc, t))) for t in data_types] for
        loc in location_names]

# now we stack the data for each location so that we have only one image containing 4 channels per site
# the resulting image is of size height, width, number of channels (here 4 (DTM, CHM, LRM, VDI)
data = [np.concatenate(loc, axis=2) for loc in data]


# %%
# for testing the model, a normalized tiff needs to be created
# first, define a function "array to tif"
def array_to_tif(array, dst_filename, num_bands='multi', save_background=True, src_raster: str = "", transform=None,
                 crs=None):
    """ Takes a numpy array and writes a tif. Uses deflate compression.

    Args:
        array: numpy array
        dst_filename (str): Destination file name/path
        num_bands (str): 'single' or 'multi'. If 'single' is chosen, everything is saved into one layer. The values
            in each layer of the input array are multiplied with the layer index and summed up. This is suitable for
            mutually exclusive categorical labels or single layer arrays. 'multi' is for normal images.
        save_background (bool): Whether or not to save the last layer, which is often the background class.
            Set to `True` for normal images.
        src_raster (str): Raster file used to determine the corner coords.
        transform: A geotransform in the gdal format
        crs: A coordinate reference system as proj4 string
    """
    if src_raster != "":
        src_raster = gdal.Open(src_raster)
        x_pixels = src_raster.RasterXSize
        y_pixels = src_raster.RasterYSize
    elif transform is not None and crs is not None:
        y_pixels, x_pixels = array.shape[:2]
    else:
        raise RuntimeError("Please provide either a source raster file or geotransform and coordinate reference "
                           "system.")

    bands = min(array.shape) if array.ndim == 3 else 1
    if not save_background and array.ndim == 3: bands -= 1

    driver = gdal.GetDriverByName('GTiff')

    datatype = str(array.dtype)
    datatype_mapping = {'byte': gdal.GDT_Byte, 'uint8': gdal.GDT_Byte, 'uint16': gdal.GDT_UInt16,
                        'uint32': gdal.GDT_UInt32, 'int8': gdal.GDT_Byte, 'int16': gdal.GDT_Int16,
                        'int32': gdal.GDT_Int32, 'float16': gdal.GDT_Float32, 'float32': gdal.GDT_Float32}
    options = ["COMPRESS=DEFLATE"]
    if datatype == "float16":
        options.append("NBITS=16")

    out = driver.Create(
        dst_filename,
        x_pixels,
        y_pixels,
        1 if num_bands == 'single' else bands,
        datatype_mapping[datatype],
        options=options)

    if src_raster != "":
        out.SetGeoTransform(src_raster.GetGeoTransform())
        out.SetProjection(src_raster.GetProjection())
    else:
        out.SetGeoTransform(transform)
        srs = osr.SpatialReference()
        srs.ImportFromProj4(crs)
        out.SetProjection(srs.ExportToWkt())

    if array.ndim == 2:
        out.GetRasterBand(1).WriteArray(array)
        out.GetRasterBand(1).SetNoDataValue(0)
    else:
        if num_bands == 'single':
            singleband = np.zeros(array.shape[:2], dtype=array.dtype)
            for i in range(bands):
                singleband += (i + 1) * array[:, :, i]
            out.GetRasterBand(1).WriteArray(singleband)
            out.GetRasterBand(1).SetNoDataValue(0)

        elif num_bands == 'multi':
            for i in range(bands):
                out.GetRasterBand(i + 1).WriteArray(array[:, :, i])
                out.GetRasterBand(i + 1).SetNoDataValue(0)

    out.FlushCache()  # Write to disk.


for (d, loc) in zip(data, location_names):
    array_to_tif(normalize_percentile(d).astype(np.float32), f"/data/normalized/{loc}.tif",
                 src_raster=f"/data/raw_data/{loc}_DTM.tif")

# %%
labels = [np.nan_to_num(read_img("/data/raw_data/{}_Trails.tif".format(loc))) for loc in
          location_names]

# %%
# [:2] to only compare the y and x value
for t, l in zip(data, labels):
    assert t.shape[:2] == l.shape[:2]

# %%
width = 224

# tiles
for name, img, label in zip(location_names, data, labels):
    normalized_data = normalize_percentile(img)
    for i, y in enumerate(range(0, img.shape[0], width)):
        for j, x in enumerate(range(0, img.shape[1], width)):
            crop = normalized_data[y:y + width, x:x + width]
            # do not create tiles which are empty and continue:
            if np.all(crop[:, :, 0] == 0):
                continue
            label_crop = label[y:y + width, x:x + width]
            assert crop.shape[:2] == label_crop.shape[:2]
            crop = np.pad(crop, ((0, width - crop.shape[0]), (0, width - crop.shape[1]), (0, 0)))
            label_crop = np.pad(label_crop,
                                ((0, width - label_crop.shape[0]), (0, width - label_crop.shape[1]), (0, 0)))

            # use pickle to write data, which works easier than i.e. a tif
            # for image (wb is only a pickle-specific binary parameter:
            with open("/data/tiles/{}_{}_{}.pickle".format(name, i, j), 'wb') as f:
                pickle.dump(crop, f)
            # # # for label (wb is only a pickle-specific binary parameter:
            with open("/data/labels/{}_{}_{}.pickle".format(name, i, j), 'wb') as f:
                pickle.dump(label_crop, f)

# %%
