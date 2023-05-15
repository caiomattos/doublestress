import numpy as np
import os
import glob
from osgeo import gdal
import rasterio as rio
from rasterio.merge import merge
import numpy as np
import rasterio as rio
import xarray
from rasterio.warp import reproject
from rasterio import Affine as A
import geopandas as gpd
from shapely.geometry import box, mapping
import rioxarray


def merge_geotiff_files(files, outfile, mask=False, region=None, band=None):
    """
    Merges a list of GeoTIFF files and optionally applies a mask to classify the data.

    Parameters
    ----------
    files: list
        List of GeoTIFF files to be merged.
    outfile: str
        Name of the output GeoTIFF file.
    mask: bool, optional (default: False)
        If True, replaces land cover classes with valid (0) or invalid (1) values.
    region: str, optional (default: None)
        Defines the region to which the land cover classes are defined. Either "Amazon", "Brazil", or "Copernicus".
    band: int, optional (default: None)
        The band number to be used from the input files. If not provided, the first band (1) is used.

    Raises
    ------
    ValueError
        If an unsupported region value is provided.
    """
    # Check for valid region
    if region not in (None, "Amazon", "Brazil", "Copernicus"):
        raise ValueError(
            "Region not supported. Valid values are 'Amazon', 'Brazil', or 'Copernicus'."
        )

    bandval = band if band is not None else 1

    arr, transf = merge(files)
    newtransf = (transf[2], transf[0], transf[1], transf[5], transf[3], transf[4])

    copyarr = np.copy(arr)

    if mask:
        # Get mask based on the region
        if region == "Brazil":
            valid = [1, 3, 4, 5, 49, 10, 11, 12, 32, 29, 13]
        elif region == "Amazon":
            valid = [1, 3, 4, 5, 6, 10, 11, 12, 29, 13]
        elif region == "Copernicus":
            valid = [
                111,
                112,
                113,
                114,
                115,
                116,
                121,
                122,
                123,
                124,
                125,
                126,
                20,
                30,
                90,
                100,
                60,
            ]

        maskvalid = np.isin(copyarr, valid)
        arr[copyarr < 1] = 255
        arr[copyarr >= 1] = 1
        arr[maskvalid] = 0

    ds = gdal.Open(files[0])
    bands, rows, cols = arr.shape
    driver = gdal.GetDriverByName("GTiff")
    band = ds.GetRasterBand(bandval)
    outdata = driver.Create(
        outfile, cols, rows, 1, eType=band.DataType, options=["COMPRESS=LZW"]
    )
    outdata.SetGeoTransform(newtransf)
    outdata.SetProjection(ds.GetProjection())
    outdata.GetRasterBand(1).WriteArray(arr[0, :, :])

    if mask:
        outdata.GetRasterBand(1).SetNoDataValue(255)

    outdata.FlushCache()  # Saves to disk
    outdata = None
    band = None
    ds = None


def reproject_data(
    source_file,
    dest_file,
    file_type="netCDF",
    band=None,
    variable=None,
    dest_res=None,
    grid_file=None,
    src_nodata=None,
    resampling_method=rio.enums.Resampling.average,
):
    """
    Reprojects data from source_file to either a specified grid or a specified resolution.

    Parameters:
    ----------
    source_file (str):
           Path to the file to be reprojected
    dest_file (str):
        Path to save the reprojected data
    file_type (str, optional):
        File type of source_file, either "GeoTIFF" or "netCDF". Default is "netCDF"
    band (int, optional):
        Band number for GeoTIFF files. Required if file_type is "GeoTIFF"
    variable (str, optional):
        Variable name for netCDF files. Required if file_type is "netCDF"
    dest_res (float, optional):
        Desired destination resolution. Required if grid_file is not provided
    grid_file (str, optional):
        Path to the grid file for defining the reprojected data grid. Required if dest_res is not provided
    src_nodata (float, optional):
        No data value for the source file. Default is None
    resampling_method (rio.enums.Resampling, optional):
        Resampling method to use during reprojection. Default is rio.enums.Resampling.average

    Returns:
        None
    """

    # Open source file
    if file_type == "GeoTIFF":
        source_data = xarray.open_rasterio(source_file).sel(band=band).drop("band")
    elif file_type == "netCDF":
        source_data = xarray.open_dataset(source_file).sel(variable=variable)
    else:
        raise ValueError("Invalid file_type. Must be either 'GeoTIFF' or 'netCDF'")

    source = source_data.values
    dim1, dim2 = source_data.dims

    src_transform = A.translation(
        source_data[dim1][0].values, source_data[dim2][0].values
    ) * A.scale(
        source_data[dim1][1].values - source_data[dim1][0].values,
        source_data[dim2][1].values - source_data[dim2][0].values,
    )
    src_crs = {"init": "EPSG:4326"}

    if grid_file is not None:
        # Use provided grid file for defining the reprojected data grid
        input_data = xarray.open_dataset(grid_file)
        destination = np.empty(input_data.shape)
        destination[:] = np.nan
        dest_res = input_data.x[1].values - input_data.x[0].values
    elif dest_res is not None:
        # Calculate the reprojected data grid based on destination resolution
        width = int(
            (source_data[dim1][-1].values - source_data[dim1][0].values) / dest_res
        )
        height = int(
            (source_data[dim2][-1].values - source_data[dim2][0].values) / dest_res
        )
        destination = np.empty((height, width))
        destination[:] = np.nan
    else:
        raise ValueError("Either dest_res or grid_file must be provided")

    dst_transform = A.translation(
        source_data[dim1][0].values, source_data[dim2][0].values
    ) * A.scale(dest_res, -dest_res)
    dest_crs = {"init": "EPSG:4326"}

    # Perform reprojection
    reproject(
        source,
        destination,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=dest_crs,
        esampling=resampling_method,
    )

    # Save reprojected output
    output_name = variable if file_type == "netCDF" else f"BAND{band}"
    out = xarray.Dataset(
        {variable: (["lat", "lon"], destination)},
        coords={
            "lon": (
                ["lon"],
                np.linspace(
                    source_data[dim2][0].values,
                    source_data[dim2][-1].values,
                    destination.shape[1],
                ),
            ),
            "lat": (
                ["lat"],
                np.linspace(
                    source_data[dim1][0].values,
                    source_data[dim1][-1].values,
                    destination.shape[0],
                ),
            ),
        },
    )

    out["lon"].attrs = {"long_name": "longitude", "units": "degrees_E"}
    out["lat"].attrs = {"long_name": "latitude", "units": "degrees_N"}

    if "long_name" in source_data.attrs:
        out[output_name].attrs["long_name"] = source_data.attrs["long_name"]
    else:
        out[output_name].attrs["long_name"] = output_name

    out.to_netcdf(
        dest_file,
        encoding={output_name: {"dtype": "float32", "zlib": True, "complevel": 9}},
    )


def combine_land_cover_products(
    mapbiomas_file: str,
    copernicus_file: str,
    mapbiomas_var: str,
    copernicus_var: str,
    output_file: str,
    nodata_value=None,
):
    """
    Combines MapBiomas and Copernicus land cover products, keeping MapBiomas where it's
    available (due to higher resolution) and using Copernicus elsewhere.

    Parameters:
    ----------
    mapbiomas_file (str):
        Path to MapBiomas land cover product file.
    copernicus_file (str):
        Path to Copernicus land cover product file.
    mapbiomas_var (str):
        Variable name for MapBiomas land cover data.
    copernicus_var (str):
        Variable name for Copernicus land cover data.
    output_file (str):
        Path to the combined output file.
    nodata_value (float, optional):
        No-data value. If not provided, it will be derived from the MapBiomas attrs.

    Returns:
        None
    """
    mapb = xarray.open_dataset(mapbiomas_file)
    cop = xarray.open_dataset(copernicus_file)

    if mapb[mapbiomas_var].shape != cop[copernicus_var].shape:
        raise ValueError("MapBiomas and Copernicus files have different dimensions.")

    if nodata_value is None:
        nodata_value = mapb[mapbiomas_var].attrs.get("_FillValue")

    if nodata_value is None:
        raise ValueError("No-data value not provided nor present in MapBiomas file.")

    newarr = np.copy(cop[copernicus_var].values)

    mapb_mask = mapb[mapbiomas_var].values != nodata_value

    newarr[mapb_mask] = mapb[mapbiomas_var].values[mapb_mask]

    dim1, dim2 = mapb.dims

    out = xarray.Dataset(
        {
            mapbiomas_var: (["lat", "lon"], newarr),
        },
        coords={
            "lon": (["lon"], mapb[dim1].values),
            "lat": (["lat"], mapb[dim2].values),
        },
    )

    out["lon"].attrs = {"long_name": "longitude", "units": "degrees_E"}
    out["lat"].attrs = {"long_name": "latitude", "units": "degrees_N"}

    long_name = mapb[mapbiomas_var].attrs.get("long_name")
    if long_name is None:
        long_name = cop[copernicus_var].attrs.get("long_name")
    if long_name is not None:
        out[mapbiomas_var].attrs = {"long_name": long_name}

    out.to_netcdf(
        output_file,
        encoding={
            mapbiomas_var: {"dtype": "float32", "zlib": True, "complevel": 9},
        },
    )


def area_of_pixel(center_lat, pixel_size=0.00833321):
    """
    Calculate the m^2 area of a WGS84 square pixel.

    Adapted from: https://gis.stackexchange.com/a/127327/2397

    Parameters:
    ----------
        center_lat (float):
            Latitude of the center of the pixel. Note this  value +/- half the `pixel_size` must not exceed 90/-90
            degrees latitude or an invalid area will be calculated.
        pixel_size (float, optional):
            Length of the side of the pixel in degrees. Default is 0.00833321.

    Returns:
        float:
            Area of the square pixel of side length `pixel_size` centered at `center_lat` in m^2.
    """
    a = 6378137  # meters
    b = 6356752.3142  # meters
    e = np.sqrt(1 - (b / a) ** 2)
    area_list = []

    for f in [center_lat + pixel_size / 2, center_lat - pixel_size / 2]:
        zm = 1 - e * np.sin(np.radians(f))
        zp = 1 + e * np.sin(np.radians(f))
        area_list.append(
            np.pi
            * b**2
            * (np.log(zp / zm) / (2 * e) + np.sin(np.radians(f)) / (zp * zm))
        )

    return pixel_size / 360.0 * (area_list[0] - area_list[1])


def read_and_clip_data_to_box(
    data, bounds="tropical_South_America", x_dim="lon", y_dim="lat", crs="epsg:4326"
):
    """
    Read and clip data according to the provided bounding box.

    Parameters:
    ----------
        data (xarray.Dataset or xarray.DataArray):
            Input data to be clipped.
        bounds (str or list, optional):
            Bounding box to clip the data. Can be either "tropical_South_America" (predefined) or a list of
            [minx, miny, maxx, maxy]. Default is "tropical_South_America".
        x_dim (str, optional):
            Name of the x dimension in the input data. Default is "lon".
        y_dim (str, optional):
            Name of the y dimension in the input data. Default is "lat".
        crs (str, optional):
            Coordinate reference system for the bounding box. Default is "epsg:4326".

    Returns:
        xarray.Dataset or xarray.DataArray:
            Clipped input data.
    """

    if bounds == "tropical_South_America":
        bounds_geometry = box(-85, -35, -33, 15)
    elif isinstance(bounds, list) and len(bounds) == 4:
        bounds_geometry = box(bounds[0], bounds[1], bounds[2], bounds[3])
    else:
        raise ValueError(
            "Invalid bounds provided. Please use 'tropical_South_America' or a list of [minx, miny, maxx, maxy]."
        )

    bounds_gdf = gpd.GeoDataFrame(geometry=[bounds_geometry], crs=crs)

    data = xarray.open_dataset(data)
    data.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
    data.rio.write_crs(crs, inplace=True)

    clipped_data = data.rio.clip(
        bounds_gdf.geometry.apply(mapping), bounds_gdf.crs, drop=False
    )

    return clipped_data


def filter_data_by_shapefile(data, shape_path):
    """
    Filter the data to keep only the points inside the shapefile.

    Args:
    ----------
        data (gpd.GeoDataFrame):
            A GeoDataFrame containing the data to be filtered, including columns 'LAT' and 'LON'.
            shape_path (str): The path to the shapefile.

    Returns:
    ----------
        gpd.GeoDataFrame:
            The filtered GeoDataFrame containing only the points inside the shapefile.
    """
    shapefile = gpd.read_file(shape_path)
    data_within_shape = gpd.sjoin(data, shapefile, predicate="within")
    return data_within_shape
