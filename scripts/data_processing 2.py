import xarray
import numpy as np
import geoprocessing
import pandas as pd
import os
from shapely.geometry import box, mapping
import geopandas as gpd


def calculate_dsi(wl, dr):
    """
    Calculate the Double Stress Index (DSI) for a given pixel based on its waterlogging (wl) and drought (dr)
    sum exceedance values (SEV).

    The DSI is a metric that quantifies the combined stress from waterlogging and drought in a single index value.
    Higher values indicate higher combined stress, while a value of 0 indicates no stress.

    Args:
    ----------
    wl (float):
        Waterlogging SEV value for the pixel.
    dr (float):
        Drought SEV value for the pixel.

    Returns:
    -------
    float:
        The Double Stress Index (DSI) for the given pixel.
    """
    if wl == 0 and dr == 0:
        return 0
    else:
        return 2 * ((wl * dr) / (wl + dr))


def classify_dsi(dsi):
    """
    Classify the Double Stress Index (DSI) into stress level categories.

    This function takes a DSI value as input and returns an integer value
    representing the stress level category as follows:
    1 - No Stress
    2 - Weak Double Stress
    3 - Strong Double Stress

    Args:
    ----------
    dsi (float):
        Double Stress Index (DSI) value for the pixel.

    Returns:
    -------
    int:
        An integer value representing the stress level category.
    """
    if dsi == 0:
        return 1
    elif 0 < dsi <= 4:
        return 2
    else:
        return 3


def calculate_sev(data, threshold, type):
    """
    Calculates the sum exceedance value (SEV) of a variable, based on the Silvertown (1999) paper.
    For the Mattos et al. (2023) paper, the SEV is the number of months a variable s above or below a certain threshold.

    Args:
    ----------
      data (xarray.Dataset or xarray.DataArray):
          A Xarray object containing either water table depth or flooding height data
      threshold (float):
          The threshold value.
      type (str):
          The type of SEV, can be "Drought", "Waterlogging" or "Flooding"

    Returns:
    -------
      SEV (int):
          The number of months the variable is above or below the threshold.

    Raises:
    -------
        ValueError:
            If the type is not valid.
    """

    # Check if the severity is valid.
    if type not in ["Drought", "Waterlogging", "Flooding"]:
        raise ValueError("Invalid severity.")

    # Get the variable name.
    var_name = {"drought": "WTD", "waterlogging": "WTD", "flooding": "FLOODINGHEIGHT"}[
        type
    ]

    # Get the months where the variable is above or below the threshold.
    months = (
        data[var_name] > threshold if type == "drought" else data[var_name] < threshold
    )

    # Count the number of months.
    SEV = months.count()

    # Return the number of months.
    return SEV


def create_random_sample(data, sample_size, columns, seed=100294, output_file=None):
    """
    Create a random sample of a data frame and optionally save it to a file.

    Args:
    ----------
        data (pd.DataFrame):
            Input data frame to be sampled.
        sample_size (int):
            Number of samples to be drawn from the input data frame.
        columns (list):
            List of column names to be included in the output data frame.
        seed (int, optional):
            Seed for the random number generator. Default is 100294.
        output_file (str, optional):
            Path to the output file. If provided, the sampled data frame will be saved as a CSV file.
            Default is None.

    Returns:
    -------
        pd.DataFrame:
            Sampled data frame.
    """

    rng = np.random.RandomState(seed)  # Set seed for subset
    ind = rng.choice(len(data), size=sample_size, replace=False)  # Get subset indexes

    # Create a dictionary with sampled data for each column
    sampled_data = {col: data[col].values[ind] for col in columns}

    # Store in a data frame
    sampled_df = pd.DataFrame(sampled_data)

    # Save to a CSV file if output_file is provided
    if output_file is not None:
        sampled_df.to_csv(output_file, index=False)

    return sampled_df


def process_data(elev_file, pp_file, tree_file, count_file, sev_file):
    """
    Process input data for Mattos et al. (2023) PNAS and return a masked and processed DataFrame.

    This function reads elevation, precipitation, tree cover, valid tree coverpixel count,
    and sum exceedance values (SEV) files, applies masks and transformations, and returns
    a DataFrame with relevant data for further analysis.

    Args:
    -------
    elev_file (str):
        Path to the elevation file.
    pp_file (str):
        Path to the precipitation file.
    tree_file (str):
        Path to the tree cover file.
    count_file (str):
        Path to valid tree cover pixel count file.
    sev_file (str):
        Path to SEV file.

    Returns:
    -------
    pandas.DataFrame:
        A masked and processed DataFrame containing data for further analysis.
    """

    # Read elevation dataset and create elevation mask
    elev = xarray.open_dataset(elev_file)
    elev_mask = np.logical_and(
        elev.DEM.values[::-1] > -100, elev.DEM.values[::-1] < 1200
    )

    # Read and process precipitation dataset
    pp = geoprocessing.read_and_clip_data_to_box(
        pp_file, bounds="tropical_South_America"
    )

    # Read and process tree cover and count datasets
    tree = geoprocessing.read_and_clip_data_to_box(
        tree_file, bounds="tropical_South_America"
    )
    count = geoprocessing.read_and_clip_data_to_box(
        count_file, bounds="tropical_South_America"
    )

    # Create tree cover mask
    tc_mask = count.TC.values < 11
    tree.TC.values[tc_mask] = np.nan
    tc_mask = ~np.isnan(tree.TC.values)

    # Read and process SEV data
    data = xarray.open_dataset(sev_file)

    # Combine elevation and tree cover masks
    maskfinal = np.logical_and(elev_mask, tc_mask)

    # Create and process mask dataset
    mask = xarray.Dataset(
        {"MASK": (["lat", "lon"], maskfinal)},
        coords={"lon": (["lon"], tree.lon.values), "lat": (["lat"], tree.lat.values)},
    )
    mask.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    mask.rio.write_crs("epsg:4326", inplace=True)

    # Process input data
    maskvals = mask.MASK.values
    ppvals = pp.MAP.values * 30 * 1000
    wl_vals = np.around(data.WL_SEV.values / 15)
    dr_vals = np.around(data.DR_SEV.values / 15)

    # Calculate areas of grid cells
    lats = np.tile(np.array([tree.lat.values]).transpose(), (1, tree.lon.size))
    areas = geoprocessing.area_of_pixel(lats)

    # Create 2D arrays of latitude and longitude for each grid cell
    lon_grid, lat_grid = np.meshgrid(tree.lon.values, tree.lat.values)

    # Flatten and filter data
    mask_f = maskvals.flatten()
    mask_f[
        ppvals.flatten() > 6000
    ] = False  # Excluding too high P, possible artifact in ERA5?
    indices = np.nonzero(mask_f)

    dataframe = pd.DataFrame(
        {
            "TC": tree.TC.values.flatten()[indices],
            "WL_SEV": wl_vals.flatten()[indices],
            "DR_SEV": dr_vals.flatten()[indices],
            "AREA": areas.flatten()[indices],
            "MAP": ppvals.flatten()[indices],
            "LAT": lat_grid.flatten()[indices],
            "LON": lon_grid.flatten()[indices],
        }
    )

    # Classify the data
    dataframe["CLASS"] = "Savanna"
    dataframe.loc[dataframe["TC"] >= 60, "CLASS"] = "Forest"

    dataframe["DSI"] = dataframe.apply(
        lambda x: calculate_dsi(x["WL_SEV"], x["DR_SEV"]), axis=1
    )
    dataframe["DS_CLASS"] = dataframe["DSI"].apply(classify_dsi)

    return dataframe


def create_mask(elev_file, pp_file, tree_file, count_file):
    """
    Creates a mask of South America based on elevation, precipitation, tree cover, and water deficit.

    Args:
    -------
      elev_file (str):
          Path to the elevation file.
      pp_file (str):
          Path to the precipitation file.
      tree_file (str):
          Path to the tree cover file.
      count_file (str):
          Path to valid tree cover pixel count file.

    Returns:
    -------
      boolean:
          Mask of valid pixels.
    """

    # Read the elevation dataset.
    elev = xarray.open_dataset(elev_file)

    # Create an elevation mask.
    elev_mask = np.logical_and(
        elev.DEM.values[::-1] > -100, elev.DEM.values[::-1] < 1200
    )

    # Read the precipitation dataset.
    pp = geoprocessing.read_and_clip_data_to_box(
        pp_file, bounds="tropical_South_America"
    )

    # Read the tree cover dataset.
    tree = geoprocessing.read_and_clip_data_to_box(
        tree_file, bounds="tropical_South_America"
    )

    # Read the tree count dataset.
    count = geoprocessing.read_and_clip_data_to_box(
        count_file, bounds="tropical_South_America"
    )

    # Create a tree cover mask.
    tc_mask = count.TC.values < 11
    tree.TC.values[tc_mask] = np.nan
    tc_mask = ~np.isnan(tree.TC.values)
    tc_mask = np.logical_and((pp.MAP.values * 30 * 1000) < 6000, tc_mask)
    tc_mask = np.logical_and(tc_mask, elev_mask)

    # Return the mask.
    return tc_mask


def calculate_pp_wtd(shapefile_paths, data, wtd_concat):
    """
    The function `calculate_pp_wtd` calculates the mean monthly precipitation and water table depth for
    a list of regions specified by shapefiles. The function processes the input data, crops, and masks
    it according to the study area, then computes the mean values.

    Parameters
    ----------
    shapefile_paths : list
        A list of shapefile paths. Each shapefile represents a distinct region for which the mean
        monthly precipitation and water table depth will be calculated.

    data : xarray.Dataset
        An xarray.Dataset containing the precipitation data (variable "tp"). The dataset should have dimensions
        "latitude", "longitude", and "time" (with a monthly frequency).

    wtd_concat : xarray.Dataset
        An xarray.Dataset containing the water table depth data (variable "WTD"). The dataset should have dimensions
        "lat", "lon", and "time" (with a monthly frequency).

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame with the following columns:
        - 'shapefile': Contains the file paths of the shapefiles. Each shapefile represents a distinct region.
        - 'pp_mean': Contains the mean monthly precipitation data for the corresponding region.
        - 'wtd_mean': Contains the mean monthly water table depth data for the corresponding region.
    """

    # Initialize an empty DataFrame
    df = pd.DataFrame(columns=["shapefile", "pp_mean", "wtd_mean"])

    for shapefile_path in shapefile_paths:
        # Get the name of the shapefile
        name = os.path.splitext(os.path.basename(shapefile_path))[0]

        # Read the shapefile
        shape_gpd = gpd.read_file(shapefile_path)

        # Crop and mask input data to the extent of the study area
        bounds = shape_gpd.total_bounds
        data_clipped = data.rio.clip_box(
            minx=bounds[0], maxx=bounds[2], miny=bounds[1], maxy=bounds[3]
        )
        data_masked = data_clipped.rio.clip(
            shape_gpd.geometry.apply(mapping), shape_gpd.crs, drop=False
        )

        # Calculate monthly mean precipitation for all the pixels
        pp_monthly = data_masked.tp.groupby("time.month").mean()
        pp_mean = pp_monthly.mean(dim=["latitude", "longitude"]) * 1000 * 30

        # Crop and mask wtd_concat data to the extent of the study area
        wtd_clipped = wtd_concat.rio.clip_box(
            minx=bounds[0], maxx=bounds[2], miny=bounds[1], maxy=bounds[3]
        )
        wtd_masked = wtd_clipped.rio.clip(
            shape_gpd.geometry.apply(mapping), shape_gpd.crs, drop=False
        )

        # Calculate monthly mean water table depth for all the pixels
        wtd_monthly = wtd_masked.WTD.groupby("time.month").mean()
        wtd_min = wtd_monthly.max(dim="month")

        # Create a waterlogged mask
        mask = wtd_min >= -0.25

        # Mask out all non-waterlogged pixels
        wtd_monthly_masked = wtd_monthly.where(mask.values)

        # Calculate monthly means
        wtd_mean = wtd_monthly_masked.mean(dim=["lat", "lon"])

        # Calculate monthly mean water table depth for all the pixels
        # wtd_monthly = wtd_masked.WTD.groupby("time.month").mean()
        # wtd_monthly_masked = wtd_monthly.where(wtd_monthly >= -0.25)
        # wtd_mean = wtd_monthly_masked.mean(dim=['lat', 'lon'])

        # Append the results to the DataFrame
        df = df.append(
            {"shapefile": name, "pp_mean": pp_mean.values, "wtd_mean": wtd_mean.values},
            ignore_index=True,
        )

    return df


def raster_dsi_class(netcdf_file):
    """
    Generate a raster of DSI classes from a given netcdf file.

    Args:
    ----------
    netcdf_file (str):
        Path to the netcdf file.

    Returns:
    -------
    xarray.Dataset:
        A dataset containing the DSI classes.
    """
    ds = xarray.open_dataset(netcdf_file)
    wl_sev = ds["WL_SEV"].load()
    dr_sev = ds["DR_SEV"].load()

    # Calculate the DSI for each pixel using vectorized operations
    dsi = np.where(
        (wl_sev != 0) & (dr_sev != 0), 2 * ((wl_sev * dr_sev) / (wl_sev + dr_sev)), 0
    )

    # Vectorize the classify_dsi function so it can accept array inputs
    vectorized_classify_dsi = np.vectorize(classify_dsi)

    # Classify the DSI for each pixel using vectorized operations
    dsi_classes = vectorized_classify_dsi(dsi)

    # Create a new dataset with the dsi_classes array
    ds_classes = xarray.Dataset(
        {"dsi_classes": (("lat", "lon"), dsi_classes)},
        coords={"lat": ds["lat"], "lon": ds["lon"]},
    )

    return ds_classes


def calculate_forest_areas(tree_cover_file, count_file, mask, elevation_file):
    """
    Calculates the total forest area and the forest area that will be affected by double stress in the future.

    Args:
    ----------
    tree_cover_file : str
        Path to the TreeCover netCDF file.
    count_file : str
        Path to the Count netCDF file.
    mask : xarray.DataArray
        A boolean mask indicating the areas that will be affected by double stress in the future.
    elevation_file : str
        Path to the elevation netCDF file.

    Returns:
    -------
    tuple
        A tuple containing the total forest area (in square kilometers) and the forest area that will be affected by double stress in the future (in square kilometers).
    """

    # Load the TreeCover and Count netCDF files as xarray datasets
    ds_tc = xarray.open_dataset(tree_cover_file)
    ds_count = xarray.open_dataset(count_file)

    # Load the elevation netCDF file as an xarray dataset
    ds_elev = xarray.open_dataset(elevation_file)

    # Find the valid elevation areas
    valid_elevations = np.logical_and(
        ds_elev["ELEV"].values > -100, ds_elev["ELEV"].values < 1200
    )

    # Update the mask to include valid elevations
    updated_mask = mask & valid_elevations

    # First create a mxn array with the latitudes (where m and n are dimensions of input data)
    # lat = np.tile((mask2km.lat.values),(np.shape(mask2km)[1],1)).transpose()
    lats = np.tile(np.array([ds_tc["lat"].values]), (ds_tc["lon"].size, 1)).transpose()

    # Now get an area array
    pixel_areas = xarray.apply_ufunc(area_of_pixel, lats, vectorize=True)

    # Mask out invalid TC values
    countvals = ds_count.TC.values
    valid_treevals = countvals >= 44

    # Mask out invalid tree cover pixels from the pixel areas
    pixel_areas = np.where(valid_treevals, pixel_areas, np.nan)

    # Calculate the total forest area (where Tree Cover > 60% and valid elevation)
    total_forest_area = (
        pixel_areas * ds_tc["TC"].where((ds_tc["TC"] >= 60) & valid_elevations)
    ).sum().values / 1e6

    # Calculate the forest area that will be affected by double stress in the future (valid elevation)
    future_double_stress_area = (
        pixel_areas * ds_tc["TC"].where(updated_mask & (ds_tc["TC"] >= 60))
    ).sum().values / 1e6

    return total_forest_area, future_double_stress_area


def get_low_forest_bins(df):
    """
    Calculates in what bins the forest proportion is equal to or lower than 37.5% and returns
    a list of WL_SEV and DR_SEV bins that satisfy that condition.

    Args:
    ----------
    df : pd.DataFrame
        DataFrame containing at least drought SEV (DR_SEV), waterlogging SEV (WL_SEV), area (AREA),
        and vegetation class (CLASS) information.

    Returns:
    -------
    list
        List of tuples, each containing the DR_SEV and WL_SEV bin indices where forest proportion
        is equal to or less than 37.5%
    """

    # Calculate statistics for the whole area
    stat, x_edge, y_edge, binnumber = binned_statistic_2d(
        df["DR_SEV"],
        df["WL_SEV"],
        df["AREA"],
        statistic="sum",
        bins=np.arange(0, 14, 1),
    )

    # Filter the DataFrame to include only forest areas
    forest_df = df[df["CLASS"] == "Forest"]

    # Calculate statistics for the forest area
    stat_for, x_edge, y_edge, binnumber = binned_statistic_2d(
        forest_df["DR_SEV"],
        forest_df["WL_SEV"],
        forest_df["AREA"],
        statistic="sum",
        bins=np.arange(0, 14, 1),
    )

    # Calculate the fraction of area occupied by forests for each bin
    forest_fraction = (stat_for / stat) * 100

    # Identify the bins where the forest fraction is equal to or less than 37.5%
    low_forest_bins = np.argwhere(forest_fraction <= 37.5)

    return [(x_edge[i], y_edge[j]) for i, j in low_forest_bins]
