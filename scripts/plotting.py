import numpy as np
import xarray
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import seaborn as sns
from scipy.stats import binned_statistic_2d
from scipy import integrate
import geoprocessing
import data_processing
import matplotlib.ticker as ticker
import os
import geopandas as gpd
import matplotlib.colors as colors


def plot_map(
    data_path,
    variable,
    file_path,
    mask=False,
    elev_path=None,
    pp_path=None,
    tree_path=None,
    count_path=None,
    extent=[-85, -33, -35, 15],
    plot_shapes=False,
    shape_list=None,
    dpi=1200,
    cmap="viridis",
    vmin=None,
    vmax=None,
):
    """
    Plots a map of a dataset.

    Args:
    ----------
        data_path (str):
            Path to the dataset to plot.
        variable (str):
            Name of the variable to plot.
        file_path (str):
            Path to save the output plot image.
        mask (bool, optional):
            If True, a mask will be applied to the data. Default is False.
        elev_path (str, optional):
            Path to the elevation dataset. Required if mask is True.
        pp_path (str, optional):
            Path to the precipitation dataset. Required if mask is True.
        tree_path (str, optional):
            Path to the tree cover dataset. Required if mask is True.
        count_path (str, optional):
            Path to the tree count dataset. Required if mask is True.
        extent (list, optional):
            The extent of the plot as a list of four floats (xmin, xmax, ymin, ymax).
            Default is [-85, -33, -35, 15] which is tropical South America.
        plot_shapes (bool, optional):
            If True, shapefiles will be plotted. Default is False.
        shape_list (list of str, optional):
            List of paths to shapefiles to plot. Required if plot_shapes is True.
        dpi (int, optional):
            Resolution of the output file. Default is 1200.
        cmap (str, optional):
            Colormap to use for the plot. Default is 'viridis'.
        vmin (float, optional):
            Minimum value for colormap. Default is None.
        vmax (float, optional):
            Maximum value for colormap. Default is None.

    Raises:
    ----------
        ValueError:
            If mask is True but paths to elevation, precipitation, tree cover, or tree count datasets are not provided.
        ValueError:
            If extent is provided but it does not contain exactly four float values.
        ValueError:
            If plot_shapes is True but shape_list is not provided or is None.

    Returns:
        None. The function saves the plot to the file specified by file_path.
    """

    if mask and (
        elev_path is None or pp_path is None or tree_path is None or count_path is None
    ):
        raise ValueError(
            "If mask is True, the paths to the elevation, precipitation, tree cover, and tree count datasets must be provided."
        )

    if extent is not None and len(extent) != 4:
        raise ValueError("The extent must be a list of four floats.")

    if plot_shapes and shape_list is None:
        raise ValueError(
            "If plot_shapes is True, a list with shapefile paths must be provided."
        )

    data = xarray.open_dataset(data_path)

    fig, ax = plt.subplots(
        figsize=[8, 5], subplot_kw={"projection": ccrs.PlateCarree()}
    )

    ax.set_extent(extent, crs=ccrs.PlateCarree())

    lat_dim, lon_dim = data[variable].dims

    if mask:
        mask = data_processing.create_mask(elev_path, pp_path, tree_path, count_path)
        plotval = data[variable].values
        plotval[~mask] = np.nan

        im = ax.pcolormesh(
            data[lon_dim].values,
            data[lat_dim].values,
            plotval,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        im = ax.pcolormesh(
            data[lon_dim],
            data[lat_dim],
            data[variable].values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    ax.coastlines(resolution="50m")
    ax.add_feature(cartopy.feature.BORDERS)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=2,
        color="grey",
        alpha=0.5,
        linestyle="--",
    )

    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator(np.arange(-80, -39, 20))
    gl.ylocator = mticker.FixedLocator(np.arange(-30, 11, 15))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Add a colorbar to the plot.
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    fig.add_axes(ax_cb)
    plt.colorbar(
        im,
        cax=ax_cb,
        orientation="vertical",
        label=data[variable].long_name,
        extend="max",
    )

    if plot_shapes:
        for shape_path in shape_list:
            gdf = gpd.read_file(shape_path)
            gdf.plot(ax=ax, edgecolor="purple", facecolor="None", linewidth=2)

    # Save the plot.
    plt.savefig(file_path, bbox_inches="tight", dpi=dpi)


def plot_histogram_byarea(df, total_area, filename):
    """
    Plots a histogram of tree cover percentages and saves the result to a file.

    Args:
    ----------
    df : pd.DataFrame
        DataFrame containing tree cover data. The DataFrame should have at least two columns 'TC' and 'AREA'.
    total_area : float
        Total area of the region being analyzed. This value is used to normalize the histogram.
    filename : str
        Path to the output file where the histogram will be saved.

    Returns:
    -------
    None
    """

    # Set font size for the plot
    matplotlib.rcParams.update({"font.size": 20})

    # Create a new figure with specified dimensions
    fig = plt.figure(figsize=[5, 5])

    # Plot the histogram of tree cover percentages
    plt.hist(
        df["TC"],
        bins=np.arange(0, 101, 10),
        weights=df["AREA"].values / total_area,
        facecolor="black",
        edgecolor="black",
    )

    # Set x-axis properties
    plt.xticks(np.arange(0, 91, 30))
    plt.xlim(-5, 105)
    plt.xlabel("Tree Cover (%)")

    # Set y-axis properties
    plt.ylabel("Area fraction (%)")
    plt.ylim(0, 0.65)
    plt.yticks(np.arange(0, 0.61, 0.2))

    # Save the figure to the specified file
    plt.savefig(filename)


def plot_treecover_rainfall_histogram(df):
    """
    Plots a bar chart of the frequency of savanna and forest areas for different mean annual precipitation ranges.

    Args:
    ----------
    df : pd.DataFrame
        DataFrame containing tree cover and rainfall data. The DataFrame should have at least
        columns 'MAP', 'CLASS', and 'AREA'.

    Returns:
    -------
    None
    """

    # Set font size for the plot
    plt.rcParams.update({"font.size": 20})

    # Create a new figure with specified dimensions
    fig = plt.figure(figsize=[10, 5])
    ax = fig.add_subplot(111)

    # Define bins for mean annual precipitation
    bins = np.arange(0, 3001, 100)
    map_dig = np.digitize(df["MAP"].values, bins)

    # Create masks for savanna and forest classes
    savanna_mask = (df["CLASS"] == "Savanna").values
    forest_mask = (df["CLASS"] == "Forest").values

    # Calculate frequencies for savanna and forest classes
    savanna = np.array(
        [
            np.sum(df[np.logical_and(savanna_mask, map_dig == i)]["AREA"])
            for i in range(1, len(bins))
        ]
    ) / (np.array([np.sum(df[map_dig == i]["AREA"]) for i in range(1, len(bins))]))

    forest = np.array(
        [
            np.sum(df[np.logical_and(forest_mask, map_dig == i)]["AREA"])
            for i in range(1, len(bins))
        ]
    ) / (np.array([np.sum(df[map_dig == i]["AREA"]) for i in range(1, len(bins))]))

    # Plot bar chart for savanna and forest frequencies
    ax.bar(
        bins[:-1],
        savanna,
        width=(bins[1] - bins[0]) / 4,
        align="edge",
        fc="orange",
        ec="black",
    )
    ax.bar(
        bins[:-1] + 70,
        forest,
        width=(bins[1] - bins[0]) / 4,
        align="edge",
        fc="green",
        ec="black",
    )

    # Set x-axis and y-axis properties
    ax.set_xlabel("Mean Annual Precipitation (mm $yr^{-1}$)")
    ax.set_ylabel("Frequency")
    _ = ax.set_xticks(np.arange(0, 3001, 500))

    plt.savefig("tc_hist_rainfall.svg", bbox_inches="tight")


def plot_doublestress_2d(df):
    """
    Plots a heatmap showing the fraction of area occupied by forests for different combinations of
    drought and waterlogging severities.

    Args:
    ----------
    df : pd.DataFrame
        DataFrame containing at least drought SEV (DR_SEV), waterlogging SEV (WL_SEV), area (AREA),
        and vegetation class (CLASS) information.

    Returns:
    -------
    None
    """

    # Update the font size for the plot
    plt.rcParams.update({"font.size": 20})

    # Create a new figure with specified dimensions
    fig, ax = plt.subplots(figsize=(7, 7))

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

    # Count
    stat_count, x_edge, y_edge, binnumber = binned_statistic_2d(
        df["DR_SEV"],
        df["WL_SEV"],
        df["AREA"],
        statistic="count",
        bins=np.arange(0, 14, 1),
    )

    # Set forest statistics to NaN where there are fewer than 100 data points
    stat_for[stat_count < 100] = np.nan

    # Define colormap and normalization for the heatmap
    cmap4 = matplotlib.colors.ListedColormap(
        [
            "#E12222",
            "#E15822",
            "#F2972D",
            "#F2D32D",
            "#6EDF80",
            "#33B648",
            "#198343",
            "#054531",
        ]
    )
    bounds = np.linspace(0, 100, 9)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap4.N)

    # Plot the heatmap
    sns.heatmap(
        np.rot90((stat_for / stat) * 100, axes=(0, 1)),
        linewidth=1,
        cmap=cmap4,
        norm=norm,
        cbar_kws={"label": "Fraction of area occupied by forests"},
    )

    # Set x-axis and y-axis properties
    plt.xlabel("Drought SEV (months)")
    plt.ylabel("Waterlogging SEV (months)")
    ax.set_yticks([0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5])
    ax.set_yticklabels(["12", "10", "8", "6", "4", "2", "0"], rotation=0)
    ax.set_xticks([0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5])
    ax.set_xticklabels(["0", "2", "4", "6", "8", "10", "12"], rotation=0)

    # Save the plot
    plt.savefig("doublestress_2d.png", bbox_inches="tight", dpi=600)


def kde_integral(x, y, lower_bound, upper_bound):
    indices = np.where((x >= lower_bound) & (x <= upper_bound))
    return integrate.simps(y[indices], x[indices])


def plot_stats_kde(data, names, shape_paths):
    """
    Create and display a KDE estimate of the distribution for tree cover percentage
    for different double stress classes, combining weak and strong double stress in one class.

    Args:
    ----------
        data (pd.DataFrame):
            A DataFrame containing the data to be plotted, including columns 'DS_CLASS' and 'TC'.
        names (list of str):
            List of names for the output files to be saved.
        shape_paths (list of str):
            List of paths to the shapefiles to filter data by.

    Returns:
    ----------
        None
    """
    for name, shape_path in zip(names, shape_paths):
        print(f"Processing data for {name}...")

        # Filter data by shapefile if provided
        if shape_path is not None:
            data_clip = geoprocessing.filter_data_by_shapefile(data, shape_path)

        # Combine weak and strong double stress in one class
        data_clip["DS_CLASS"] = data_clip["DS_CLASS"].apply(
            lambda x: 2 if x >= 2 else x
        )

        # Create a figure and axis with the specified size
        fig, ax = plt.subplots(figsize=(5, 5))

        # Set global font size
        plt.rcParams.update({"font.size": 28})

        # Plot the KDE for each double stress class
        no_double_stress = data_clip[data_clip["DS_CLASS"] == 1]["TC"]
        double_stress = data_clip[data_clip["DS_CLASS"] == 2]["TC"]

        kde_no_double_stress = sns.kdeplot(
            no_double_stress,
            ax=ax,
            color="black",
            lw=3,
            label="No Double Stress",
            linestyle="-",
        )
        kde_double_stress = sns.kdeplot(
            double_stress,
            ax=ax,
            color="black",
            lw=3,
            label="Double Stress",
            linestyle="--",
        )

        # Set x and y ticks
        xticks = [0, 40, 80]
        yticks = np.around(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 3), 2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        # Add a vertical, dotted, red line at tree cover = 60
        ax.axvline(x=60, color="red", linestyle=":", linewidth=2)

        # Set axis labels
        ax.set_xlabel("Tree Cover (%)")
        ax.set_ylabel("Density")

        # Add legend
        # ax.legend()

        # Save the plot
        plt.savefig(f"{name}_stats.png", bbox_inches="tight", dpi=300)

        # Calculate the integrals and print the results
        x_no_double_stress, y_no_double_stress = kde_no_double_stress.get_lines()[
            0
        ].get_data()
        x_double_stress, y_double_stress = kde_double_stress.get_lines()[1].get_data()
        no_double_stress_integral_0_to_60 = (
            kde_integral(x_no_double_stress, y_no_double_stress, 0, 60) * 100
        )
        no_double_stress_integral_60_to_100 = (
            kde_integral(x_no_double_stress, y_no_double_stress, 60, 100) * 100
        )
        double_stress_integral_0_to_60 = (
            kde_integral(x_double_stress, y_double_stress, 0, 60) * 100
        )
        double_stress_integral_60_to_100 = (
            kde_integral(x_double_stress, y_double_stress, 60, 100) * 100
        )

        # Print the integrals for each category
        print(
            f"No Double Stress: {no_double_stress_integral_0_to_60:.2f}, {no_double_stress_integral_60_to_100:.2f}"
        )
        print(
            f"Double Stress: {double_stress_integral_0_to_60:.2f}, {double_stress_integral_60_to_100:.2f}"
        )


def plot_topoclimate_hists(df):
    """
    Function to plot histograms and print area statistics for different topoclimate classes based on 'MAP' column.

    Args:
    ----------
    df (pd.DataFrame): Input dataframe with columns 'MAP', 'DR_SEV', 'WL_SEV', 'AREA', and 'CLASS'.

    Returns:
    ----------
    None
    """
    # Define the ranges for the MAP classes
    MAP_classes = {
        "Dry": (None, 1400),
        "Intermediate": (1400, 1800),
        "Wet": (1800, None),
    }

    # Define the class conditions
    class_conditions = [
        lambda x: x["DR_SEV"] == 12,  # Always below 2
        lambda x: np.logical_and(x["WL_SEV"] >= 1, x["DR_SEV"] >= 1),
        lambda x: x["WL_SEV"] == 12,  # Never below 2
    ]

    # Iterate over MAP classes
    for map_class, (lower, upper) in MAP_classes.items():
        # Apply the MAP threshold
        if lower is None:
            subset = df[df["MAP"] < upper]
        elif upper is None:
            subset = df[df["MAP"] > lower]
        else:
            subset = df[np.logical_and(df["MAP"] >= lower, df["MAP"] <= upper)]

        print(f"\nMAP Class: {map_class}, MAP thresholds: {lower} - {upper}")

        # Iterate over class conditions
        for i, condition in enumerate(class_conditions):
            class_df = subset[condition(subset)]
            total = class_df["AREA"].sum()
            sav = class_df[class_df["CLASS"] == "Savanna"]["AREA"].sum()
            fore = class_df[class_df["CLASS"] == "Forest"]["AREA"].sum()

            print(
                f"In class {i+1}, forest is {round((fore/total)*100,1)}, and savanna is {round((sav/total)*100,1)}"
            )
            # print(
            #     "Total area is:", np.format_float_scientific(total * 1e-6, precision=2)
            # )
            # print(
            #     "Savanna area is:", np.format_float_scientific(sav * 1e-6, precision=2)
            # )
            # print(
            #     "Forest area is:", np.format_float_scientific(fore * 1e-6, precision=2)
            # )

            plot_histogram_byarea(
                class_df, total, (f"\n{map_class}_class{i+1}_histnew.svg")
            )


def plot_climatology(df):
    """
    The function `plot_climatology` generates and saves bar and line plots illustrating the monthly average
    precipitation and water table depth respectively for different regions. The regions and associated data are
    provided in the form of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with the following columns:
        - 'shapefile': Contains the file paths of the shapefiles. Each shapefile represents a distinct region.
        - 'pp_mean': Contains the mean precipitation data for the corresponding region.
        - 'wtd_mean': Contains the mean water table depth data for the corresponding region.

    Returns
    -------
    None. However, this function will save a .png file for each region's plot in the current working directory. The file
    is named as "{region_name}_climatology.png", where {region_name} is replaced by the actual name of the region.
    """
    # Set font size for the plot
    matplotlib.rcParams.update({"font.size": 28})

    # Loop over rows in the DataFrame
    for idx, row in df.iterrows():
        # Extract shapefile path and name
        shapefile_path = row["shapefile"]
        name = os.path.splitext(os.path.basename(shapefile_path))[0]

        # Extract pp_mean and wtd_mean from the DataFrame
        pp_mean = row["pp_mean"]
        wtd_mean = row["wtd_mean"]

        # Create the plot
        fig, ax = plt.subplots(figsize=[5, 4])
        ax2 = ax.twinx()

        index = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
            "",
        ]
        tick_spacing = 5.5
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        # Precipitation plot
        ax.bar(index[:-1], pp_mean, edgecolor="black", facecolor="black")
        ax.set_ylim(0, 450)
        ax.set_ylabel("Rainfall (mm)")
        ax.set_yticks([0, 100, 300])

        # Water table plot
        ax2.plot(-wtd_mean, color="blue", linewidth=3)
        ax2.set_ylim([5, -0.5])
        ax2.set_ylabel("WTD (m)")
        ax2.set_yticks([0, 2.5, 5])

        # Save the plot
        plt.savefig(f"{name}_climatology.png", dpi=300, bbox_inches="tight")
        # plt.close(fig)


def plot_dsi_map(
    data,
    variable,
    file_path,
    mask=False,
    elev_path=None,
    pp_path=None,
    tree_path=None,
    count_path=None,
    extent=[-85, -33, -35, 15],
    dpi=1200,
    cmap=["#1a3ce8", "#fed976", "#ff4d4d"],
):
    """
    Plots a map of DSI classes.

    Args:
    ----------
        data (xarray.DataArray or xarray.Dataset):
            The dataset to plot.
        variable (str):
            The name of the variable to plot.
        file_path (str):
            The path to save the output file.
        mask (bool, optional):
            If True, a mask will be applied to the data. Default is False.
        elev_path (str, optional):
            The path to the elevation dataset. Required if mask is True.
        pp_path (str, optional):
            The path to the precipitation dataset. Required if mask is True.
        tree_path (str, optional):
            The path to the tree cover dataset. Required if mask is True.
        count_path (str, optional):
            The path to the tree count dataset. Required if mask is True.
        extent (list, optional):
            The extent of the plot, as a list of four floats (xmin, xmax, ymin, ymax).
            If None, the default extent will be used. Default is tropical South America.
        dpi (int, optional):
            The resolution of the output file. Default is 1200.
        cmap (list, optional):
            The colormap to use for the plot as a list of color strings.
            Default is ["#1a3ce8", "#fed976", "#ff4d4d"].

    Returns:
    -------
        matplotlib.figure.Figure:
            The figure object.
    """

    if mask and (
        elev_path is None or pp_path is None or tree_path is None or count_path is None
    ):
        raise ValueError(
            "If mask is True, the paths to the elevation, precipitation, tree cover, and tree count datasets must be provided."
        )

    if extent is not None and len(extent) != 4:
        raise ValueError("The extent must be a list of four floats.")

    fig, ax = plt.subplots(
        figsize=[8, 5], subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # 4. Aggregate the classes to 0.25-degree resolution

    mask_arr = data_processing.create_mask(elev_path, pp_path, tree_path, count_path)
    data = xarray.where(~mask_arr, np.nan, data)

    coarsening_factor = int(0.1 * 3600 / 30)
    dsi_coarsened_count = (
        data[variable]
        .coarsen(lon=coarsening_factor, lat=coarsening_factor, boundary="trim")
        .count()
    )
    dsi_coarsened = (
        data[variable]
        .coarsen(lon=coarsening_factor, lat=coarsening_factor, boundary="trim")
        .mean(skipna=True)
    )

    dsi_coarsened = xarray.where(
        dsi_coarsened_count < (coarsening_factor**2) * 0.8, np.nan, dsi_coarsened
    )

    ax.set_extent(extent, crs=ccrs.PlateCarree())

    custom_cmap = matplotlib.colors.ListedColormap(cmap)

    if mask:
        im = ax.pcolormesh(
            dsi_coarsened["lon"],
            dsi_coarsened["lat"],
            dsi_coarsened[::-1],
            cmap=custom_cmap,
        )
    else:
        im = ax.pcolormesh(
            dsi_coarsened["lon"],
            dsi_coarsened["lat"],
            dsi_coarsened[::-1],
            cmap=custom_cmap,
        )

    ax.coastlines(resolution="50m")
    ax.add_feature(cartopy.feature.BORDERS)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=2,
        color="grey",
        alpha=0.5,
        linestyle="--",
    )

    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator(np.arange(-80, -39, 20))
    gl.ylocator = mticker.FixedLocator(np.arange(-30, 11, 15))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Save the plot.
    plt.savefig(file_path, bbox_inches="tight", dpi=dpi)


def plot_upscaled_mask(mask, lats, resolution_factor=30):
    """
    This function calculates the proportion of the area covered by 2km pixels within
    each 0.5 degree pixel, and plots this as a heatmap.

    Args:
    ----------
    mask_2km: xarray.DataArray
        A 2D xarray DataArray with boolean values, where True indicates the presence of a 2km pixel.
    resolution_factor: int
        The factor by which to upscale the 2km data. Default is 30, which corresponds to a 0.25 degree resolution.

    Returns:
    ----------
    A plot of the proportion of the area covered by 2km pixels within each 0.5 degree pixel.
    """
    matplotlib.rcParams.update({"font.size": 20})

    # First create a mxn array with the latitudes (where m and n are dimensions of input data)
    lat = np.tile((mask.lat.values), (np.shape(mask)[1], 1)).transpose()

    # Now get an area array
    area_pixels_2km = geoprocessing.area_of_pixel(lat, pixel_size=0.01666641)

    # Calculate binary mask where 2km pixels are set to 1 if mask is True, and 0 otherwise
    binary_mask_2km = mask.where(mask, 1)
    binary_mask_2km = binary_mask_2km.where(~mask, 0)

    # Upscale binary mask and calculate total area of 2km pixels within each 0.5 degree pixel
    mask_half_degree = binary_mask_2km.coarsen(
        dim={"lat": resolution_factor, "lon": resolution_factor}, boundary="trim"
    ).sum()

    # Also upscale the area pixels to match the shape of mask_half_degree
    area_pixels_2km_xarray = xarray.DataArray(
        area_pixels_2km, dims=["lat", "lon"], coords={"lat": mask.lat, "lon": mask.lon}
    )

    area_pixels_2km_xarray = area_pixels_2km_xarray.where(binary_mask_2km == 0, 0)

    area_pixels_2km_upscaled = area_pixels_2km_xarray.coarsen(
        dim={"lat": resolution_factor, "lon": resolution_factor}, boundary="trim"
    ).sum()

    lat = np.tile(
        mask_half_degree.lat.values, (np.shape(mask_half_degree)[1], 1)
    ).transpose()

    area_pixel_half_degree = geoprocessing.area_of_pixel(
        lat,
        pixel_size=np.abs(
            mask_half_degree.lat[1].values - mask_half_degree.lat[0].values
        ),
    )

    # Calculate proportion of area covered by 2km pixels within each 0.5 degree pixel
    proportion_area = (area_pixels_2km_upscaled / area_pixel_half_degree) * 100

    # Plotting the proportion area
    fig = plt.figure(figsize=[6.3, 5])
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent(
        (
            np.min(proportion_area.lon.values),
            np.max(proportion_area.lon.values),
            np.min(proportion_area.lat.values),
            np.max(proportion_area.lat.values),
        ),
        crs=ccrs.PlateCarree(),
    )

    im = ax.pcolormesh(
        proportion_area.lon.values,
        proportion_area.lat.values,
        proportion_area.values,
        cmap="gist_heat_r",
        vmin=0,
        vmax=30,
    )

    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=2,
        color="grey",
        alpha=0.5,
        linestyle="--",
    )

    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator(np.arange(-80, -39, 20))
    gl.ylocator = mticker.FixedLocator(np.arange(-30, 11, 10))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)

    fig.add_axes(ax_cb)
    plt.colorbar(
        im, cax=ax_cb, ticks=np.arange(0, 35, 10), label="Double Stress Fraction (%)"
    )

    ## Now plot zoom windows ##
    ur = gpd.read_file("SHAPES/upper_bb.shp")
    ur.plot(ax=ax, edgecolor="purple", facecolor="None", linewidth=2)

    madeira = gpd.read_file("SHAPES/madeira_bb.shp")
    madeira.plot(ax=ax, edgecolor="purple", facecolor="None", linewidth=2)

    roraima = gpd.read_file("SHAPES/roraima_bb.shp")
    roraima.plot(ax=ax, edgecolor="purple", facecolor="None", linewidth=2)

    pmfb_bb = gpd.read_file("SHAPES/pmfb_bb.shp")
    pmfb_bb.plot(ax=ax, edgecolor="purple", facecolor="None", linewidth=2)

    plt.savefig(
        "samerica_future_forests_doublestress.png", bbox_inches="tight", dpi=1200
    )


def plot_tree_cover_mask(tree_cover_file, mask, count_file, shape_file):
    """
    This function plots the valid tree cover and mask data for a given shape file area.

    It uses the count data to mask invalid tree cover data, and then overlays the mask data in purple.

    Args:
    ----------
    tree_cover_file: str
        Path to the tree cover file. Should be in a format readable by xarray (e.g., netCDF).
    mask: xarray.DataArray or xarray.DataSet
        A 2D xarray DataArray or Dataset with boolean values, where True indicates the presence of a feature.
    count_file: str
        Path to the count file, used to mask invalid tree cover data.
    shape_file: str
        Path to the shape file that defines the area of interest.

    Returns:
    ----------
    A plot of the valid tree cover and mask data for the shape file area.
    """

    # Load the tree cover data
    tree_cover = xarray.open_dataset(tree_cover_file)

    # Load the count data
    count = xarray.open_dataset(count_file)

    # Mask the invalid tree cover data
    valid_tree_cover = tree_cover.where(count.TC >= 44)

    # Load the shape file
    gdf = gpd.read_file(shape_file)

    # Get the bounding box coordinates of the shape file
    minx, miny, maxx, maxy = gdf.total_bounds

    # Select the tree cover and mask data for the shape file area
    valid_tree_cover = valid_tree_cover.sel(
        lat=slice(maxy, miny), lon=slice(minx, maxx)
    )
    mask = mask.sel(lat=slice(maxy, miny), lon=slice(minx, maxx))

    # Create a figure and subplot
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot the valid tree cover data
    im = ax.pcolormesh(
        valid_tree_cover.lon.values[: len(valid_tree_cover.lat.values)],
        valid_tree_cover.lat.values,
        valid_tree_cover.TC.values[:, : len(valid_tree_cover.lat.values)],
        cmap="YlGn",
        vmin=0,
        vmax=100,
    )

    # Overlay the mask data in purple
    plot = np.zeros(np.shape(mask.values))
    plot[~mask] = np.nan
    plot[mask] = 1
    im = ax.pcolormesh(
        mask.lon.values, mask.lat.values, plot, cmap=colors.ListedColormap(["purple"])
    )

    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks

    ax.set_xticklabels([])  # Remove x-axis labels
    ax.set_yticklabels([])  # Remove y-axis labels

    # Show the plot
    # Save
    # os.path.basename() gets the filename from the full path
    filename = os.path.basename(shape_file)

    # os.path.splitext() separates the filename from the extension
    name, ext = os.path.splitext(filename)

    # splits the name into parts using underscore as a separator
    parts = name.split("_")

    # get the first part of the name
    subset_name = parts[0]

    plt.savefig(f"{subset_name}_future_inset.png", bbox_inches="tight", dpi=1200)
