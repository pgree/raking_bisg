"""
this module contains functions for plotting the figures and latex tables of the
paper [citation].
the functions of this module write the figures and tables to a user-specified directory.
the figures and tables fall into three categories:
- voter file-only : results relating to predictions made using only voter file information from
voter files with labeled with self-identified race/ethnicity
- subsampled validation : results for validation performed by subsampling voter files
- calibration map validation : results for validation on full voter files using a
calibration map

Functions
---------
self_contained_figures_tables
    tables and figures for the voter file only predictions
subsampled_figures_tables
    tables and figures for the subsampled validation
calib_map_figures_tables
    tables and figures for the calibration map validation
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from fields import *
import process_census
import process_cps
from raking_bisg import (
    make_df_agg,
    dist_summary,
    subpopulation_preds,
    one_calib,
    l1_l2_tables,
    ll_tables,
    load_voter_file,
    make_calib_map,
)


def main():
    dir_out = "/Users/ramin/bisg/writeup/overleaf/"
    states = ["nc", "nc", "fl"]
    years = [2020, 2010, 2020]
    states = ["nc", "nc"]
    years = [2020, 2010]
    for state, year in zip(states, years):
        print(state, year)
        # subsampled_figures_tables(state, year, dir_out, load=True)
        calib_map_figures_tables(state, year, dir_out, load=True)
        # self_contained_figures_tables(state, year, dir_out, load=True)


def subsampled_figures_tables(state, year, dir_out, load=False):
    """
    make all figures and latex tables for the subsampled validation and save them
    in subdirectories of the inputted directory dir_out. if these directories
    don't exist, they are created

    Parameters
    ----------
    state : string
        state
    year : integer
        year
    dir_out : str
        home directory of figures and tables
    load : bool, optional
        load the predictions dataframe df_agg

    Returns
    -------
    None
    """
    # directories to save the figures and tables
    figures_dir = dir_out + "figures/subsampled/"
    tables_dir = dir_out + "tables/subsampled/"
    # if the directories don't exist, create them
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir)

    # load df_agg
    df_agg = make_df_agg(state, year, subsample=True, load=load)

    # make raking predictions
    cols1 = RAKE_COLS
    preds1, true_pops1, true_probs1 = subpopulation_preds(
        df_agg, cols1, region="county", calib_map=None
    )

    # make bisg predictions
    cols2 = BISG_BAYES_COLS
    preds2, true_pops2, true_probs2 = subpopulation_preds(
        df_agg, cols2, region="county", calib_map=None
    )

    # scatterplot
    filename = figures_dir + f"{state}{year}_scatters.pdf"
    abs_rel_scatters(
        filename, preds1, true_pops1, preds2, true_pops2, label1="raking", label2="BISG"
    )

    # barplots, but florida bar plots have an axis break, so they require special functions
    if state.lower() == "fl":
        filename = figures_dir + f"{state}{year}_bars.pdf"
        fl_bar_plots(
            filename,
            preds1,
            true_pops1,
            preds2,
            true_pops2,
            label1="raking",
            label2="BISG",
        )

    else:
        filename = figures_dir + f"{state}{year}_bars.pdf"
        bar_plots(
            filename,
            preds1,
            true_pops1,
            preds2,
            true_pops2,
            label1="raking",
            label2="BISG",
        )

    # write validation table to file
    filename_county = tables_dir + f"{state}{year}_validation_table_counties.tex"
    filename_region = tables_dir + f"{state}{year}_validation_table_regions.tex"
    validation_err_table(
        df_agg, filename_county, filename_region, state, calib_map=None
    )

    # calibration plots
    filename = figures_dir + f"{state}{year}_calib_grid.pdf"
    cols1 = BISG_BAYES_COLS
    cols2 = RAKE_COLS
    kstats = calib_plots(filename, df_agg, cols1, cols2, calib_map=None)

    # calibration table
    filename = tables_dir + f"{state}{year}_kuiper.tex"
    calib_table(filename, kstats, state, year)


def calib_map_figures_tables(state, year, dir_out, load=False):
    """
    make all figures and latex tables for the calibration map validation and save them
    in subdirectories of the inputted directory dir_out. if these directories
    don't exist, they are created

    Parameters
    ----------
    state : string
        state
    year : integer
        year
    dir_out : str
        home directory of figures and tables
    load : bool, optional
        load the predictions dataframe df_agg

    Returns
    -------
    None
    """
    figures_dir = dir_out + "figures/calib_map/"
    tables_dir = dir_out + "tables/calib_map/"
    # if the directory to which we're going to write figures and tables
    # doesn't exist, then create it
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir)

    df_cps_reg_voters = process_cps.cps_geo_race(state, year, voters=True)
    df_agg = make_df_agg(state, year, subsample=False, load=load)
    calib_map = make_calib_map(df_cps_reg_voters, df_agg, verbose=False)

    # make raking predictions
    cols1 = RAKE_COLS
    preds1, true_pops1, true_probs1 = subpopulation_preds(
        df_agg, cols1, region="county", calib_map=calib_map
    )
    # make bisg predictions
    cols2 = BISG_BAYES_COLS
    preds2, true_pops2, true_probs1 = subpopulation_preds(
        df_agg, cols2, region="county", calib_map=calib_map
    )
    # make voter file predictions
    preds1vf, true_pops1vf, _ = subpopulation_preds(
        df_agg, VF_BISG_COLS, region="county", calib_map=None
    )
    preds2vf, true_pops2vf, _ = subpopulation_preds(
        df_agg, VF_BISG_COLS, region="region", calib_map=None
    )

    # scatter plot
    filename = figures_dir + f"{state}{year}_scatters.pdf"
    abs_rel_scatters(
        filename, preds1, true_pops1, preds2, true_pops2, label1="raking", label2="BISG"
    )

    # barplot (florida has its own function because there's an axis break)
    if state.lower() == "fl":
        filename = figures_dir + f"{state}{year}_bars.pdf"
        fl_bar_plots(
            filename,
            preds1,
            true_pops1,
            preds2,
            true_pops2,
            label1="raking",
            label2="BISG",
        )
    else:
        filename = figures_dir + f"{state}{year}_bars.pdf"
        bar_plots(
            filename,
            preds1,
            true_pops1,
            preds2,
            true_pops2,
            label1="raking",
            label2="BISG",
        )

    # write validation table to file
    filename_county = tables_dir + f"{state}{year}_validation_table_counties.tex"
    filename_region = tables_dir + f"{state}{year}_validation_table_regions.tex"
    validation_err_table(df_agg, filename_county, filename_region, state, calib_map)

    # calibration plots
    filename = figures_dir + f"{state}{year}_calib_grid.pdf"
    cols1 = BISG_BAYES_COLS
    cols2 = RAKE_COLS
    kstats = calib_plots(filename, df_agg, cols1, cols2, calib_map)

    # calibration table
    filename = tables_dir + f"{state}{year}_kuiper.tex"
    calib_table(filename, kstats, state, year)


def self_contained_figures_tables(state, year, dir_out, load=False):
    """
    write figures and tables for the paper that involve only the
    self-contained and labeled predictions, that is, all of these predictions
    involve only the data from labeled voter files. this function will create
    directories for saving the figures and tables if they don't exist

    Parameters
    ----------
    state : string
        one of 'fl', 'nc'
    year : int
        year, one of 2020 or 2010 for nc and 2020 for fl
    dir_out : string
        the home directory of the figures and tables
    load : bool, optional
        load the predictions dataframe df_agg

    Returns
    -------
    None
    """
    figures_dir = dir_out + "figures/voter_file_only/"
    tables_dir = dir_out + "tables/voter_file_only/"
    # if the directory to which we're going to write figures and tables
    # doesn't exist, then create it
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir)

    # load df_agg
    df_agg = make_df_agg(state, year, subsample=False, load=load)
    # create and write df_summary
    df_cen_usa = process_census.usa_census(year)
    df_cen_counties = process_census.county_census(state, year)
    df_cps_reg_voters = process_cps.cps_geo_race(state, year, voters=True)
    df_vf = load_voter_file(state, year, load=load)
    df_summary = dist_summary(
        state, year, df_cen_usa, df_cen_counties, df_vf, df_cps_reg_voters
    )

    # make voter file predictions, including only names that appear in the surname list
    df_tmp = df_agg[df_agg["in_cen_surs"]]
    preds1vf, true_pops1vf, _ = subpopulation_preds(
        df_tmp, VF_BISG_COLS, region="county", calib_map=None
    )
    preds2vf, true_pops2vf, _ = subpopulation_preds(
        df_tmp, VF_BISG_COLS, region="region", calib_map=None
    )

    # self-contained voter file scatter plots
    filename = figures_dir + f"{state}{year}_vf_scatters.pdf"
    two_row_comp(
        filename=filename,
        preds=preds1vf,
        true_pops=true_pops1vf,
        preds2=preds2vf,
        true_pops2=true_pops2vf,
        label1="county",
        label2="region",
    )

    # self-contained voterfile table for full state
    filename = tables_dir + f"{state}{year}_vf_table_state.tex"
    voterfile_state_table(filename, preds1vf, true_pops1vf)

    # self-contained voter file table for all counties
    filename1 = tables_dir + f"{state}{year}_vf_table_counties.tex"
    filename2 = tables_dir + f"{state}{year}_vf_table_region.tex"
    vf_err_tables(filename1, filename2, df_agg, state)

    # summary dataframe
    filename = dir_out + f"tables/{state}{year}_summary.tex"
    write_df_summary(filename, df_summary)


def two_row_comp(
    filename,
    preds,
    true_pops,
    preds2=None,
    true_pops2=None,
    label1="BISG",
    label2="raking",
    c1=None,
    c2=None,
):
    """
    save a figure with two rows of sactter plots with the absolute error (top row) and relative error (bottom)
    of one or two predictions.

    Parameters
    ----------
    filename : string
        name of file that's written
    preds : n x m numpy array
        predictions
    true_pops : n x m numpy array
        the correct totals that preds is trying to predict
    preds2 : n x m numpy array, optional
        predictions
    true_pops2 : n x m numpy array, optional
        the correct totals that preds2 is trying to predict
    label1 : string, optional
        the label in the legend of the first set of predictions
    label2 : string, optional
        the label in the legend of the second set of predictions
    c1 : string, optional
        pyplot color of scatter plot dots for first predictions
    c2 : strong, optional
        pyplot color of scatter plot dots for second predictions

    Returns
    -------
    None
    """

    # exclude the "other" category
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))

    # default colors
    if c1 is None:
        c1 = "r"
    if c2 is None:
        c2 = "b"

    # compute absolute errors
    abs_errs = preds - true_pops
    rel_errs = preds / true_pops - 1
    true_probs = np.divide(true_pops, np.sum(true_pops, axis=1).reshape((-1, 1)))
    if preds2 is not None:
        abs_errs2 = preds2 - true_pops2
        rel_errs2 = preds2 / true_pops2 - 1
        true_probs2 = np.divide(true_pops2, np.sum(true_pops2, axis=1).reshape((-1, 1)))

    for i, race in enumerate(RACES[:5]):
        # plot a horizontal bar at 0 and make sure the bar extends to the right length
        if true_pops2 is not None:
            xmin = np.minimum(
                np.min(np.log10(true_pops[:, i])), np.min(np.log10(true_pops2[:, i]))
            )
            xmax = np.maximum(
                np.max(np.log10(true_pops[:, i])), np.max(np.log10(true_pops2[:, i]))
            )
        else:
            xmin = np.min(np.min(np.log10(true_pops[:, i])))
            xmax = np.max(np.max(np.log10(true_pops[:, i])))
        # in rare cases, a subpopulation might contain zero people
        if np.isneginf(xmin):
            xmin = 0
        xs = np.linspace(xmin, xmax, 2)
        # size of dots
        s = 15
        axs[0, i].plot(xs, np.zeros_like(xs), c="black")
        axs[0, i].scatter(
            np.log10(true_pops[:, i]), abs_errs[:, i], label=label1, c=c1, s=s
        )
        axs[0, i].set_title(f"{PRETTY_COLS[i]}", fontsize=24)
        axs[0, i].tick_params(axis="x", labelsize=14)
        axs[0, i].tick_params(axis="y", labelsize=14)
        axs[0, i].yaxis.get_offset_text().set_fontsize(14)

        # comparison
        if preds2 is not None:
            axs[0, i].scatter(
                np.log10(true_pops2[:, i]), abs_errs2[:, i], label=label2, c=c2, s=s
            )

        # formatting
        axs[0, i].ticklabel_format(style="sci", axis="both", scilimits=(-2, 2))
        axs[1, i].plot(xs, np.zeros_like(xs), c="black")

        # scatter plot
        axs[1, i].scatter(
            np.log10(true_pops[:, i]), rel_errs[:, i], label=label1, c=c1, s=s
        )
        axs[1, i].tick_params(axis="x", labelsize=14)
        axs[1, i].tick_params(axis="y", labelsize=14)
        axs[1, i].set_title(f"{PRETTY_COLS[i]}", fontsize=24)

        # comparison
        if preds2 is not None:
            axs[1, i].scatter(
                np.log10(true_pops2[:, i]), rel_errs2[:, i], label=label2, c=c2, s=s
            )

    # set zero to be the center of the y-axis
    for i, race in enumerate(RACES[:5]):
        yabs_max = np.max(np.abs(axs[0, i].get_ylim()))
        axs[0, i].set_ylim(ymin=-yabs_max, ymax=yabs_max)
        yabs_max = np.max(np.abs(axs[1, i].get_ylim()))
        axs[1, i].set_ylim(ymin=-yabs_max, ymax=yabs_max)

    # one legend for the full plot
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        fontsize=24,
        bbox_to_anchor=(0.5, 1.15),
        fancybox=True,
        shadow=True,
        ncol=2,
    )
    fig.text(
        0.5, 0.51, "$\\log_{10}$ of population", ha="center", va="center", fontsize=24
    )
    fig.text(
        0.0,
        0.75,
        "Absolute error",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=22,
    )
    fig.text(
        0.5, -0.02, "$\\log_{10}$ of population", ha="center", va="center", fontsize=24
    )
    fig.text(
        0.0,
        0.25,
        "Relative error",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=22,
    )
    fig.tight_layout(h_pad=7)

    # save figure
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")


def calib_subplots(width, height):
    """
    create the axes for the calibration plots with 5 subpopulations, 3 in the first row and 2 in the next

    Parameters
    ----------
    width : float
        width of the figure
    height : float
        height of the figure

    Returns
    -------
    pyplot figure
        figure with calibration plot grid
    list
        pyplot axes of calibration plot
    """
    fig = plt.figure(layout=None, figsize=(width, height))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=9,
        left=0.00,
        right=1.0,
        bottom=0.0,
        top=1.0,
        hspace=0.35,
        wspace=0.7,
    )
    ax00 = fig.add_subplot(gs[0, 0:3])
    ax01 = fig.add_subplot(gs[0, 3:6])
    ax02 = fig.add_subplot(gs[0, 6:9])
    ax03 = fig.add_subplot(gs[1, 1:4])
    ax04 = fig.add_subplot(gs[1, 5:8])

    axs = np.array([ax00, ax01, ax02, ax03, ax04])

    return fig, axs


def scatterplot_subplots():
    """
    create the axes for the scatter plots with 5 subpopulations and absolute and relative errors.
    these subplots will contain 3 scatter plots in the first two rows and two in the next 2.

    Parameters
    ----------
    None

    Returns
    -------
    pyplot figure
        figure with scatterplots
    list
        pyplot axs
    """
    fig = plt.figure(layout=None, figsize=(20, 18))
    gs = fig.add_gridspec(
        nrows=4,
        ncols=9,
        left=0.00,
        right=1.0,
        bottom=0.0,
        top=1.0,
        hspace=0.5,
        wspace=0.7,
    )
    ax00 = fig.add_subplot(gs[0, 0:3])
    ax01 = fig.add_subplot(gs[0, 3:6])
    ax02 = fig.add_subplot(gs[0, 6:9])
    ax10 = fig.add_subplot(gs[1, 0:3])
    ax11 = fig.add_subplot(gs[1, 3:6])
    ax12 = fig.add_subplot(gs[1, 6:9])
    ax03 = fig.add_subplot(gs[2, 1:4])
    ax04 = fig.add_subplot(gs[2, 5:8])
    ax13 = fig.add_subplot(gs[3, 1:4])
    ax14 = fig.add_subplot(gs[3, 5:8])

    axs = np.array([[ax00, ax01, ax02, ax03, ax04], [ax10, ax11, ax12, ax13, ax14]])

    return fig, axs


def abs_rel_scatters(
    filename, preds1, true_pops1, preds2, true_pops2, label1="raking", label2="BISG"
):
    """
    construct a 4 x 3 grid of scatter plots of subpopulation estimation
    for two different predictions. the first 2 x 1 upper left subplots
    correspond to one race/ethnicity, the upper scatter plot shows
    absolute error and the lower one relative error.

    Parameters
    ----------
    filename : string
        name of file that's written
    preds1 : n x m numpy array
        predictions
    true_pops1 : n x m numpy array
        the correct totals that preds is trying to predict
    preds2 : n x m numpy array
        predictions
    true_pops2 : n x m numpy array
        the correct totals that preds2 is trying to predict
    label1 : string, optional
        the label in the legend of the first set of predictions
    label2 : string, optional
        the label in the legend of the second set of predictions

    Returns
    -------
    None

    """

    alph = 1.0
    fig, axs = scatterplot_subplots()

    # size of dots in scatter plot
    s = 30

    # absolute errors
    abs_errs = preds1 - true_pops1
    abs_errs2 = preds2 - true_pops2

    # relative errors for predictions 1 and 2
    rel_errs = preds1 / true_pops1 - 1
    rel_errs2 = preds2 / true_pops2 - 1

    for i, col in enumerate(VF_RACES[:5]):

        # x and y coordinates for each set of predictions
        x1 = np.log10(true_pops1[:, i])
        y1 = abs_errs[:, i]
        x2 = np.log10(true_pops2[:, i])
        y2 = abs_errs2[:, i]
        # horizontal line at y=0
        xmin = np.min((np.min(x1), np.min(x2)))
        if np.isneginf(xmin):
            xmin = 0
        xs = np.linspace(xmin, np.max((np.max(x1), np.max(x2))), 2)
        axs[0, i].plot(xs, np.zeros_like(xs), c="black", linewidth=1)
        # scatterplots for absolute errors
        axs[0, i].scatter(x1, y1, alpha=alph, label=label1, c="b", s=s)
        axs[0, i].scatter(x2, y2, alpha=alph, label=label2, c="r", s=s)
        # labeling title and axes
        axs[0, i].set_title(f"{PRETTY_COLS[i]}", fontsize=32)
        axs[0, i].tick_params(axis="x", labelsize=17)
        axs[0, i].tick_params(axis="y", labelsize=17)
        axs[0, i].ticklabel_format(style="sci", axis="both", scilimits=(-2, 2))
        axs[0, i].yaxis.get_offset_text().set_fontsize(17)

        # relative errors
        y1 = rel_errs[:, i]
        y2 = rel_errs2[:, i]

        # cap relative errors
        mask = (y1 < 5) & np.isfinite(y1)
        x1 = x1[mask]
        y1 = y1[mask]
        mask2 = (y2 < 5) & np.isfinite(y2)
        x2 = x2[mask2]
        y2 = y2[mask2]

        # horizontal line at y=0
        xs = np.linspace(
            np.min((np.min(x1), np.min(x2))), np.max((np.max(x1), np.max(x2))), 2
        )
        axs[1, i].plot(xs, np.zeros_like(xs), c="black", linewidth=1)
        # scatter plots for relative errors
        axs[1, i].scatter(x1, y1, alpha=alph, label=label1, c="b", s=s)
        axs[1, i].scatter(x2, y2, alpha=alph, label=label2, c="r", s=s)
        # labeling axes
        axs[1, i].tick_params(axis="x", labelsize=17)
        axs[1, i].tick_params(axis="y", labelsize=17)
        axs[1, i].set_title(f"{PRETTY_COLS[i]}", fontsize=32)

    # set zero to be the center of the y-axis
    for i in range(2):
        for j in range(5):
            yabs_max = np.max(np.abs(axs[i, j].get_ylim()))
            axs[i, j].set_ylim(ymin=-yabs_max, ymax=yabs_max)

    # one legend for the full plot
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        fontsize=36,
        bbox_to_anchor=(0.5, 1.11),
        fancybox=True,
        shadow=True,
        ncol=2,
    )

    # x-axis labels
    fig.text(
        0.5, 0.775, "$\\log_{10}$ of population", ha="center", va="center", fontsize=28
    )
    fig.text(
        0.5, 0.505, "$\\log_{10}$ of population", ha="center", va="center", fontsize=28
    )
    fig.text(
        0.28, 0.24, "$\\log_{10}$ of population", ha="center", va="center", fontsize=28
    )
    fig.text(
        0.28, -0.03, "$\\log_{10}$ of population", ha="center", va="center", fontsize=28
    )
    fig.text(
        0.73, 0.24, "$\\log_{10}$ of population", ha="center", va="center", fontsize=28
    )
    fig.text(
        0.73, -0.03, "$\\log_{10}$ of population", ha="center", va="center", fontsize=28
    )
    # y-axis labels
    fig.text(
        -0.04,
        0.91,
        "Absolute error",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=27,
    )
    fig.text(
        -0.04,
        0.64,
        "Relative error",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=27,
    )
    fig.text(
        0.08,
        0.365,
        "Absolute error",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=27,
    )
    fig.text(
        0.068,
        0.095,
        "Relative error",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=27,
    )
    # save figure
    plt.savefig(filename, bbox_inches="tight", dpi=fig.dpi)


def bar_plots(
    filename, preds1, true_pops1, preds2, true_pops2, label1="raking", label2="BISG"
):
    """
    create two barplots, one on top of the other, showing mean absolute
    errors (top) and average errors (bottom) for each race/ethnicity.
    each plot shows errors for the two predictions.

        Parameters
    ----------
    filename : string
        name of file that's written
    preds1 : n x m numpy array
        predictions
    true_pops1 : n x m numpy array
        the correct totals that preds is trying to predict
    preds2 : n x m numpy array
        predictions
    true_pops2 : n x m numpy array
        the correct totals that preds2 is trying to predict
    label1 : string, optional
        the label in the legend of the first set of predictions
    label2 : string, optional
        the label in the legend of the second set of predictions

    Returns
    -------
    numpy array
        average errors of first predictions
    numpy array
        average errors of second predictions
    numpy array
        average absolute errors of first predictions
    numpy array
        average absolute errors of second predictions

    """

    # bar plot
    fig, axs = plt.subplots(2, 1, figsize=(7, 24))

    # first plot, mean absolute deviation
    labels = PRETTY_COLS
    # ignore "other" category
    labels = labels[:5]
    preds1 = preds1[:, :5]
    true_pops1 = true_pops1[:, :5]
    preds2 = preds2[:, :5]
    true_pops2 = true_pops2[:, :5]

    # x axis label locations
    xlocs = np.arange(len(labels))
    width = 0.35  # the width of the bars

    ma_errs1 = np.abs(preds1 - true_pops1)
    ma_errs1 = ma_errs1.sum(axis=0) / true_pops1.sum(axis=0)
    ma_errs2 = np.abs(preds2 - true_pops2)
    ma_errs2 = ma_errs2.sum(axis=0) / true_pops2.sum(axis=0)
    l1_err1 = np.sum(np.abs(preds1 - true_pops1)) / np.sum(true_pops1)
    l1_err2 = np.sum(np.abs(preds2 - true_pops2)) / np.sum(true_pops2)
    axs[1].bar(xlocs - width / 2, ma_errs1, width, label=label1, color="b")
    axs[1].bar(xlocs + width / 2, ma_errs2, width, label=label2, color="r")
    axs[1].bar(labels, 0)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[1].set_xticks(xlocs, labels)
    axs[1].legend(fontsize=26, frameon=False)
    axs[1].tick_params(axis="x", labelsize=18)
    axs[1].tick_params(axis="y", labelsize=18)

    # second plot, weighted average errors
    avg_errs1 = preds1.sum(axis=0) / true_pops1.sum(axis=0) - 1
    avg_errs2 = preds2.sum(axis=0) / true_pops2.sum(axis=0) - 1
    max_err = np.max(np.abs((avg_errs1, avg_errs2)))

    # if errors are 0, make them small, but non-zero so that a small bar appears
    fac = 0.004
    avg_errs1[np.abs(avg_errs1) < max_err * fac] = max_err * fac
    avg_errs2[np.abs(avg_errs2) < max_err * fac] = max_err * fac

    axs[0].bar(xlocs - width / 2, avg_errs1, width, label=label1, color="b")
    axs[0].bar(xlocs + width / 2, avg_errs2, width, label=label2, color="r")
    axs[0].bar(labels, 0)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[0].set_xticks(xlocs, labels)
    axs[0].legend(fontsize=26, frameon=False)
    ymax = np.max(np.abs([avg_errs1, avg_errs2])) + 0.05
    axs[0].axis(ymin=-ymax, ymax=ymax)
    axs[0].tick_params(axis="x", labelsize=18)
    axs[0].tick_params(axis="y", labelsize=18)

    # titles
    axs[1].set_title(
        f"Mean Absolute Deviation\n {label1}: {l1_err1:0.2}, {label2}: {l1_err2:0.2}",
        fontsize=35,
    )
    axs[0].set_title(f"Average Error", fontsize=35)

    fig.tight_layout(h_pad=6)
    fig.savefig(filename, bbox_inches="tight", dpi=fig.dpi)
    return avg_errs1, avg_errs2, ma_errs1, ma_errs2


def calib_table(filename, kstats, state, year):
    """
    create the kuiper statistics latex table

    Parameters
    ----------
    filename : string
        location where latex table will be written
    kstats : numpy array
        statistics to be included in table
    state : string
        state
    year : integer
        year

    Returns
    -------
    None
    """
    # ignore "other" category

    df_calib = pd.DataFrame(
        data=kstats[:, :5],
        columns=PRETTY_COLS[:5],
        index=[f"{state.upper()} {year} BISG", f"{state.upper()} {year} raking"],
    )
    table_str = df_calib.to_latex(float_format="%.4f")
    # write to file
    with open(filename, "w") as text_file:
        text_file.write(table_str)


def calib_plots(filename, df_agg, cols1, cols2, calib_map=None):
    """
    grid of five calibration plots, with three in the first row and two in
    the second

    Parameters
    ----------
    filename : string
        where plot will be written
    df_agg : pandas dataframe
        aggregated predictions
    cols1 : list
        column names with first set of predictions
    cols2 : list
        column names with second set of predictions
    calib_map : 6 x 6 numpy array, optional
        calibration matrix to be applied to predictions

    Returns
    -------
    numpy array
        kuiper statistics
    """

    f, axs = calib_subplots(width=30, height=18)

    # BISG calibration
    preds = df_agg[cols1].values
    if calib_map is not None:
        preds = np.dot(calib_map, preds.T).T
    # add noise to estimates since for calibration scheme all scores must be unique
    preds += np.random.normal(loc=0, scale=1e-7, size=np.shape(preds))

    # raking calibration
    preds2 = df_agg[cols2].values
    if calib_map is not None:
        preds2 = np.dot(calib_map, preds2.T).T
    # add noise to estimates since for calibration scheme all scores must be unique
    preds2 += np.random.normal(loc=0, scale=1e-7, size=np.shape(preds2))

    trues = df_agg[VF_RACES].values / df_agg["vf_tot"].values[:, np.newaxis]
    weights = df_agg["vf_tot"].values.astype(np.float64)

    # get true totals and predictions
    kstats = np.zeros((2, 6))
    for i, col in enumerate(VF_RACES[:5]):
        x, s, c, k = one_calib(preds[:, i], trues=trues[:, i], weights=weights)
        axs[i].plot(x, c, c="red", label="BISG", linewidth=4)
        kstats[0, i] = k

        x2, s2, c2, k2 = one_calib(preds2[:, i], trues=trues[:, i], weights=weights)
        axs[i].plot(x2, c2, c="blue", label="Raking", linewidth=4)
        kstats[1, i] = k2

        majorticks = 6
        ssub = np.insert(s, 0, [0])
        inds = np.linspace(0, len(ssub)-1, majorticks).astype(int)
        ss = ["{:.1e}".format(a) for a in ssub[inds].tolist()]
        ss = [lab[:-2] + lab[-1] for lab in ss]
        ss[0] = "0.0e0"
        axs[i].set_xticks(x[inds])
        axs[i].set_xticklabels(ss, rotation=45)

        axs[i].set_title(PRETTY_COLS[i], fontsize=48)
        axs[i].plot(x, np.zeros_like(x), c="black", linewidth=2)
        axs[i].tick_params(axis="x", labelsize=30)
        axs[i].tick_params(axis="y", labelsize=30)
        axs[i].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
        axs[i].yaxis.get_offset_text().set_fontsize(30)
        axs[i].xaxis.get_offset_text().set_fontsize(30)

    handles, labels = axs[0].get_legend_handles_labels()
    f.legend(
        handles,
        labels,
        loc="upper center",
        fontsize=48,
        bbox_to_anchor=(0.5, 1.15),
        fancybox=True,
        shadow=True,
        ncol=2,
    )
    # f.tight_layout(h_pad=6)

    f.text(
        -0.05,
        0.5,
        "Cumulative deviation from perfect calibration",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=54,
    )
    f.text(0.5, -0.11, "Predicted probability", ha="center", va="center", fontsize=54)
    plt.savefig(filename, bbox_inches="tight", dpi=f.dpi)

    return kstats


def fl_bar_plots(
    filename, preds1, true_pops1, preds2, true_pops2, label1="raking", label2="BISG"
):
    """
     bar plots with florida that include axis breaks. matplotlib doesn't have nice support for
     axis breaks, so they are done manually. they work by creating two plots, one on top of the
     other, then manually setting axis limits, removing the appropriate horizontal bars, and
     adding the marks for axis breaks. Since there are two bar plots in this figure, this means that
     four axes are dedicated to plotting. we also add another set of axes in the middle which is used
     to introduce space between the two plots

    Parameters
     ----------
     filename : string
         name of file that's written
     preds1 : n x m numpy array
         predictions
     true_pops1 : n x m numpy array
         the correct totals that preds is trying to predict
     preds2 : n x m numpy array, optional
         predictions
     true_pops2 : n x m numpy array, optional
         the correct totals that preds2 is trying to predict
     label1 : string, optional
         the label in the legend of the first set of predictions
     label2 : string, optional
         the label in the legend of the second set of predictions

     Returns
     -------
     numpy array
         mean absolute errors of prediction 1
     numpy array
         mean absolute errors of prediction 2
     numpy array
         average errors of prediction 1
     numpy array
         average errors of prediction 2
    """
    # create the five axes
    fig, axs = plt.subplots(
        5, 1, figsize=(7, 24), gridspec_kw={"height_ratios": [1, 5, 1.5, 1, 5]}
    )

    # first plot, mean absolute deviation
    labels = PRETTY_COLS

    # don't use "other" category
    labels = labels[:5]
    preds1 = preds1[:, :5]
    preds2 = preds2[:, :5]
    true_pops1 = true_pops1[:, :5]
    true_pops2 = true_pops2[:, :5]

    # x axis labels
    xlocs = np.arange(len(labels))
    # the width of the bars
    width = 0.35

    # compute average errors
    avg_errs1 = preds1.sum(axis=0) / true_pops1.sum(axis=0) - 1
    avg_errs2 = preds2.sum(axis=0) / true_pops2.sum(axis=0) - 1
    max_err = np.max(np.abs((avg_errs1, avg_errs2)))
    # when errors are negligible, still show a slight bar
    fac = 0.002
    avg_errs1[np.abs(avg_errs1) < max_err * fac] = max_err * fac
    avg_errs2[np.abs(avg_errs2) < max_err * fac] = max_err * fac
    # add bars to both top and bottom plots (above break and below)
    axs[0].bar(xlocs - width / 2, avg_errs1, width, label=label1, color="b")
    axs[0].bar(xlocs + width / 2, avg_errs2, width, label=label2, color="r")
    axs[0].bar(labels, 0)
    axs[1].bar(xlocs - width / 2, avg_errs1, width, label=label1, color="b")
    axs[1].bar(xlocs + width / 2, avg_errs2, width, label=label2, color="r")
    axs[1].bar(labels, 0)

    # zoom-in / limit the view to different portions of the data
    axs[0].set_ylim(1.8, 2.2)  # outliers only
    axs[1].set_ylim(-0.7, 0.5)  # most of the data

    # hide the spines between axes above and below break
    axs[0].spines["bottom"].set_visible(False)
    axs[1].spines["top"].set_visible(False)
    axs[0].xaxis.tick_top()
    axs[0].tick_params(labeltop=False, top=False)  # don't put tick labels at the top
    axs[0].set_yticks([2.0])
    axs[1].xaxis.tick_bottom()
    axs[1].tick_params(left=True, labeltop=False)  # don't put tick labels at the top
    # set font sizes of axes ticks
    axs[1].tick_params(axis="x", labelsize=16)
    axs[1].tick_params(axis="y", labelsize=16)
    axs[0].tick_params(axis="y", labelsize=16)

    # make the cut-out slanted lines to denote the break
    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    axs[0].plot([0, 1], [0, 0], transform=axs[0].transAxes, **kwargs)
    axs[1].plot([1, 0], [1, 1], transform=axs[1].transAxes, **kwargs)
    # title and legend
    axs[0].set_title(f"Average Error", fontsize=32)
    axs[0].legend(fontsize=26, loc="upper right", frameon=False)

    # the third set of axes exist for creating empty space between the top
    # plot and the bottom plot. turn the axes markings off
    axs[2].axis("off")

    # x axis label locations
    xlocs = np.arange(len(labels))  # the label locations

    # compute mean absolute errors
    ma_errs1 = np.abs(preds1 - true_pops1)
    ma_errs1 = ma_errs1.sum(axis=0) / true_pops1.sum(axis=0)
    ma_errs2 = np.abs(preds2 - true_pops2)
    ma_errs2 = ma_errs2.sum(axis=0) / true_pops2.sum(axis=0)
    l1_err1 = np.sum(np.abs(preds1 - true_pops1)) / np.sum(true_pops1)
    l1_err2 = np.sum(np.abs(preds2 - true_pops2)) / np.sum(true_pops2)

    # add bars to plot
    axs[3].bar(xlocs - width / 2, ma_errs1, width, label=label1, color="b")
    axs[3].bar(xlocs + width / 2, ma_errs2, width, label=label2, color="r")
    axs[3].bar(labels, 0)
    axs[4].bar(xlocs - width / 2, ma_errs1, width, label=label1, color="b")
    axs[4].bar(xlocs + width / 2, ma_errs2, width, label=label2, color="r")
    axs[4].bar(labels, 0)

    # zoom-in / limit the view to different portions of the data
    axs[3].set_ylim(1.8, 2.2)  # outliers only
    axs[4].set_ylim(0, 0.45)  # most of the data

    # hide the spines between ax and ax2
    axs[3].spines["bottom"].set_visible(False)
    axs[4].spines["top"].set_visible(False)
    axs[3].xaxis.tick_top()
    axs[3].tick_params(labeltop=False, top=False)  # don't put tick labels at the top
    axs[3].set_yticks([2.0])
    axs[4].xaxis.tick_bottom()
    axs[4].tick_params(left=True, labeltop=False)  # don't put tick labels at the top
    # set font sizes of axes ticks
    axs[4].tick_params(axis="x", labelsize=16)
    axs[3].tick_params(axis="y", labelsize=16)
    axs[4].tick_params(axis="y", labelsize=16)

    # make the cut-out slanted lines to denote the break
    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    axs[3].plot([0, 1], [0, 0], transform=axs[3].transAxes, **kwargs)
    axs[4].plot([1, 0], [1, 1], transform=axs[4].transAxes, **kwargs)

    # title and legend
    axs[3].set_title(
        f"Mean Absolute Deviation\n {label1}: {l1_err1:0.2}, {label2}: {l1_err2:0.2}",
        fontsize=32,
    )
    axs[3].legend(
        fontsize=26, loc="lower left", bbox_to_anchor=(0.0, -0.3), frameon=False
    )
    axs[3].legend(fontsize=26, loc="upper right", frameon=False)
    plt.subplots_adjust(wspace=0, hspace=0.07)

    # save figure
    fig.savefig(filename, bbox_inches="tight", dpi=fig.dpi)

    return ma_errs1, ma_errs2, avg_errs1, avg_errs2


def write_df_summary(filename, df_summary):
    """
    write a latex table of "df_summary", a dataframe with several relevant race distributions

    Parameters
    ----------
    filename : string
        where to write table
    df_summary : pandas dataframe
        distributions to include in table

    Returns
    -------
    None
    """
    table_str = df_summary.to_latex(float_format="%.3f", index=False)
    with open(filename, "w") as text_file:
        text_file.write(table_str)


def voterfile_state_table(filename, preds1vf, true_pops1vf):
    """
    this function writes a .tex file consisting of one table. that table
    provides the statewide subpopulation estimation for voterfile self-contained
    bisg which can only be run on data labeled with race/ethnicity

    Parameters
    ----------
    filename : string
        the name of the .tex file that's written
    preds1vf : numpy array
        predictions of 6 race/ethnicity categories for each (surname, geolocation) pair
    true_pops1vf : numpy array
        true populations predicted by preds1vf

    Returns
    -------
    None
    """
    # get marginal race/ethnicity totals for predictions and ground truth
    true_state = np.sum(true_pops1vf, axis=0)
    pred_state = np.sum(preds1vf, axis=0)
    # create a 4 x 6 numpy array of strings that will be printed
    d2 = np.empty((4, 6)).astype(str)
    d2[0] = [f"{int(t):,}" for t in true_state]
    d2[1] = [f"{int(t):,}" for t in pred_state]
    d2[2] = [f"{int(t):,}" for t in (true_state - pred_state)]
    d2[3] = [f"{t:.2%}" for t in (true_state - pred_state) / true_state]
    # create pandas dataframe
    df_tmp = pd.DataFrame(
        data=d2,
        columns=PRETTY_COLS,
        index=["True", "BISG", "Absolute Error", "Relative Error"],
    )
    table_str = df_tmp.to_latex(column_format="ccccccc")
    # write to file
    with open(filename, "w") as text_file:
        text_file.write(table_str)


def validation_l1l2ll_str_county(df_county):
    """
    take in a pandas dataframe with l1, l2, and negative log-likelihood in each county,
    and return a string with a latex table where that data is formatted nicely

    Parameters
    ----------
    df_county : pandas dataframe
        errors in each county

    Returns
    -------
    string
        table in latex form
    """
    # convert the dataframe to a latex table
    tab1 = df_county.to_latex(index=False, float_format="%.3f")
    # extract only the rows of the table, ignoring all formatting so that we can add our own
    ind = tab1.find("\\midrule\n")
    ind += len("\\midrule\n")
    body_str = tab1[ind:]
    header_str = """\\resizebox{!}{11cm}{ \\begin{tabular}{l | cc | cc | cc } & \\multicolumn{2}{|c|}{$\\ell^1$ 
    errors} & \\multicolumn{2}{|c|}{$\\ell^2$ errors} & \\multicolumn{2}{|c}{negative log-likelihood} \\\\ County &  
    raking &  BISG &  raking &  BISG &  raking &  BISG \\\\ \\hline"""
    # add hline between the last county and the row with the full state
    line_breaks = [i for i in range(len(body_str)) if body_str.startswith("\n", i)]
    ind = line_breaks[-4] + 1
    body_str = body_str[:ind] + "\\hline \n" + body_str[ind:]
    table_str = header_str + body_str + "} % resize box"
    return table_str


def validation_l1l2ll_str_region(df_region):
    """
    take in a pandas dataframe with l1, l2, and negative log-likelihood in each region,
    and return a string with a latex table where that data is formatted nicely

    Parameters
    ----------
    df_region : pandas dataframe
        errors in each region

    Returns
    -------
    string
        table in latex form
    """
    # convert the dataframe to a latex table
    tab1 = df_region.to_latex(index=False, float_format="%.3f")
    # extract only the rows of the table, ignoring all formatting so that we can add our own
    ind = tab1.find("\\midrule\n")
    ind += len("\\midrule\n")
    body_str = tab1[ind:]
    header_str = """\\resizebox{0.8\\textwidth}{!}{ \\begin{tabular}{l | cc | cc | cc } & \\multicolumn{2}{|c|}{
    $\\ell^1$ errors} & \\multicolumn{2}{|c|}{$\\ell^2$ errors} & \\multicolumn{2}{|c}{negative log-likelihood} \\\\ 
    Region &  raking &  BISG &  raking &  BISG &  raking &  BISG \\\\ \\hline"""
    # add hline between the last county and the row with the full state
    line_breaks = [i for i in range(len(body_str)) if body_str.startswith("\n", i)]
    ind = line_breaks[-4] + 1
    body_str = body_str[:ind] + "\hline \n" + body_str[ind:]
    table_str = header_str + body_str + "} % resize box"
    return table_str


def validation_err_table(df_agg, filename_county, filename_region, state, calib_map):
    """
    compute l1, l2 errors and negative log-likelihood in each county and region and
    turn that data into a latex table

    Parameters
    ----------
    df_agg : pandas dataframe
        aggregated predictions
    filename_county : string
        where county-level error table will be written
    filename_region : string
        where region-level error table will be written
    state : string
        state
    calib_map : 6 x 6 numpy array
        calibration map to apply to predictions

    Returns
    -------
    string
        latex table with county-level errors
    string
        latex table with region-level errors
    """

    # columns of df_agg to use for predictions
    cols = [RAKE_COLS, BISG_BAYES_COLS]

    # construct error tables
    df_region_l1l2, df_county_l1l2 = l1_l2_tables(state, df_agg, cols, calib_map)
    df_region_ll, df_county_ll = ll_tables(state, df_agg, cols, calib_map)

    # report errors aggregated at the county level
    df_county = df_county_l1l2.merge(df_county_ll)
    table_str = validation_l1l2ll_str_county(df_county)

    # report errors aggregated at the region level
    df_region = df_region_l1l2.merge(df_region_ll)
    table_region_str = validation_l1l2ll_str_region(df_region)

    # write to file
    with open(filename_county, "w") as text_file:
        text_file.write(table_str)
    with open(filename_region, "w") as text_file:
        text_file.write(table_region_str)

    return table_str, table_region_str


def vf_l1l2ll_str_county(df_county):
    """
    take in a pandas dataframe with l1, l2, and negative log-likelihood in each county
    for predictions using voter file-only predictions
    and return a string with a latex table where that data is formatted nicely

    Parameters
    ----------
    df_county : pandas dataframe
        errors in each county

    Returns
    -------
    string
        table in latex form
    """
    # convert the dataframe to a latex table
    tab1 = df_county.to_latex(index=False, float_format="%.3f")
    # extract only the rows of the table, ignoring all formatting so that we can add our own
    ind = tab1.find("\midrule\n")
    ind += len("\midrule\n")
    body_str = tab1[ind:]
    header_str = """\\resizebox{!}{11cm}{ \\begin{tabular}{l | rrrrr | rrrrr | rrrrr |} & \\multicolumn{5}{|c|}{
    $\\ell^1$ errors} & \\multicolumn{5}{|c|}{$\\ell^2$ errors} & \\multicolumn{5}{|c|}{negative log-likelihood} \\\\ 
    County &  rake2 &  rake3 &  BISG &  $ R \\mid G$ &  $R \\mid S$ &   rake2 &  rake3 &  BISG &  $ R \\mid G$ &  $R 
    \\mid S$ &  rake2 &  rake3 &  BISG &  $ R \\mid G$ &  $R \\mid S$  \\\\ \\hline"""
    # add an hline between the last county and the line with the full state
    line_breaks = [i for i in range(len(body_str)) if body_str.startswith("\n", i)]
    ind = line_breaks[-4] + 1
    body_str = body_str[:ind] + "\hline \n" + body_str[ind:]
    table_str = header_str + body_str + "} % resize box"

    return table_str


def vf_l1l2ll_str_region(df_region):
    """
    take in a pandas dataframe with l1, l2, and negative log-likelihood in each region
    for predictions using voter file-only predictions
    and return a string with a latex table where that data is formatted nicely

    Parameters
    ----------
    df_region : pandas dataframe
        errors in each region

    Returns
    -------
    string
        table in latex form
    """
    # convert the dataframe to a latex table
    tab1 = df_region.to_latex(index=False, float_format="%.3f")
    # extract only the rows of the table, ignoring all formatting so that we can add our own
    ind = tab1.find("\midrule\n")
    ind += len("\midrule\n")
    body_str = tab1[ind:]
    header_str = """\\resizebox{\\textwidth}{!}{ \\begin{tabular}{l | rrrrr | rrrrr | rrrrr } & \\multicolumn{5}{
    |c|}{$\\ell^1$ errors} & \\multicolumn{5}{|c|}{$\\ell^2$ errors} & \\multicolumn{5}{|c}{negative log-likelihood} 
    \\\\ Region &  rake2 &  rake3 &  BISG &  $ R \\mid G$ &  $R \\mid S$ &   rake2 &  rake3 &  BISG &  $ R \\mid G$ & 
     $R \\mid S$ &  rake2 &  rake3 &  BISG &  $ R \\mid G$ &  $R \\mid S$  \\\\ \\hline"""
    # add an hline between the last region and the line with the full state
    line_breaks = [i for i in range(len(body_str)) if body_str.startswith("\n", i)]
    ind = line_breaks[-4] + 1
    body_str = body_str[:ind] + "\hline \n" + body_str[ind:]
    table_region_str = header_str + body_str + "} % resize box"

    return table_region_str


def vf_err_tables(filename_county, filename_region, df_agg, state):
    """
    write to file a table in latex form with the l1, l2 errors and negative
    log-likelihood in each county for the voter file-only predictions.
    this function includes both the county-level errors and the region-level errors.

    Parameters
    ----------
    df_agg : pandas dataframe
        aggregated predictions
    filename_county : string
        where county-level error table will be written
    filename_region : string
        where region-level error table will be written
    state : string
        state

    Returns
    -------
    None
    """
    cols = [
        VF_RAKE2_COLS,
        VF_RAKE3_COLS,
        VF_BISG_COLS,
        VF_R_GIVEN_GEO_COLS,
        VF_R_GIVEN_SUR_COLS,
    ]

    # construct error tables
    df_region_l1l2, df_county_l1l2 = l1_l2_tables(state, df_agg, cols, calib_map=None)
    df_region_ll, df_county_ll = ll_tables(state, df_agg, cols, calib_map=None)

    # merge l1, l2, and log-likelihood
    df_county = df_county_l1l2.merge(df_county_ll, on=["county"])
    table_str = vf_l1l2ll_str_county(df_county)

    # report errors aggregated at the region level
    # merge l1, l2, and log-likelihood
    df_region = df_region_l1l2.merge(df_region_ll)
    table_region_str = vf_l1l2ll_str_region(df_region)

    # write to file
    with open(filename_county, "w") as text_file:
        text_file.write(table_str)
    with open(filename_region, "w") as text_file:
        text_file.write(table_region_str)


if __name__ == "__main__":
    main()
