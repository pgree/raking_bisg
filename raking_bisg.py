"""
this module includes the primary statistical tools of this repository. it contains codes for creating
various predictions for (surname, county) pairs in various states in the united states.
the primary output of this code is a pandas dataframe (called df_agg), for one state,
that has various "predictions" of race/ethnicity given surname and geolocation. Each row
corresponds to one (surname, county) pair.

Functions
---------
make_df_agg
    construct the dataframe of race/ethnicity predictions for a given state
load_voter_file
    get the voter file
make_calib_map
    calibration map by solving an optimization problem
dist_summary
    pandas dataframe that has several useful subpopulation distributions
one_calib
    a plot of tygert cumulative miscalibration
ll_tables
    tables with negative log-likelihood
l1_l2_tables
    tables with l1, l2 errors
get_v_given_r
    (proportional to) probability of being a voter given race
subpopulation_preds
    predictions for subpopulation estimates over geographical regions
ipf2d
    two-way raking
ipf3d
    three-way raking
two_way_tables
    build the three two-way margins of (surname, geolocation, race) table

"""
import os
import numpy as np
import pandas as pd
from tabulate import tabulate
import cvxpy as cp

from fields import *
import process_census
import process_cps
import washington_vf as wa_vf
import vermont_vf as vt_vf
import oklahoma_vf as ok_vf
import ohio_vf
import north_carolina_vf as nc_vf
import new_york_vf as ny_vf
import florida_vf

# ignore certain numpy errors
np.seterr(divide="ignore", invalid="ignore")


def main():
    state = "fl"
    year = 2020
    df_agg = make_df_agg(state, year, subsample=False, load=True)
    name = "takriti"
    mask = df_agg["name"] == name
    df_tmp = df_agg.loc[mask, ["name", "county", "vf_tot"] + RAKE_COLS]
    print(tabulate(df_tmp, headers="keys", tablefmt="psql"))


def write_dataverse_files():
    """
    write df_agg dataframes that are put online. these only include names that appear at least
    five times in the state.
    """
    state = "nc"
    year = 2020
    states = ["fl", "ny", "nc", "nc", "oh", "ok", "vt", "wa"]
    years = [2020, 2020, 2020, 2010, 2020, 2020, 2020, 2020]
    for state, year in zip(states, years):
        filename = f"generated_data/df_agg_{state}{year}.feather"
        df_agg = pd.read_feather(filename)
        df_agg = df_agg.drop(columns=["vf_tot"])
        # cols_to_keep = ['name', 'county'] + BISG_CEN_COUNTY_COLS + BISG_BAYES_COLS + RAKE_COLS
        # df_agg = df_agg[cols_to_keep]

        # filter out names that don't appear frequently enough (~1mil)
        min_names = 5
        df_tmp = df_agg.groupby(["name"]).size()
        df_tmp = df_tmp.to_frame("count").reset_index()
        df_tmp = df_tmp.loc[(df_tmp["count"] >= min_names), :]
        names = df_tmp["name"].unique()
        df_agg = df_agg.loc[df_agg["name"].isin(names), :]
        print(df_agg.shape)
        file_out = f"generated_data/df_agg_{state}{year}_dataverse.pkl"
        df_agg.to_pickle(file_out)


def make_df_vf_sub(df_vf, df_cps_reg_voters, verbose=False):
    """
    subsample the voter file such that the race distribution of the subsampled
    voter file coincides with the race distribution of the cps (input df_cps_reg_voters)

    Parameters
    ----------
    df_vf : pandas dataframe
        the original voter file with a "race" columns
    df_cps_reg_voters : pandas dataframe
        race distribution of registered voters estimated by cps
    verbose : bool, optional

    Returns
    -------
    pandas dataframe
        the subsampled voter file
    """

    # cps race/ethnicity distribution
    dist_cps = df_cps_reg_voters[RACES].values[0]
    dist_vf = df_vf.groupby("race").size()[RACES].values
    dist_vf = dist_vf / np.sum(dist_vf)
    ratios = dist_cps / dist_vf
    # find race with maximum ration of cps / voter file
    ind_max = np.argmax(ratios)

    # construct adjusted voterfile totals that would match cps distribution
    vf_count_dist = df_vf.groupby("race").size()[RACES].values
    vf_count_dist_adj = dist_cps * vf_count_dist[ind_max] / dist_cps[ind_max]
    vf_count_dist_adj = vf_count_dist_adj.astype(int)

    # subsample each race separately and combine dataframes at the end
    dfs = []
    for i, race in enumerate(RACES):
        df_tmp = df_vf[df_vf["race"] == race]
        nsample = int(vf_count_dist_adj[i])
        dfs.append(df_tmp.sample(n=nsample, replace=False, random_state=1))
    df_vf_sub = pd.concat(dfs)

    if verbose:
        print(f"initial cps distribution: {dist_cps}")
        print(f"initial voter file distribution: {dist_vf}")
        print(
            f"voter file target distribution: {vf_count_dist_adj / np.sum(vf_count_dist_adj)}"
        )
        print(
            f'final voter file distribution: {df_vf_sub.groupby("race").size()[RACES].values / df_vf_sub.shape[0]}'
        )

    return df_vf_sub


def load_voter_file(state, year, load=True):
    """
    load the voter file of the inputted state (and in the case of north carolina, year)

    Parameters
    ----------
    state : string
        state, e.g., 'fl'
    year : integer
        year, only relevant for north carolina
    load : bool, optional
        load the voter file if it has already been written

    Returns
    -------
    pandas dataframe
        the state voter file
    """

    states = ["fl", "nc", "ny", "oh", "ok", "vt", "wa"]
    assert state in states, f"no voter file for {state}"

    # voter file
    if state == "fl":
        df_vf = florida_vf.fl_voter_file(min_names=0, verbose=False, load=load)
    elif state == "nc":
        df_vf = nc_vf.nc_voter_file(verbose=False, year=year, load=load)
    elif state == "ny":
        df_vf = ny_vf.ny_voter_file(load=load)
    elif state == "oh":
        df_vf = ohio_vf.oh_voter_file(load=load)
    elif state == "ok":
        df_vf = ok_vf.ok_voter_file(load=load)
    elif state == "vt":
        df_vf = vt_vf.vt_voter_file(load=load)
    elif state == "wa":
        df_vf = wa_vf.wa_voter_file(load=load)
    else:
        raise Exception(f"no voter file for {state}")

    return df_vf


def make_df_agg(state, year, subsample=False, load=False):
    """
    create the dataframe df_agg which contains various predictions for all
    registered voters in a state. each row of df_agg corresponds to
    one (surname, geolocation)

    Parameters
    ----------
    state : string
        state
    year : integer
        year
    subsample : bool, optional
        construct subsampled df_agg where the race distribution of all people
        matches the cps survey approximation
    load : bool, optional
        load df_agg if it has already been created and written to file

    Returns
    -------
    pandas dataframe
        df_agg
    """

    # check if df_agg has already been created
    if subsample:
        filename_df_agg = f"generated_data/df_agg_{state}{year}_sub.feather"
    else:
        filename_df_agg = f"generated_data/df_agg_{state}{year}.feather"

    # if df_agg has already been created, load and return it
    if os.path.exists(filename_df_agg) and load:
        print(f"*loading dataframe from {filename_df_agg}")
        df_agg = pd.read_feather(filename_df_agg)
        return df_agg

    # load cps data
    df_cps_reg_voters = process_cps.cps_geo_race(
        state, year, voters=True, load=True, save=True
    )
    # load census data
    df_cen_counties = process_census.county_census(state, year, load=True)
    df_cen_usa = process_census.usa_census(year=year)
    df_cen_surs = process_census.surnames()

    # voter file
    df_vf = load_voter_file(state, year, load=True)

    # check if voter file contains race labels
    race_labels = "race" in df_vf.columns

    # subsample voter file?
    if subsample and race_labels:
        print("*subampling voter file")
        df_vf = make_df_vf_sub(df_vf, df_cps_reg_voters, verbose=False)

    # initialize df_agg, which will contain various name/county predictions
    df_agg = construct_df_agg_init(state, df_vf)

    # name in census surname list
    df_agg["in_cen_surs"] = df_agg["name"].isin(df_cen_surs["name"].unique())

    # add census data for surname, geolocation to df_agg
    df_agg = process_census.append_r_given_g_cols(df_agg, df_cen_counties)
    df_agg = process_census.append_r_given_s_cols(df_agg, df_cen_surs)

    # add voter file race columns to df_agg if we have labels
    if race_labels:
        print("*appending to df_agg voter file race information")
        df_agg = append_vf_race_cols(df_agg, df_vf)

    # black box bisg
    df_agg = black_box_bisg(df_agg, df_cen_counties, df_cen_surs, df_cen_usa)

    # bisg for voters
    print("*calculating voter bisg using cps")
    df_agg = append_voter_bisg(df_agg, df_cps_reg_voters, df_cen_counties, df_cen_surs)

    # raking w/ BISG initialization
    print("*raking predictions to match cps and voter file margins")
    df_agg = append_raking_preds(df_agg, df_cps_reg_voters)

    # write df_agg to file
    df_agg.to_feather(filename_df_agg)

    return df_agg


def make_calib_map(df_cps_reg_voters, df_agg, verbose=False):
    """
    find the stochastic 6 x 6 matrix a such that a * u = v
    that minimizes | a - I |_F where u and v are the race distribution
    of the cps and voter file respectively. by stochastic matrix, we mean
    entries are non-negative and each column sums to 1

    Parameters
    ----------
    df_cps_reg_voters : pandas dataframe
        race distribution of registered voters in a state approximated by cps
    df_agg : pandas dataframe
        df_agg
    verbose : bool, optional

    Returns
    -------
    6 x 6 numpy array
        matrix that maps voter file distribution to cps distribution
    """
    # create map from CPS predictions to VF predictions
    print("*constructing calibration map")
    u = df_cps_reg_voters[RACES].values.T
    # voter file race distribution
    df_tmp = df_agg[VF_RACES]
    v = df_tmp.sum(axis=0).div(df_tmp.sum(axis=0).sum()).values.reshape((-1, 1))
    cps_to_vf_map, _ = cps_to_vf_matrix(u, v, verbose=verbose)
    return cps_to_vf_map


def dist_summary(state, year, df_cen_usa, df_cen_counties, df_vf, df_cps_reg_voters):
    """
    create a dataframe with the race distribution of several important populations:
    the full country, the state, the 18+ state, the cps estimate of registered voters,
    and if available, the race distribution of registered voters from the voter file

    Parameters
    ----------
    state : string
        state
    year : integer
        year
    df_cen_usa : pandas dataframe
        race distribution of all americans
    df_cen_counties : pandas dataframe
        race distribution of all counties in state
    df_vf : pandas dataframe
        voter file
    df_cps_reg_voters : pandas dataframe
        race distribution of registered voters in state

    Returns
    -------
    pandas dataframe
        several relevant race distributions
    """
    # census, usa
    dist_usa = df_cen_usa[RACES] / df_cen_usa[RACES].sum().sum()

    # census, state
    dist_cen_state = (
        df_cen_counties[RACES].sum(axis=0) / df_cen_counties[RACES].sum().sum()
    )
    dist_cen_state = pd.DataFrame(dist_cen_state[RACES]).T

    # census, state, over18
    dist_cen18_state = (
        df_cen_counties[RACES18].sum(axis=0) / df_cen_counties[RACES18].sum().sum()
    )
    dist_cen18_state = pd.DataFrame(dist_cen18_state[RACES18]).T
    dist_cen18_state.columns = RACES

    # voter file, state
    if "race" in df_vf.columns:
        dist_vf_state = (
            df_vf.groupby("race").size() / df_vf.groupby("race").size().sum()
        )
        dist_vf_state = pd.DataFrame(dist_vf_state[RACES]).T

        df = pd.concat(
            [
                dist_usa,
                dist_cen_state,
                dist_cen18_state,
                df_cps_reg_voters,
                dist_vf_state,
            ]
        )
        df["Population"] = [
            f"USA census {year}",
            f"{state.upper()} {year} census",
            f"{state.upper()} {year} 18+ census",
            f"{state.upper()} {year} CPS voters",
            f"{state.upper()} {year} voter file",
        ]
    else:
        df = pd.concat([dist_usa, dist_cen_state, dist_cen18_state, df_cps_reg_voters])
        df["Population"] = [
            f"USA {year} census",
            f"{state.upper()} {year} census",
            f"{state.upper()} {year} 18+ census",
            f"{state.upper()} {year} CPS voters",
        ]
    # reorder columns so population is first
    df = df[["Population"] + RACES]
    # and make the race columns pretty
    df.columns = ["Population"] + PRETTY_COLS

    return df


def one_calib(preds, trues, weights):
    """
    one tygert calibration plot. this is basically a copy/paste of the code here --
    https://github.com/facebookresearch/fbcdgraph/blob/main/codes/calibr_weighted.py
    (as of 3/30/2023)

    Parameters
    ----------
    preds : numpy array
        prediction with probabilities that various people are of one particular race
    trues : numpy array
        actual probability that various people are of one particular race
    weights : numpy array
        number of people corresponding to each entry of preds and trues

    Returns
    -------
    numpy array
        ordered weights with a 0 inserted at the beginning
    numpy array
        ordered predictions
    numpy array
        cumulative miscalibration
    float
        kuiper statistic
    """
    w = weights.copy()
    w /= w.sum()

    # sort
    args = np.argsort(preds)
    r = trues[args]
    s = preds[args]
    w = w[args]

    # Accumulate the weighted r and s, as well as w.
    f = np.insert(np.cumsum(w * r), 0, [0])
    ft = np.insert(np.cumsum(w * s), 0, [0])
    x = np.insert(np.cumsum(w), 0, [0])
    fft = f - ft
    kuiper = np.max(fft) - np.min(fft)

    return x, s, fft, kuiper


def ll_tables(state, df_agg, cols, calib_map=None):
    """
    create a pandas dataframe with the negative log-likelihood of subpopulation predictions.
    specifically, in each geographical region, g, sum over all surnames and races the following

        -x_{sgr} * log(m_{sgr} / x_{+g+}

    where x_{sgr} is the correct number of people of a particular surname, geolocation, race,
    and m_{sgr} is the prediction.

    Parameters
    ----------
    state : string
        state
    df_agg : pandas dataframe
        dataframe with predictions
    cols : list
        the columns with the (normalized) predictions
    calib_map : 6 x 6 numpy array, optional
        apply a calibration map to the predictions

    Returns
    -------
    pandas dataframe
        negative loglikelihood at region level
    pandas dataframe
        negative loglikelihood at county level
    """

    # create list of columns
    agg_cols = ["county", "region", "vf_tot"] + VF_RACES
    for col_list in cols:
        agg_cols = agg_cols + col_list
    df_tmp = df_agg.loc[:, agg_cols]

    # filter out rows without predictions
    mask = ~df_tmp[agg_cols].isnull().any(axis=1)
    df_tmp = df_tmp.loc[mask, :]

    # create columns with lls
    ncomps = len(cols)
    ll_cols = [f"ll_{cols[i][0][:-8]}" for i in range(ncomps)]
    # for each prediction scheme, compute errors
    for i in range(ncomps):
        # compute predictions
        if calib_map is not None:
            arrtmp = np.dot(calib_map, df_tmp[cols[i]].values.T).T
        else:
            arrtmp = df_tmp[cols[i]].values

        arrtmp = np.log10(arrtmp)
        arrtmp[arrtmp == -np.inf] = 0
        arrtmp = np.sum(df_tmp[VF_RACES].values * arrtmp, axis=1)
        df_tmp[ll_cols[i]] = -arrtmp

    # florida-wide
    df_tmp_state = df_tmp[ll_cols].sum() / df_tmp["vf_tot"].sum()
    if state == "nc":
        df_tmp_state["region"] = "North Carolina"
        df_tmp_state["county"] = "North Carolina"
    elif state == "fl":
        df_tmp_state["region"] = "Florida"
        df_tmp_state["county"] = "Florida"

    # regions
    df_region = df_tmp.groupby("region").sum()
    df_region = df_region.div(df_region["vf_tot"].values[:, np.newaxis])
    df_region = df_region[ll_cols].reset_index()
    df_region = df_region.append(df_tmp_state[["region"] + ll_cols], ignore_index=True)
    df_region["region"] = df_region["region"].str.title()

    # counties
    df_tmp_county = df_tmp.groupby("county").sum()
    df_tmp_county = df_tmp_county.div(df_tmp_county["vf_tot"].values[:, np.newaxis])
    df_tmp_county = df_tmp_county[ll_cols].reset_index()

    # replace county abbreviations with full county name
    if state == "fl":
        df_tmp_county["county"] = df_tmp_county["county"].map(florida_vf.COUNTY_DICT)
    df_tmp_county = df_tmp_county.sort_values("county")
    df_tmp_county = df_tmp_county.append(
        df_tmp_state[["county"] + ll_cols], ignore_index=True
    )

    return df_region, df_tmp_county


def l1_l2_tables(state, df_agg, cols, calib_map=None):
    """
    create a pandas dataframe with the l1 and l2 errors of subpopulation predictions.
    specifically, for a geographical region, g, sum over all surnames and races

       l1:  |x_{sgr} - m_{sgr}| / x_{+g+}
       l2:  \frac{1}{x_{+g+}} \sum_{s} ||x_{sgr} - m_{sgr}||_2

    where x_{sgr} is the correct number of people of a particular surname, geolocation, race,
    and m_{sgr} is the prediction.

    Parameters
    ----------
    state : string
        state
    df_agg : pandas dataframe
        dataframe with predictions
    cols : list
        the columns with the (normalized) predictions
    calib_map : 6 x 6 numpy array, optional
        apply a calibration map to the predictions

    Returns
    -------
    pandas dataframe
        l1 and l2 errors at region level
    pandas dataframe
        l1 and l2 errors at county level
    """

    # create list of columns
    agg_cols = ["county", "region", "vf_tot"]
    if VF_BAYES_OPT_COLS not in cols:
        agg_cols += VF_BAYES_OPT_COLS
    for colsi in cols:
        agg_cols = agg_cols + colsi
    df_tmp = df_agg.loc[:, agg_cols]

    # filter out rows without predictions
    mask = ~df_tmp[agg_cols].isnull().any(axis=1)
    df_tmp = df_tmp.loc[mask, :]

    ncomps = len(cols)
    # names of columns
    l1_cols = [f"l1_{cols[i][0][:-8]}" for i in range(ncomps)]
    l2_cols = [f"l2_{cols[i][0][:-8]}" for i in range(ncomps)]
    # for each prediction scheme, compute errors
    for i in range(ncomps):
        # compute unweighted errors for each (surname, geolocation)
        if calib_map is not None:
            arrtmp = (
                df_tmp[VF_BAYES_OPT_COLS].values
                - np.dot(calib_map, df_tmp[cols[i]].values.T).T
            )
            # if we're checking accuracy of the ground truth for debugging purposes,
            # then don't apply the calibration map
            if cols[i] == VF_BAYES_OPT_COLS:
                arrtmp = df_tmp[VF_BAYES_OPT_COLS].values - df_tmp[cols[i]].values
        else:
            arrtmp = df_tmp[VF_BAYES_OPT_COLS].values - df_tmp[cols[i]].values
        # scale errors by number of appearances of (surname, geolocation)
        arrtmp = arrtmp.astype(np.float64)
        df_tmp[l1_cols[i]] = np.sum(np.abs(arrtmp), axis=1) * df_tmp["vf_tot"].values
        df_tmp[l2_cols[i]] = (
            np.sqrt(np.sum(arrtmp**2, axis=1)) * df_tmp["vf_tot"].values
        )

    # florida-wide, this will be added as the last row of the county/region dataframes
    df_tmp_state = df_tmp[l1_cols + l2_cols].sum() / df_tmp["vf_tot"].sum()
    if state == "nc":
        df_tmp_state["region"] = "North Carolina"
        df_tmp_state["county"] = "North Carolina"
    elif state == "fl":
        df_tmp_state["region"] = "Florida"
        df_tmp_state["county"] = "Florida"

    # regions
    df_region = df_tmp.groupby("region").sum()
    df_region = df_region.div(df_region["vf_tot"].values[:, np.newaxis])
    df_region = df_region[l1_cols + l2_cols].reset_index()
    df_region = df_region.append(
        df_tmp_state[["region"] + l1_cols + l2_cols], ignore_index=True
    )
    df_region["region"] = df_region["region"].str.title()

    # counties
    df_county = df_tmp.groupby("county").sum()
    df_county = df_county.div(df_county["vf_tot"].values[:, np.newaxis])
    df_county = df_county[l1_cols + l2_cols].reset_index()

    # replace county abbreviations with full county name
    if state == "fl":
        df_county["county"] = df_county["county"].map(florida_vf.COUNTY_DICT)
    df_county = df_county.sort_values("county")
    df_county = df_county.append(
        df_tmp_state[["county"] + l1_cols + l2_cols], ignore_index=True
    )

    return df_region, df_county


def construct_df_agg_init(state, df_vf):
    """
    construct a dataframe with columns "name", "county", "vf_tot", and in the case of labeled voter file, VF_RACES

    Parameters
    ----------
    state : string
        abbreviation of one of the states supported by this package
    df_vf : pandas dataframe
        name, county, and possibly race of registered voters

    Returns
    -------
    pandas dataframe
        containing name, county, total appearances in voter file, possible race totals

    """
    df_agg = df_vf.copy()

    # is race is labeled, then create a column for each race corresponding to the
    # number of people of that race of a particular (surname, geolocation) pair
    if "race" in df_vf.columns:
        df_agg[VF_RACES] = 0
        # add race columns with hot-one encoding
        for i, race in enumerate(RACES):
            df_agg[VF_RACES[i]] = df_agg["race"] == race
        # create dataframe with county, name, race columns
        df_agg = df_agg.groupby(["name", "county"]).sum().reset_index()
        # total number of people with a given surname in a given region
        df_agg["vf_tot"] = df_agg[VF_RACES].sum(axis=1)
    else:
        # create dataframe with name and county
        df_agg = df_agg.groupby(["name", "county"]).size().reset_index()
        df_agg.columns = ["name", "county", "vf_tot"]

    # add regions
    if state == "fl":
        df_agg["region"] = df_agg["county"].map(
            lambda x: florida_vf.COUNTY_DICT.get(x, x)
        )
        df_agg["region"] = df_agg["region"].map(
            lambda x: florida_vf.COUNTY_TO_REGION_DICT.get(x, x)
        )
    elif state == "nc":
        df_agg["region"] = df_agg["county"].map(
            lambda x: nc_vf.COUNTY_TO_REGION_DICT.get(x, x)
        )

    return df_agg


def append_vf_race_cols(df_agg, df_vf, verbose=True):
    """
    append columns to df_agg that require the race labels of, e.g. florida and north carolina's
    voter file.

    Parameters
    ----------
    df_agg : pandas dataframe
    df_vf : pandas dataframe
    verbose : bool, optional

    Returns
    -------
    pandas dataframe
        df_agg
    """
    # two-way margins
    df_vf_sr, _, df_vf_gr = two_way_tables(df_vf)
    # vf bayes optimal, race given surname, race given geo, bisg
    df_agg = basic_cols(df_agg, df_vf_sr, df_vf_gr)
    # vf bayes 2way and 3way raking
    if verbose:
        print("*two-way raking on voter file")
    df_agg = two_way_raking(df_agg)
    if verbose:
        print("*three-way raking on voter file")
    df_agg = three_way_raking(df_agg, tol=1e-3)
    return df_agg


def black_box_bisg(df_agg, df_cen_counties, df_cen_surs, df_cen_usa):
    """
    append black box bisg (full usa population) columns to df_agg

    Parameters
    ----------
    df_agg : pandas dataframe
    df_cen_counties : pandas dataframe
    df_cen_surs : pandas dataframe
    df_cen_usa : pandas dataframe

    Returns
    -------
    pandas dataframe
        df_agg
    """
    # add census data for surname, geolocation to df_agg
    df_agg = process_census.append_r_given_g_cols(df_agg, df_cen_counties)
    df_agg = process_census.append_r_given_s_cols(df_agg, df_cen_surs)

    # census bisg, compute three factors in bisg formula
    prob_of_r = df_cen_usa[PROB_COLS].values
    r_given_surname = df_agg[CEN_R_GIVEN_SUR_COLS].values
    r_given_geo = df_agg[CEN_R_GIVEN_GEO_COLS].values

    # construct prediction (county-level)
    bisg_preds = r_given_geo * r_given_surname / prob_of_r
    bisg_preds = bisg_preds / np.sum(bisg_preds, axis=1)[:, np.newaxis]
    df_agg[BISG_CEN_COUNTY_COLS] = bisg_preds

    return df_agg


def bisg_np(prob_of_r, r_given_surname, r_given_geo):
    """
    compute the bisg formula

    Parameters
    ----------
    prob_of_r : numpy array of length m
        probability of each race in population
    r_given_surname : n x m numpy array
        probability of race given surname
    r_given_geo : n x m numpy array
        probability of race given geolocation

    Returns
    -------
    n x m numpy array
        bisg predictions
    """
    bisg_preds = r_given_geo * r_given_surname / prob_of_r
    bisg_preds = bisg_preds / np.sum(bisg_preds, axis=1)[:, np.newaxis]
    return bisg_preds


def get_v_given_r(df_cps_reg_voters, df_cen_counties):
    """
    compute P(V | R), or rather something proportional to that via the formula
    C(R, V) / C(R) where C(R, V) is, up to a multiplicative
    constant independent of R, equal to the number of voters of each race obtained
    from the cps. C(R), the count of each race comes from the census
    over-18 population

    Parameters
    ----------
    df_cps_reg_voters : pandas dataframe
        subpopulation distribution of registered voters
    df_cen_counties : pandas dataframe
        subpopulation distribution in each county

    Returns
    -------
    numpy array
        probability of being a registered voter given race, up to a constant
    """
    # P(R, V) up to a constant independent of R
    rv_dist = df_cps_reg_voters[RACES].values
    # P(R) up to a constant independent of R
    r_counts = df_cen_counties[RACES18].values.sum(axis=0)
    # P(V | R) up to a constant independent of R
    v_given_r = rv_dist / r_counts
    # normalize so numbers aren't tiny
    v_given_r /= np.sum(v_given_r)

    return v_given_r


def append_r_given_gv(df_agg, df_cen_counties, v_given_r):
    """
    add to df_agg columns with
    estimate for P(R | G, V) using the conditional independence assumption
    gives the approximation P(V | R) * P(R | G). (side note: this happens to be
    equivalent to one step of ipf)

    Parameters
    ----------
    df_agg : pandas dataframe
        aggregated predictions
    df_cen_counties : pandas dataframe
        census race distribution in each county in state
    v_given_r : numpy array
        proportional to probability of being a voter given race

    Returns
    -------
    pandas dataframe
        df_agg with new columns
    """
    preds_new = df_cen_counties[PROB18_RACES].values * v_given_r
    preds_new /= np.sum(preds_new, axis=1)[:, np.newaxis]

    # create dataframe with new predictions
    df_cps_bayes_r_given_g = pd.DataFrame(data=preds_new, columns=RACES)
    df_cps_bayes_r_given_g["county_code"] = df_cen_counties["county"].values
    df_cps_bayes_r_given_g["total"] = df_cen_counties["total"].values

    # add to df_agg
    for race in RACES:
        dict1 = df_cps_bayes_r_given_g.set_index("county_code").to_dict()[race]
        df_agg[f"cps_bayes_r_given_geo_{race}"] = df_agg["county"].map(
            lambda x: dict1.get(x, x)
        )

    return df_agg


def append_r_given_sv(df_agg, df_cen_surs, v_given_r):
    """
    add to df_Agg columns with
    estimate for P(R | G, V) using the conditional independence assumption
    gives the approximation P(V | R) * P(R | G). (side note: this happens to be
    equivalent to one step of ipf)

    Parameters
    ----------
    df_agg : pandas dataframe
        aggregated predictions
    df_cen_surs : pandas dataframe
        census race distribution for each surname
    v_given_r : numpy array
        proportional to probability of being a voter given race

    Returns
    -------
    pandas dataframe
        df_agg with new columns
    """
    preds_new = df_cen_surs[CEN_R_GIVEN_SUR_COLS].values * v_given_r
    preds_new /= np.sum(preds_new, axis=1)[:, np.newaxis]
    df_cps_bayes_r_given_sur = pd.DataFrame(data=preds_new, columns=RACES)
    # print(df_cps_bayes_r_given_sur)
    df_cps_bayes_r_given_sur["name"] = df_cen_surs["name"].values
    df_cps_bayes_r_given_sur["count"] = df_cen_surs["count"].values

    # add to df_agg
    for race in RACES:
        col = f"cps_bayes_r_given_sur_{race}"
        dict1 = df_cps_bayes_r_given_sur.set_index("name").to_dict()[race]
        df_agg[col] = df_agg["name"].map(lambda x: dict1.get(x, x))

    # when a name isn't in census surnames, use "all other names" for race given surname
    if "all other names" in df_cen_surs["name"].unique():
        preds_other = (
            df_cen_surs.loc[
                df_cen_surs["name"] == "all other names", CEN_R_GIVEN_SUR_COLS
            ].values[0]
            * v_given_r
        )
        preds_other = preds_other / np.sum(preds_other)
        mask = ~df_agg["name"].isin(df_cen_surs["name"].unique())
        df_agg.loc[mask, CPS_BAYES_R_GIVEN_SUR_COLS] = preds_other

    # set columns to floats
    df_agg[CPS_BAYES_R_GIVEN_SUR_COLS] = df_agg[CPS_BAYES_R_GIVEN_SUR_COLS].astype(
        "float64"
    )

    return df_agg


def append_raking_preds(df_agg, df_cps_reg_voters):
    """
    take as input the pandas dataframe df_agg and add to it raking predictions.
    the margins being raked to are the cps estimate for the race distribution of
    registered voters in a state, and the joint distribution of (surname, geolocation)

    Parameters
    ----------
    df_agg : pandas dataframe
        must contain columns BISG_BAYES_COLS and 'vf_tot' (the total number of people of a
        particular (name, county) pair)
    df_cps_reg_voters : pandas dataframe
        race distribution of registered voters

    Returns
    -------
    pandas dataframe
        df_agg, the inputted df_agg with raking predictions appended in columns RAKE_COLS
    """

    # initialization of raking, bisg predictions
    init = df_agg.loc[:, BISG_BAYES_COLS].values
    row_margin = df_cps_reg_voters[RACES].values * df_agg.loc[:, "vf_tot"].sum()
    col_margin = df_agg.loc[:, "vf_tot"].values
    preds_new, _, _ = ipf2d(
        init, row_margin, col_margin, tol=1e-6, iter_max=1000, normalize=True
    )
    # append predictions to df_agg
    df_agg.loc[:, RAKE_COLS] = preds_new

    return df_agg


def append_voter_bisg(df_agg, df_cps_reg_voters, df_cen_counties, df_cen_surs):
    """
    take as input the pandas dataframe df_agg and add to it bisg predictions
    of registered voters. the formula is the usual bisg formula for the full population
    with a factor of P(V | R) (the probability of being a registered voter given race).
    that is, the formula implemented by this function is:

        $$P(R | G, S, V) \propto P(V | R) P(R | G) P(R | S) / P(R)$$

    Parameters
    ----------
    df_agg : pandas dataframe
        with columns CPS_BAYES_R_GIVEN_GEO, CPS_BAYES_R_GIVEN_SUR cols
    df_cps_reg_voters : pandas dataframe
         race distribution of registered voters with columns RACES
    df_cen_counties : pandas dataframe
        race distribution for each county from census
    df_cen_surs : pandas dataframe
        race distribution for each surname in the census list

    Returns
    -------
    pandas dataframe
       df_agg, the inputted df_agg with bisg predictions appended in columns BISG_BAYES_COLS
    """
    # calculate P(V | R)
    v_given_r = get_v_given_r(df_cps_reg_voters, df_cen_counties)
    # bayes rule for race given geo for voters
    df_agg = append_r_given_gv(df_agg, df_cen_counties, v_given_r)
    # bayes rule for race given surname for voters
    df_agg = append_r_given_sv(df_agg, df_cen_surs, v_given_r)
    # three factors in bisg formula
    prob_of_r = df_cps_reg_voters[RACES].values
    r_given_geo = df_agg[CPS_BAYES_R_GIVEN_GEO_COLS].values
    r_given_surname = df_agg[CPS_BAYES_R_GIVEN_SUR_COLS].values
    # prediction
    bisg_preds = bisg_np(prob_of_r, r_given_surname, r_given_geo)
    df_agg[BISG_BAYES_COLS] = bisg_preds

    return df_agg


def cps_to_vf_matrix(u, v, verbose=False):
    """
    Find the 6 x 6 matrix A that minimizes |A - I|_F, (the Frobenius norm of the
    difference between A and the identity matrix) subject to the constraints
    A*u = v where u and v are inputted into the function and A is stochastic, that
    is A has non-negative entries and each column of A sums to 1. use cvxpy to
    solve this optimization problem.

    Parameters
    ----------
    u : numpy array of length 6
        the outputed matrix maps u to v
    v : numpy array of length 6
        the outputed matrix maps u to v
    verbose : bool, optional

    Returns
    -------
    6 x 6 numpy array
        solution of matrix-valued optimization
    float
        value of the objective function at minimum
    """
    n = 6
    a = cp.Variable((n, n), nonneg=True)
    objective = cp.Minimize(cp.sum_squares(a - np.eye(n)))
    constraints = [a @ u == v, a.T @ np.ones(n) == np.ones(n)]
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    _ = prob.solve(verbose=verbose, max_iter=100000)
    f_at_min = np.sum((np.dot(a.value, u) - v) ** 2)
    cps_to_vf_map = a.value

    return cps_to_vf_map, f_at_min


def subpopulation_preds(df_agg, cols, region="county", calib_map=None):
    """
    given a pandas dataframe, df_agg, make predictions of race/ethnicity for
    each (surname, geolocation) pair which corresponds to one row in df_agg.

    Parameters
    ----------
    df_agg : pandas dataframe
        with columns "name", "vf_tot", VF_RACES, region (input), cols
    cols : list
        names of 6 race/ethnicity prediction columns in df_agg
    region : string, optional
        name of geographical region in df_agg
    calib_map : 6 x 6 numpy array, optional
        predictions can be transformed via this matrix

    Returns
    -------
    n x m numpy array
        predictions
    n x m numpy array
        true populations
    n x m numpy array
        true probabilities (normalized populations)
    """

    # construct columns with count predictions (distribution multiplied by total
    # number of people) where distributions are in columns 'cols'
    df_subpops = df_agg.loc[:, cols].multiply(df_agg.loc[:, "vf_tot"], axis=0)

    # add helpful columns
    df_subpops[region] = df_agg[region]
    df_subpops[VF_RACES] = df_agg[VF_RACES]
    df_subpops["vf_tot"] = df_agg["vf_tot"]
    df_subpops[cols] = df_subpops[cols].astype(float)

    # remove NaN rows, which appear when surnames are missing from the census
    # list
    mask = ~df_subpops.isna().any(axis=1)
    df_subpops = df_subpops.loc[mask, :]

    # get predictions for each region
    preds = df_subpops.groupby(region).sum()[cols].values

    # convert to voter file population
    if calib_map is not None:
        print("*applying calib map to predictions")
        preds = np.dot(calib_map, preds.T).T

    # get trues
    true_pops = df_subpops.groupby(region).sum()[VF_RACES].values
    true_probs = np.divide(true_pops, np.sum(true_pops, axis=1).reshape((-1, 1)))

    return preds, true_pops, true_probs


def ipf2d(init, row_margin, col_margin, tol=1e-6, iter_max=1000, normalize=True):
    """
    iterative proportional fitting (ipf) or raking of a matrix to
    row and column margins

    Parameters
    ----------
    init : n x m numpy array
        the starting point for ipf
    row_margin : numpy array of length m
        margin to fit to
    col_margin : numpy array of length n
        margin to fit to
    tol : float
        relative tolerance at which to declare convergence
    iter_max : integer
        maximum number of iterations, at least 1
    normalize : bool, optional
        normalize rows

    Returns
    -------
    n x m numpy array
        output of ipf
    float
        l2 norm of error in column margin
    float
        l2 norm of error in row margin
    """
    # set values for the row and column sizes
    [n, m] = np.shape(init)

    # some checks, though these are by no means exhaustive
    assert np.allclose(np.sum(row_margin), np.sum(col_margin))
    assert np.size(col_margin) == n
    assert np.min(col_margin) >= 0
    assert np.size(row_margin) == m
    assert np.min(row_margin) >= 0
    # transform inputs for ease of numpy operations
    col_margin = col_margin.reshape((n, 1))
    row_margin = row_margin.reshape((m, 1))

    # start ipf
    preds = init
    for i in range(iter_max):
        # scale the columns
        col_margin_i = np.sum(preds, axis=1).reshape((n, 1))
        preds = preds * col_margin / col_margin_i
        # if there's a zero entry in a column margin, set column to zero
        if np.min(col_margin) == 0:
            ind = np.where(col_margin == 0)[0][0]
            init[ind, :] = 0
        # scale the rows
        row_margin_i = np.sum(preds, axis=0).reshape((m, 1))
        preds = (preds.T * row_margin / row_margin_i).T
        # if there's a zero entry in a row margin, set row to zero
        if np.min(row_margin_i) == 0:
            ind = np.where(row_margin == 0)[0][0]
            preds[:, ind] = 0

        # check errors
        col_margin_i = np.sum(preds, axis=1).reshape((n, 1))
        row_margin_i = np.sum(preds, axis=0).reshape((m, 1))
        err_row = np.sqrt(np.sum((row_margin_i - row_margin) ** 2))
        err_col = np.sqrt(np.sum((col_margin_i - col_margin) ** 2))
        if (err_row < tol) and (err_col < tol):
            break

    # normalize rows
    if normalize:
        preds = preds / np.sum(preds, axis=1).reshape((n, 1))

    return preds, err_col, err_row


def two_way_tables(df_vf):
    """
    take a state voter file in the form of a dataframe
    with columns 'name', 'race', 'county' and construct dataframes with
    the three two-way margins (name x race), (name x county), (county x race)

    Parameters
    ----------
    df_vf : pandas dataframe
        a dataframe with columns 'name', 'race', 'county'

    Returns
    -------
    pandas dataframe
        name by race
    pandas dataframe
        name by county
    pandas dataframe
        county by race
    """

    # make sure dataframe has correct columns
    assert "name" in df_vf.columns
    assert "race" in df_vf.columns
    assert "county" in df_vf.columns

    # construct name by race table
    df_tmp = df_vf.groupby(["name", "race"]).size()
    df_tmp = df_tmp.to_frame("count").reset_index()
    df_tmp = pd.pivot_table(data=df_tmp, columns="race", index="name", values="count")
    df_tmp = df_tmp.fillna(0).reset_index()
    df_vf_name_race = df_tmp.copy()

    # name by county table
    df_tmp = df_vf.groupby(["name", "county"]).size()
    df_tmp = df_tmp.to_frame("count").reset_index()
    df_tmp = pd.pivot_table(data=df_tmp, columns="county", index="name", values="count")
    df_tmp = df_tmp.fillna(0).reset_index()
    df_vf_name_county = df_tmp.copy()

    # county by race
    df_tmp = df_vf.groupby(["county", "race"]).size()
    df_tmp = df_tmp.to_frame("count").reset_index()
    df_tmp = pd.pivot_table(data=df_tmp, columns="race", index="county", values="count")
    df_tmp = df_tmp.fillna(0).reset_index()
    df_vf_county_race = df_tmp[["county"] + RACES]

    return df_vf_name_race, df_vf_name_county, df_vf_county_race


def basic_cols(df_agg, df_vf_name_race, df_vf_county_race):
    """
    add several important predictions to df_agg,
    - VF_BAYES_OPT_COLS: the correct distribution for a given (name, county)
    pair
    - VF_R_GIVEN_GEO_COLS: statewide race given county
    - VF_R_GIVEN_SUR_COLS: statewide race given surname
    - VF_BISG_COLS: bisg predictions using the statewide voter file data for
    p(r | s) and p(r) and using the county totals of the voterfile for
    p(r | g)

    Parameters
    ----------
    df_agg : pandas dataframe
        the notorious df_agg
    df_vf_name_race : pandas dataframe
        voter file race by name
    df_vf_county_race : pandas dataframe
        voter file race by county

    Returns
    -------
    pandas dataframe
        df_agg with new fields
    """

    # add "bayes optimal" predictions
    df_agg[VF_BAYES_OPT_COLS] = df_agg[VF_RACES].div(
        df_agg[VF_RACES].sum(axis=1), axis=0
    )

    # race given county columns
    for race in RACES:
        dict1 = df_vf_county_race.set_index("county").to_dict()[race]
        df_agg[f"vf_r_given_geo_{race}"] = df_agg["county"].map(
            lambda x: dict1.get(x, x)
        )
    # normalize race given county
    df_agg[VF_R_GIVEN_GEO_COLS] = df_agg[VF_R_GIVEN_GEO_COLS].div(
        df_agg[VF_R_GIVEN_GEO_COLS].sum(axis=1), axis=0
    )

    # race given surname columns
    for race in RACES:
        dict1 = df_vf_name_race.set_index("name").to_dict()[race]
        df_agg[f"vf_r_given_sur_{race}"] = df_agg["name"].map(lambda x: dict1.get(x, x))
    # normalize such that each (name, county) probabilies sum to 1
    df_agg[VF_R_GIVEN_SUR_COLS] = df_agg[VF_R_GIVEN_SUR_COLS].div(
        df_agg[VF_R_GIVEN_SUR_COLS].sum(axis=1), axis=0
    )

    # bisg for county
    prob_of_r = (
        df_vf_county_race.sum(axis=0)[RACES]
        .div(df_vf_county_race[RACES].sum().sum())
        .values
    )
    r_given_geo = df_agg[VF_R_GIVEN_GEO_COLS].values
    r_given_surname = df_agg[VF_R_GIVEN_SUR_COLS].values

    # bisg prediction
    bisg_preds = r_given_geo * r_given_surname / prob_of_r
    bisg_preds = bisg_preds / np.sum(bisg_preds, axis=1)[:, np.newaxis]
    df_agg[VF_BISG_COLS] = bisg_preds

    return df_agg


def two_way_raking(df_agg, tol=1e-6, verbose=False):
    """
    in each county, rake to correct race and surname margins. starting
    point of raking is surname-only estimates

    Parameters
    ----------
    df_agg : pandas dataframe
        the notorious df_agg
    tol : float, optional
        tolerance of raking
    verbose : bool, optional

    Returns
    -------
    pandas dataframe
        df_agg with voter file 2-way raking columns
    """
    for county in df_agg["county"].unique():
        # only current county
        mask = df_agg["county"] == county
        # surname only estimates for initialization
        init = df_agg[mask][VF_R_GIVEN_SUR_COLS].values
        row_margin = np.sum(df_agg[mask][VF_RACES].values, axis=0)
        col_margin = df_agg[mask]["vf_tot"].values
        # ipf
        preds_new, err_row, err_col = ipf2d(
            init, row_margin, col_margin, tol=tol, iter_max=1000, normalize=True
        )
        if np.sum(np.isnan(preds_new)) > 0:
            print(f"NaN in two-way raking in {county}")

        if verbose:
            print(f"county: {county}")
            print(f"errs: {err_row, err_col}")
        # add to big dataframe
        df_agg.loc[mask, VF_RAKE2_COLS] = preds_new
    return df_agg


def three_way_raking(df_agg, tol=1e-3):
    """
    construct estimates for three-way raking of voter file. that is,
    predictions in the end will have all three two-way margins correct,
    (name x county), (name x race), (county x race)


    Parameters
    ----------
    df_agg : pandas dataframe
        the notorious df_agg
    tol : float, optional
        tolerance of raking

    Returns
    -------
    pandas dataframe
        df_agg with voter file 3-way raking columns
    """

    # construct empty 3d array of names x counties x races
    names = df_agg["name"].unique()
    nnames = np.shape(names)[0]
    counties = df_agg["county"].unique()
    ncounties = np.shape(counties)[0]

    arr3d = np.empty((nnames, ncounties, len(RACES)))
    # fill empty 3d array with data, going race by race
    for i, race in enumerate(VF_RACES):
        df_tmp = (
            df_agg.pivot(index="name", columns="county", values=race)
            .fillna(0)
            .reindex(names)
        )
        arr3d[:, :, i] = df_tmp[counties].values

    # construct 2-way margins
    county_race = np.sum(arr3d, axis=0)
    name_race = np.sum(arr3d, axis=1)
    name_county = np.sum(arr3d, axis=2)

    # ipf
    init = np.ones_like(arr3d)
    df_tmp = ipf3d(
        init,
        name_county,
        name_race,
        county_race,
        RACES,
        counties,
        names,
        tol=tol,
        max_iters=100,
    )

    # and merge them together into 6 columns
    df_agg = df_agg.merge(df_tmp, how="left", on=["name", "county"])

    # normalize counts so that we now have probability predictions
    df_agg[VF_RAKE3_COLS] = df_agg[VF_RAKE3_COUNTS].div(
        df_agg[VF_RAKE3_COUNTS].sum(axis=1), axis=0
    )

    return df_agg


def ipf3d(
    init,
    name_county,
    name_race,
    county_race,
    races,
    counties,
    names,
    tol=1e-6,
    max_iters=100,
    verbose=False,
):
    """
    perform so-called 3-way iterative proportional fitting (ipf) or raking.
    start with a 3d array (init) of dimensions
    n x k x p, and rake to three 2-way margins, of dimensions, n x k, n x p, k x p.
    Parameters
    ----------
    init : n x k x p numpy array
        initialization of raking procedure, or the fixed terms in the log-linear model
    name_county : n x k numpy array
        raking margin
    name_race   : n x p numpy array
        raking margin
    county_race : k x p numpy array
        raking margin
    races : list of length p
        strings of races
    counties : numpy array of length k
        strings of counties
    names : numpy array of length n
        strings of names
    tol : float, optional
        tolerance of iterative procedure
    max_iters : integer, optional
        maximum number of
    verbose : bool, optional
    Returns
    -------
    pandas dataframe
        raking solution
    """
    assert np.shape(name_county)[0] == np.shape(name_race)[0]
    assert np.shape(name_county)[1] == np.shape(county_race)[0]
    assert np.shape(county_race)[1] == np.shape(name_race)[1]
    assert np.allclose(np.sum(name_race), np.sum(name_county), rtol=1e-6)
    assert np.allclose(np.sum(name_race), np.sum(county_race), rtol=1e-6)
    arr_iter = init.copy()

    # and iterate
    for i in range(max_iters):
        # name_county
        name_countyi = np.sum(arr_iter, axis=2)
        quo = name_countyi / name_county
        quo2 = quo[:, :, np.newaxis]
        arr_iter = arr_iter / quo2
        arr_iter[np.isnan(arr_iter)] = 0

        # name_race
        name_racei = np.sum(arr_iter, axis=1)
        quo = name_racei / name_race
        quo2 = quo[:, np.newaxis, :]
        arr_iter = arr_iter / quo2
        arr_iter[np.isnan(arr_iter)] = 0

        # county_race
        county_racei = np.sum(arr_iter, axis=0)
        quo = county_racei / county_race
        quo2 = quo[np.newaxis, :, :]
        arr_iter = arr_iter / quo2
        arr_iter[np.isnan(arr_iter)] = 0

        # relative errors, but only where counts are non-zero
        mask1 = county_race > 0
        err1 = np.max(
            np.abs(county_race[mask1] - county_racei[mask1]) / (county_race[mask1])
        )
        mask2 = name_race > 0
        err2 = np.max(np.abs(name_race[mask2] - name_racei[mask2]) / (name_race[mask2]))
        mask3 = name_county > 0
        err3 = np.max(
            np.abs(name_county[mask3] - name_countyi[mask3]) / (name_county[mask3])
        )

        if verbose:
            print(f"iteration: {i}")
            print(f"total number of people predicted: {np.sum(arr_iter)}")
            print(f"max error in county x race: {err1}")
            print(f"max error in name x race: {err2}")
            print(f"max error in name x county: {err3}")

        if np.max([err1, err2, err3]) < tol:
            break

    # create three-way raking predictions, one race (column) at a time
    # for each race, take a 2d numpy array and convert it into a dataframe with
    # columns name, county, and count. after combining counts for all races,
    # counts will be normalized to become predictions
    dfs = []
    for i, _ in enumerate(races):
        df_tmp = pd.DataFrame(arr_iter[:, :, i], columns=counties, index=names)
        df_tmp = (
            df_tmp.stack()
            .reset_index(name=VF_RAKE3_COUNTS[i])
            .rename(columns={"level_0": "name", "level_1": "county"})
        )
        dfs.append(df_tmp)

    df_tmp = dfs[0]
    for i in range(1, 6):
        df_tmp = pd.merge(df_tmp, dfs[i], how="left", on=["name", "county"])

    return df_tmp


if __name__ == "__main__":
    main()
