"""
this module loads race by surname and race by geolocation information
from the us census.
the geolocation by race information comes from redistricting files from 2010 or
2020. the 2020 file are available for download here  -- 
https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/ --
and the 2010 census data is similar in format and can be downloaded from --
https://www2.census.gov/census_2010/01-Redistricting_File--PL_94-171/ --
the directory downloaded from this link for a particular state should be
saved as the following:
$ ./census_data/{state}.pl

the data that we need in for a given state, here florida in 2020,
is contained in three files in that directory:
  ./census_data/fl2020.pl/flgeo2020.pl
  ./census_data/fl2020.pl/fl000012020.pl
  ./census_data/fl2020.pl/fl000022020.pl
each of these files can be viewed as containing columns of tabular 
data where each file is aligned in its rows. i.e. the fifth row of each
of these files corresponds to the same line item. the first file 
contains geographical information about the line item, and the other 
two contain demographic information. the second file contains census
information for the full population and the third contains census 
information for the 18+ population.

Functions
---------
county_census
    race by county information for a particular state
usa_census
    race information for the full country
surnames
    race information for each surname in the census list
"""
import numpy as np
import pandas as pd
import os
import fields


# dictionary in which keys are the field names used by the census and
# values are interpretable column names used throughout this package
CEN_TO_NAME_DICT = {
    "P0010001": "total",
    "P0010002": "one_race",
    "P0010003": "white",
    "P0010004": "black",
    "P0010005": "aian",
    "P0010006": "asian",
    "P0010007": "pac_isl",
    "P0010008": "other",
    "P0020001": "total",
    "P0020002": "hispanic",
    "P0020003": "not_hispanic",
    "P0020004": "nh_one_race",
    "P0020005": "nh_white",
    "P0020006": "nh_black",
    "P0020007": "nh_aian",
    "P0020008": "nh_asian",
    "P0020009": "nh_pac_isl",
    "P0020010": "nh_other",
    "P0030001": "total18",  # P0040001 is identical to P0030001
    "P0030002": "one_race18",
    "P0030003": "white18",
    "P0030004": "black18",
    "P0030005": "aian18",
    "P0030006": "asian18",
    "P0030007": "pac_isl18",
    "P0030008": "other18",  # another race, alone
    "P0030009": "two_more_race18",
    "P0040001": "total18",  # 'P0040001' P0040001 is identical to P0030001
    "P0040002": "hispanic18",
    "P0040003": "not_hispanic18",
    "P0040004": "nh_one_race18",
    "P0040005": "nh_white18",
    "P0040006": "nh_black18",
    "P0040007": "nh_aian18",
    "P0040008": "nh_asian18",
    "P0040009": "nh_pac_isl18",
    "P0040010": "nh_other18",  # non-hispanic some other race, alone
    "P0040011": "nh_two_more_race18",  # non-hispanic two or more races
}

FL_CODE_TO_COUNTY_DICT = {
    "Alachua": "ALA",
    "Baker": "BAK",
    "Bay": "BAY",
    "Bradford": "BRA",
    "Brevard": "BRE",
    "Broward": "BRO",
    "Calhoun": "CAL",
    "Charlotte": "CHA",
    "Citrus": "CIT",
    "Clay": "CLA",
    "Collier": "CLL",
    "Columbia": "CLM",
    "Miami-Dade": "DAD",
    "Desoto": "DES",
    "Dixie": "DIX",
    "Duval": "DUV",
    "Escambia": "ESC",
    "Flagler": "FLA",
    "Franklin": "FRA",
    "Gadsden": "GAD",
    "Gilchrist": "GIL",
    "Glades": "GLA",
    "Gulf": "GUL",
    "Hamilton": "HAM",
    "Hardee": "HAR",
    "Hendry": "HEN",
    "Hernando": "HER",
    "Highlands": "HIG",
    "Hillsborough": "HIL",
    "Holmes": "HOL",
    "Indian River": "IND",
    "Jackson": "JAC",
    "Jefferson": "JEF",
    "Lafayette": "LAF",
    "Lake": "LAK",
    "Lee": "LEE",
    "Leon": "LEO",
    "Levy": "LEV",
    "Liberty": "LIB",
    "Madison": "MAD",
    "Manatee": "MAN",
    "Marion": "MRN",
    "Martin": "MRT",
    "Monroe": "MON",
    "Nassau": "NAS",
    "Okaloosa": "OKA",
    "Okeechobee": "OKE",
    "Orange": "ORA",
    "Osceola": "OSC",
    "PalmBeach": "PAL",
    "Pasco": "PAS",
    "Pinellas": "PIN",
    "Polk": "POL",
    "Putnam": "PUT",
    "SantaRosa": "SAN",
    "Sarasota": "SAR",
    "Seminole": "SEM",
    "St.Johns": "STJ",
    "St.Lucie": "STL",
    "Sumter": "SUM",
    "Suwannee": "SUW",
    "Taylor": "TAY",
    "Union": "UNI",
    "Volusia": "VOL",
    "Wakulla": "WAK",
    "Walton": "WAL",
    "Washington": "WAS",
}

# the following dictionary has census naming convention for the
# keys and voter file convention for the values. all other florida
# counties have the same name in the census and the voter file
FL_CENSUS_TO_VF_COUNTY = {
    "DeSoto": "Desoto",
    "Palm Beach": "PalmBeach",
    "Santa Rosa": "SantaRosa",
    "St. Johns": "St.Johns",
    "St. Lucie": "St.Lucie",
}

NY_CENSUS_TO_VF_COUNTY = {"st. lawrence": "st.lawrence"}

# new dictionary with flipped keys and values
NAME_TO_CEN_DICT = dict([(value, key) for key, value in CEN_TO_NAME_DICT.items()])

DIR = "census_data/2020_PLSummaryFile_FieldNames/"

# rename columns of census surname list
DICT_CEN_TO_RACES = {
    "pctwhite": "nh_white",
    "pctblack": "nh_black",
    "pctapi": "nh_api",
    "pctaian": "nh_aian",
    "pct2prace": "other",
    "pcthispanic": "hispanic",
}

# columns used by census file
CEN_RACES_COLS = [
    "pctwhite",
    "pctblack",
    "pctapi",
    "pctaian",
    "pct2prace",
    "pcthispanic",
]


def main():
    # df = county_census('nc', year=2010)
    # df = county_census('nc', year=2020)
    df = county_census("fl", year=2020)
    print(df.head())

    df_usa = usa_census(year=2010)
    print(df_usa.head())


def county_census(state, year, load=True):
    """
    this function mainly serves the purpose of wrapping the other functions
    of this module that are doing the pull, formatting, and cleaning of
    census data

    Parameters
    ----------
    state : string
        state
    year : int
        year (one of 2010 and 2020)
    load : bool, optional
        load the file if it's already been created

    Returns
    -------
    pandas dataframe
        the race distribution of each county in state
    """
    state = state.lower()

    # if this dataframe has already been created, load it
    filename = f"generated_data/county_census_{state}{year}.feather"
    if os.path.exists(filename) and load:
        print(f"*loading dataframe from {filename}")
        return pd.read_feather(filename)

    # otherwise, combine census files
    if year == 2020:
        df_geo = census2020_geo_file(state)
        df_file1 = census2020_file1(state)
        df_file2 = census2020_file2(state)
        df = pd.concat([df_geo, df_file1, df_file2], axis=1)
    else:
        # year is 2010
        df_geo = census2010_geo_file(state)
        df_file1 = census2010_file1(state)
        df_file2 = census2010_file2(state)
        df = pd.concat([df_geo, df_file1, df_file2], axis=1)

    # and do cleaning
    df = postprocess(df)

    # and state-specific cleaning
    if state == "fl":
        df = postprocess_fl(df)
    else:
        df = postprocess_state(state, df)

    # write dataframe to feather, but first reset index as required by feather
    df = df.reset_index(drop=True)
    df.to_feather(filename)

    return df


def census2020_geo_file(state):
    """
    geographical data for each row in the 2020 redistricting file for a particular state

    Parameters
    ----------
    state : string
        state

    Returns
    -------
    pandas dataframe
        geographical information for 2020 files
    """
    # pull in the columns names of the geo file
    filename = DIR + "2020 P.L. Geoheader Definitions-Table 1.csv"
    df_tmp = pd.read_csv(filename)
    # remove first two rows which are empty
    df_tmp = df_tmp.iloc[
        2:,
    ]

    # pull in geo file
    geo_cols = df_tmp["DATA DICTIONARY REFERENCE"].values
    df_geo = pd.read_csv(
        f"census_data/{state}2020.pl/{state}geo2020.pl",
        sep="|",
        header=None,
        names=geo_cols,
        dtype="object",
    )

    # columns of interest
    geo_cols = ["SUMLEV", "NAME", "STATE", "COUNTY"]
    df_geo = df_geo[geo_cols]

    return df_geo


def census2020_file1(state):
    """
    read in census 2020 file 1 to a dataframe which contains information
    on the full population

    Parameters
    ----------
    state : string
        state

    Returns
    -------
    pandas dataframe
        file 1 information
    """
    # pull in the columns names of the fist (non-geo) census file
    filename = DIR + "2020 P.L. Segment 1 Definitions-Table 1.csv"
    df_tmp = pd.read_csv(filename)

    # fields of first file
    file1_cols = df_tmp["DATA DICTIONARY REFERENCE NAME"].values
    # columns to use
    usecols = [
        "P0010002",
        "P0020001",
        "P0020002",
        "P0020003",
        "P0020004",
        "P0020005",
        "P0020006",
        "P0020007",
        "P0020008",
        "P0020009",
        "P0020010",
    ]
    df = pd.read_csv(
        f"census_data/{state}2020.pl/{state}000012020.pl",
        sep="|",
        header=None,
        names=file1_cols,
        usecols=usecols,
        dtype=float,
    )

    # change census codes to interpretable names
    df = df.rename(columns=CEN_TO_NAME_DICT)

    # combine asian and pacific-islander
    df.loc[:, "nh_api"] = df["nh_asian"] + df["nh_pac_isl"]

    return df


def census2020_file2(state):
    """
    read in census 2020 file 2 to a dataframe which contains information
    on the 18+ population

    Parameters
    ----------
    state : string
        state

    Returns
    -------
    pandas dataframe
        file 2 information
    """
    # pull in the columns names of the secon (non-geo) census file
    filename = DIR + "2020 P.L. Segment 2 Definitions-Table 1.csv"
    df_tmp = pd.read_csv(filename)
    # all columns names
    file2_cols = df_tmp["DATA DICTIONARY REFERENCE NAME"].values

    # columns of interest
    usecols = [
        "P0030001",
        "P0030002",
        "P0030003",
        "P0030004",
        "P0030005",
        "P0030006",
        "P0030007",
        "P0030008",
        "P0030009",
        "P0040002",
        "P0040003",
        "P0040004",
        "P0040005",
        "P0040006",
        "P0040007",
        "P0040008",
        "P0040009",
        "P0040010",
        "P0040011",
    ]

    df = pd.read_csv(
        f"census_data/{state}2020.pl/{state}000022020.pl",
        sep="|",
        header=None,
        names=file2_cols,
        usecols=usecols,
        dtype=float,
    )

    # change census codes to interpretable names
    df = df.rename(columns=CEN_TO_NAME_DICT)
    # combine asian and pacific-islander
    df.loc[:, "nh_api18"] = df["nh_asian18"] + df["nh_pac_isl18"]

    return df


def postprocess(df):
    """
    take the dataframe obtained from the census redistricting files
    and do some cleaning.

    Parameters
    ----------
    df : pandas dataframe
        unformatted census information

    Returns
    -------
    pandas dataframe
        formatted census information
    """

    # only counties
    df = df.loc[df["SUMLEV"] == "050"]
    # add the 'other' race
    non_other_races = ["nh_white", "nh_black", "nh_api", "nh_aian", "hispanic"]
    df.loc[:, "other"] = df.loc[:, "total"] - df.loc[:, non_other_races].sum(axis=1)

    # add the 'other' race for 18-plus
    non_other_races = [
        "nh_white18",
        "nh_black18",
        "nh_api18",
        "nh_aian18",
        "hispanic18",
    ]
    df.loc[:, "other18"] = df.loc[:, "total18"] - df.loc[:, non_other_races].sum(axis=1)

    # add races proportions, i.e. normalize
    df[fields.PROB_COLS] = df[fields.RACES].div(df[fields.RACES].sum(axis=1), axis=0)
    df[fields.PROB18_RACES] = df[fields.RACES18].div(df[fields.RACES18].sum(axis=1), axis=0)

    return df


def postprocess_fl(df):
    """
    make county names consistent with vf, there are only a few
    minor differences

    Parameters
    ----------
    df : pandas dataframe
        uncleaned florida census data

    Returns
    -------
    pandas dataframe
        cleaned florida census data
    """

    # remove the word "county"
    df.loc[:, "county_long"] = df.loc[:, "NAME"].map(lambda x: x.replace(" County", ""))
    df.loc[:, "county_long"] = df.loc[:, "county_long"].map(
        lambda x: FL_CENSUS_TO_VF_COUNTY.get(x, x)
    )

    # county codes
    df.loc[:, "county"] = df.loc[:, "county_long"].map(
        lambda x: FL_CODE_TO_COUNTY_DICT.get(x, x)
    )

    return df


def postprocess_state(state, df):
    """
    make county names consistent with vf, there should be only a few
    minor differences

    Parameters
    ----------
    df : pandas dataframe
        uncleaned census data

    Returns
    -------
    pandas dataframe
        cleaned census data
    """
    # remove the word "county"
    df.loc[:, "county"] = df.loc[:, "NAME"].map(lambda x: x.replace(" County", ""))

    df.loc[:, "county"] = df.loc[:, "county"].str.lower()
    if state.lower() == "ny":
        df.loc[:, "county"] = df.loc[:, "county"].map(
            lambda x: NY_CENSUS_TO_VF_COUNTY.get(x, x)
        )

    return df


def census2020_usa_file1():
    """
    pull in first row of the redistricting file for the full country which
    was downloaded from the "National" file here:
    https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/

    Returns
    -------
    pandas dataframe
        usa census information for full population
    """
    # pull in the columns names of the first (non-geo) census file
    filename = DIR + "2020 P.L. Segment 1 Definitions-Table 1.csv"
    df_tmp = pd.read_csv(filename)
    # fields of first file
    file1_cols = df_tmp["DATA DICTIONARY REFERENCE NAME"].values

    # columns to use
    usecols = [
        "P0010002",
        "P0020001",
        "P0020002",
        "P0020003",
        "P0020004",
        "P0020005",
        "P0020006",
        "P0020007",
        "P0020008",
        "P0020009",
        "P0020010",
    ]
    filename = "census_data/us2020.npl/us000012020.npl"
    df = pd.read_csv(
        filename,
        sep="|",
        header=None,
        names=file1_cols,
        usecols=usecols,
        dtype=float,
        nrows=1,
    )

    # change census codes to interpretable names
    df = df.rename(columns=CEN_TO_NAME_DICT)

    # combine asian and pacific-islander
    df.loc[:, "nh_api"] = df["nh_asian"] + df["nh_pac_isl"]

    return df


def census2020_usa_file2():
    """
    pull in first row of the redistricting file for the 18+ usa population

    Returns
    -------
    pandas dataframe
        usa census information for 18+ population
    """
    # pull in the columns names of the fist (non-geo) census file
    filename = DIR + "2020 P.L. Segment 2 Definitions-Table 1.csv"
    df_tmp = pd.read_csv(filename)
    # all columns names
    file2_cols = df_tmp["DATA DICTIONARY REFERENCE NAME"].values

    # columns of interest
    usecols = [
        "P0030001",
        "P0030002",
        "P0030003",
        "P0030004",
        "P0030005",
        "P0030006",
        "P0030007",
        "P0030008",
        "P0030009",
        "P0040002",
        "P0040003",
        "P0040004",
        "P0040005",
        "P0040006",
        "P0040007",
        "P0040008",
        "P0040009",
        "P0040010",
        "P0040011",
    ]

    filename = "census_data/us2020.npl/us000022020.npl"
    df = pd.read_csv(
        filename,
        sep="|",
        header=None,
        names=file2_cols,
        usecols=usecols,
        dtype=float,
        nrows=1,
    )

    # change census codes to interpretable names
    df = df.rename(columns=CEN_TO_NAME_DICT)
    # combine asian and pacific-islander
    df.loc[:, "nh_api18"] = df["nh_asian18"] + df["nh_pac_isl18"]

    return df


def postprocess_usa(df):
    """
    take the dataframe obtained from the census redistricting files
    and do some cleaning.

    Parameters
    ----------
    df : pandas dataframe
        uncleaned usa census information

    Returns
    -------
    pandas dataframe
        cleaned usa census information
    """

    # add the 'other' race
    non_other_races = ["nh_white", "nh_black", "nh_api", "nh_aian", "hispanic"]
    df.loc[:, "other"] = df.loc[:, "total"] - df.loc[:, non_other_races].sum(axis=1)

    # add the 'other' race for 18-plus
    non_other_races = [
        "nh_white18",
        "nh_black18",
        "nh_api18",
        "nh_aian18",
        "hispanic18",
    ]

    df.loc[:, "other18"] = df.loc[:, "total18"] - df.loc[:, non_other_races].sum(axis=1)

    # add races proportions, i.e. normalize
    df[fields.PROB_COLS] = df[fields.RACES].div(df[fields.RACES].sum(axis=1), axis=0)
    df[fields.PROB18_RACES] = df[fields.RACES18].div(df[fields.RACES18].sum(axis=1), axis=0)

    return df


def usa_census(year):
    """
    get usa race distribution for the full and 18+ populations.
    this function mainly wraps the other usa-related functions of this module
    that are doing the pulling, formatting, and cleaning of census data

    Parameters
    ----------
    year : int
        year

    Returns
    -------
    pandas dataframe
        usa census race information
    """
    if year == 2020:
        # combine census files
        df_file1 = census2020_usa_file1()
        df_file2 = census2020_usa_file2()
        df_usa = pd.concat([df_file1, df_file2], axis=1)
    else:
        # year is 2010
        df_file1 = census2010_usa_file1()
        df_file2 = census2010_usa_file2()
        df_usa = pd.concat([df_file1, df_file2], axis=1)

    # and do cleaning
    df_usa = postprocess_usa(df_usa)

    return df_usa


def census2010_usa_file1():
    """
    pull in first row of the redistricting file for the full country in 2010

    Returns
    -------
    pandas dataframe
        usa census information for full population
    """
    # pull in race data
    # first get table 2 info
    cols = file1_info_2010().field_short.values

    # pull in table 2
    usecols = [
        "P0010002",
        "P0020001",
        "P0020002",
        "P0020003",
        "P0020004",
        "P0020005",
        "P0020006",
        "P0020007",
        "P0020008",
        "P0020009",
        "P0020010",
    ]
    filename = f"census_data/us2010.npl/us000012010.npl"
    df_file1 = pd.read_csv(
        filename, header=None, names=cols, usecols=usecols, dtype=float, nrows=1
    )
    df_file1 = df_file1.rename(columns=CEN_TO_NAME_DICT)
    df_file1["nh_api"] = df_file1["nh_asian"] + df_file1["nh_pac_isl"]

    return df_file1


def census2010_usa_file2():
    """
    pull in first row of the redistricting file for the 18+ population in 2010

    Returns
    -------
    pandas dataframe
        usa census information for the 18+ population
    """
    cols = file2_info_2010().field_short.values
    # columns of interest
    usecols = [
        "P0030001",
        "P0030002",
        "P0030003",
        "P0030004",
        "P0030005",
        "P0030006",
        "P0030007",
        "P0030008",
        "P0030009",
        "P0040002",
        "P0040003",
        "P0040004",
        "P0040005",
        "P0040006",
        "P0040007",
        "P0040008",
        "P0040009",
        "P0040010",
        "P0040011",
    ]
    filename = f"census_data/us2010.npl/us000022010.npl"
    df_file2 = pd.read_csv(
        filename,
        sep=",",
        header=None,
        names=cols,
        usecols=usecols,
        dtype=float,
        nrows=1,
    )

    # change census codes to interpretable names
    df_file2 = df_file2.rename(columns=CEN_TO_NAME_DICT)
    # combine asian and pacific-islander
    df_file2.loc[:, "nh_api18"] = df_file2["nh_asian18"] + df_file2["nh_pac_isl18"]

    return df_file2


def append_r_given_g_cols(df_agg, df_cen_counties):
    """
    append race given geolocation columns to df_agg where race given geolocation
    is calculated with census data

    Parameters
    ----------
    df_agg : pandas dataframe
        various race/ethnicity predictions
    df_cen_counties : pandas dataframe
        census race by county in a state

    Returns
    -------
    pandas dataframe
        df_agg with the appended columns
    """
    # check that the counties of df_agg are the same as those in df_cen_counties
    counties1 = df_agg["county"].unique()
    counties2 = df_cen_counties["county"].unique()
    counties1.sort()
    counties2.sort()

    try:
        assert np.array_equal(counties1, counties2)
    except AssertionError:
        print(f"discrepancy in census and voter file counties")
        # if census and voter file don't have the same number of files, print numbers
        if counties1.shape != counties2.shape:
            print(f"voter file: {counties1.shape[0]} counties")
            print(f"census: {counties2.shape[0]} counties")
            print(
                f"this might be a consequence of the voter file not having data from "
                "certain counties such as in the case of the North Carolina 2010 voter file"
            )
        # otherwise print the differences
        else:
            for i, county in enumerate(counties1):
                if counties2[i] != county:
                    print(f"census: {counties2[i]}, voter file: {county}")

    # add race population columns corresponding to 18+ population
    for i, race in enumerate(fields.RACES):
        dict1 = df_cen_counties.set_index("county").to_dict()[fields.PROB18_RACES[i]]
        df_agg[fields.CEN18_R_GIVEN_GEO_COLS[i]] = df_agg["county"].map(
            lambda x: dict1.get(x, x)
        )

    # add race population columns corresponding to full population
    for i, race in enumerate(fields.RACES):
        dict1 = df_cen_counties.set_index("county").to_dict()[fields.PROB_COLS[i]]
        df_agg[fields.CEN_R_GIVEN_GEO_COLS[i]] = df_agg["county"].map(
            lambda x: dict1.get(x, x)
        )

    return df_agg


def append_r_given_s_cols(df_agg, df_cen_surs):
    """
    append race given surname columns to df_agg

    Parameters
    ----------
    df_agg : pandas dataframe
        various race/ethnicity predictions
    df_cen_surs : pandas dataframe
        census race by surname in the usa

    Returns
    -------
    pandas dataframe
        df_agg with the appended columns
    """
    # add surname data
    for race in fields.RACES:
        col = f"cen_r_given_sur_{race}"
        dict1 = df_cen_surs.set_index("name").to_dict()[col]
        df_agg[col] = df_agg["name"].map(lambda x: dict1.get(x, x))

        # when a name isn't in census surnames, the entry in col will be
        # a string (the surname which wasn't found in the census data),
        # set to the race of "all other names"
        mask = ~df_agg["name"].isin(df_cen_surs["name"].unique())
        df_agg.loc[mask, col] = df_cen_surs.loc[
            df_cen_surs["name"] == "all other names", col
        ].values[0]

    # convert columns to doubles
    for col in fields.CEN_R_GIVEN_SUR_COLS:
        df_agg = df_agg.astype({col: "float64"})

    return df_agg


def surnames():
    """
    this function is used to process the .csv file provided by the census bureau
    with the race/ethnicity distribution of each surname that appears at least
    100 times in the US. This script was run on the 2010 surname list which
    can be found here -- https://www.census.gov/topics/population/genealogy/data/2010_surnames.html

    this function makes one minor modification to the numbers in census file.
    in order to protect privacy, when there is a sufficiently small non-zero
    number of people of a particular surname, the census hides that number.
    here, we fill that number according to what was done in the original BISG
    paper. we distribute the unknown population uniformly over the unknown
    non-zero race/ethnicity groups.

    Returns
    -------
    pandas dataframe
        race distribution of names in us census that appear >=100 times
    """

    # read in .csv from the census
    filename = "census_data/Names_2010Census.csv"
    df_cen_surs = pd.read_csv(filename)
    df_cen_surs["name"] = df_cen_surs["name"].str.lower()

    # the name "Null" is being fed in as NaN
    nans = np.sum(df_cen_surs["name"].isna())
    if nans > 1:
        print("NaN problem with names")
    else:
        df_cen_surs.loc[df_cen_surs["name"].isna(), "name"] = "null"

    # find where (S) are, which is how the census denotes the
    # race/ethnicity cells that are too small to report
    mask_ss = df_cen_surs[CEN_RACES_COLS] == "(S)"
    # replace (S) with nan
    df_cen_surs[mask_ss] = np.nan
    # convert other entries in table to floats
    df_cen_surs.loc[:, CEN_RACES_COLS] = (
        df_cen_surs.loc[:, CEN_RACES_COLS].astype(float) / 100
    )

    # find number of known totals in each row
    df_cen_surs["known"] = (
        df_cen_surs[CEN_RACES_COLS]
        .multiply(df_cen_surs["count"].astype(float), axis=0)
        .sum(axis=1)
    )
    # number of missing people
    df_cen_surs["missing"] = df_cen_surs["count"] - df_cen_surs["known"]
    # percentage missing
    df_cen_surs["to_fill"] = df_cen_surs["missing"].div(df_cen_surs["count"])
    # how much to put in each missing entry (same across given row)
    df_cen_surs["fill_amount"] = df_cen_surs["to_fill"].div(mask_ss.sum(axis=1))
    # replace NaNs with the correct value
    for col in CEN_RACES_COLS:
        df_cen_surs[col] = df_cen_surs[col].fillna(df_cen_surs["fill_amount"])

    # change column names
    df_cen_surs = df_cen_surs.rename(columns=DICT_CEN_TO_RACES)

    # normalize rows, so that they are accurate to full precision, not 3 digits
    df_cen_surs[fields.RACES] = df_cen_surs[fields.RACES].div(df_cen_surs[fields.RACES].sum(axis=1), axis=0)

    # give race columns their usual names
    df_cen_surs = df_cen_surs.rename(
        columns=dict(zip(fields.RACES, [f"cen_r_given_sur_{race}" for race in fields.RACES]))
    )

    return df_cen_surs


def geo_info_2010():
    """
    get the field names and number of characters per field
    for the geography file of the 2010 census redistricting files

    Returns
    -------
    pandas dataframe
        contains information about names and character-width of fields
    """
    # pull in formatting information from text file for reading
    filename = "census_data/sas_scripts/sas_2010_geoheader.sas"
    cols = ["character_start", "field_short", "character_length", "field_long"]
    df = pd.DataFrame(columns=cols)

    # preprocessing for geo file
    irow = 0
    with open(filename, "r") as fp:
        for line in fp:
            if line[0] == "@":
                ll = line.split(sep=" ", maxsplit=3)
                # remove @ symbol
                ll[0] = ll[0][1:]
                # remove $ symbol and decimal
                ll[2] = ll[2][1:-1]
                # remove /* and */
                ll[3] = ll[3][2:-3]
                df.loc[irow] = ll
                irow += 1
    df.character_start = df.character_start.astype(int)
    df.character_length = df.character_length.astype(int)

    # convert to 0 indexing
    df.character_start = df.character_start - 1

    return df


def file1_info_2010():
    """
    the 2010 census redistricting files have field names that are saved in a text file.
    this function reads that text file for the first redistricting file
    and converts it to a pandas dataframe.

    Returns
    -------
    pandas dataframe
        information on fields of 2010 file 1
    """
    lls = []
    filename = "census_data/redistricting_cols_2010_file1.txt"
    with open(filename, "r") as fp:
        for line in fp:
            ll = line.split(sep=" ", maxsplit=2)
            # convert from string to int
            ll[1] = int(ll[1][1:])
            # remove comments around last element
            ll[2] = ll[2][2:-3]
            lls.append(ll)

    cols = ["field_short", "char_length", "field_long"]
    df_file_info = pd.DataFrame(lls, columns=cols)

    return df_file_info


def file2_info_2010():
    """
    the 2010 census redistricting files have field names that are saved in a text file.
    this function reads that text file for the second redistricting file
    and converts it to a pandas dataframe.

    Returns
    -------
    pandas dataframe
        information on fields of 2010 file 1
    """
    lls = []
    filename = "census_data/redistricting_cols_2010_file2.txt"
    with open(filename, "r") as fp:
        for line in fp:
            ll = line.split(sep=" ", maxsplit=2)
            # convert from string to int
            ll[1] = int(ll[1][1:])
            # remove comments around last element
            ll[2] = ll[2][2:-3]
            lls.append(ll)

    cols = ["field_short", "char_length", "field_long"]
    df_file_info = pd.DataFrame(lls, columns=cols)

    return df_file_info


def census2010_geo_file(state):
    """
    read the 2010 census redistricting geography file

    Parameters
    ----------
    state : string
        us state

    Returns
    -------
    pandas dataframe
        geography file
    """
    # pull in formatting information for geo file
    df = geo_info_2010()
    # read geo file
    filename = f"census_data/{state}2010.pl/{state}geo2010.pl"
    # character length
    cl = df["character_length"].values
    # read text file
    usecols = ["SUMLEV", "NAME", "STATE", "COUNTY"]
    df_geo = pd.read_fwf(
        filename, widths=cl, names=df["field_short"], usecols=usecols, dtype="object"
    )

    return df_geo


def census2010_file1(state):
    """
    read the 2010 census redistricting file 1

    Parameters
    ----------
    state : string
        us state

    Returns
    -------
    pandas dataframe
        file 1 of redistricting file
    """

    # first get table 2 info
    cols = file1_info_2010().field_short.values

    # pull in table 2
    usecols = [
        "P0010002",
        "P0020001",
        "P0020002",
        "P0020003",
        "P0020004",
        "P0020005",
        "P0020006",
        "P0020007",
        "P0020008",
        "P0020009",
        "P0020010",
    ]
    filename = f"census_data/{state}2010.pl/{state}000012010.pl"
    df_file1 = pd.read_csv(
        filename, header=None, names=cols, usecols=usecols, dtype=float
    )
    df_file1 = df_file1.rename(columns=CEN_TO_NAME_DICT)
    df_file1["nh_api"] = df_file1["nh_asian"] + df_file1["nh_pac_isl"]

    return df_file1


def census2010_file2(state):
    """
    read the 2010 census redistricting file 2

    Parameters
    ----------
    state : string
        us state

    Returns
    -------
    pandas dataframe
        file 2 of redistricting file
    """
    cols = file2_info_2010().field_short.values
    # columns of interest
    usecols = [
        "P0030001",
        "P0030002",
        "P0030003",
        "P0030004",
        "P0030005",
        "P0030006",
        "P0030007",
        "P0030008",
        "P0030009",
        "P0040002",
        "P0040003",
        "P0040004",
        "P0040005",
        "P0040006",
        "P0040007",
        "P0040008",
        "P0040009",
        "P0040010",
        "P0040011",
    ]
    filename = "census_data/fl2010.pl/fl000022010.pl"
    df_file2 = pd.read_csv(
        f"census_data/{state}2010.pl/{state}000022010.pl",
        sep=",",
        header=None,
        names=cols,
        usecols=usecols,
        dtype=float,
    )

    # change census codes to interpretable names
    df_file2 = df_file2.rename(columns=CEN_TO_NAME_DICT)
    # combine asian and pacific-islander
    df_file2.loc[:, "nh_api18"] = df_file2["nh_asian18"] + df_file2["nh_pac_isl18"]

    return df_file2


if __name__ == "__main__":
    main()
