"""
this module includes functions for reading in microdata of the 2010 and 2020
voter supplement of the current population survey (CPS). the 2020 data comes
from a different location from the 2010 data and is read differently.

in the cps, only non-institutionalized populations are sampled (institutionalized
consist primarily of correctional institutions and nursing homes)

2020:
microdata can be downloaded here
  -- https://www.census.gov/data/datasets/time-series/demo/cps/cps-supp_cps-repwgt/cps-voting.html
information on microdata -- https://www2.census.gov/programs-surveys/cps/techdocs/cpsnov20.pdf
survey methodology -- https://www.census.gov/housing/hvs/files/tp-66.pdf
the 2020 microdata should be located at ./cps_data/cps_microdata/nov20pub.csv

2010:
microdata can be downloaded here --
https://www.census.gov/data/datasets/time-series/demo/cps/cps-supp_cps-repwgt/cps-voting.2010.html#list-tab-1089297917
-- as can documentation.
information about the formatting of the file comes from --
https://www.nber.org/research/data/reading-current-population-survey-cps-data-sas-spss-or-stata --
in particular, the file 'cps2010_fields.dat' in this repository is a copy of part of the SAS file from
the preceding link.

Functions
---------
cps_geo_race
    race/ethnicity distribution of a us state or the full country in 2010 or 2020

"""
import os
import pandas as pd
import numpy as np
import fields

CPS_HISPANIC_CODES = {"1": "hispanic", "2": "non-hispanic", "-1": "unknown"}

CPS_RACE_CODES = {
    "-3": "blank",
    "-2": "don't know",
    "-1": "refused",
    "1": "White Only",
    "2": "Black Only",
    "3": "American Indian, Alaskan Native Only",
    "4": "Asian Only",
    "5": "Hawaiian/Pacific Islander Only",
    "6": "White-Black",
    "7": "White-AI",
    "8": "White-Asian",
    "9": "White-HP",
    "10": "Black-AI",
    "11": "Black-Asian",
    "12": "Black-HP",
    "13": "AI-Asian",
    "14": "AI-HP",
    "15": "Asian-HP",
    "16": "W-B-AI",
    "17": "W-B-A",
    "18": "W-B-HP",
    "19": "W-AI-A",
    "20": "W-AI-HP",
    "21": "W-A-HP",
    "22": "B-AI-A",
    "23": "W-B-AI-A",
    "24": "W-AI-A-HP",
    "25": "Other 3 Race Combinations",
    "26": "Other 4 and 5 Race Combinations}",
}

CPS_REG_TO_VOTE_CODES = {
    "1": "Yes",
    "2": "No",
    "-1": "Not in Universe",
    "-2": "Don't Know",
    "-3": "Refused",
    "-9": "No Response",
}

# the following is the MacDonald classification for non-hispanics
CPS_RACE_MAP = {
    "American Indian, Alaskan Native Only": "nh_aian",
    "Asian Only": "nh_api",
    "Black Only": "nh_black",
    "Black-AI": "nh_black",
    "Black-Asian": "nh_black",
    "Black-HP": "nh_black",
    "Hawaiian/Pacific Islander Only": "nh_api",
    "White Only": "nh_white",
    "White-AI": "other",
    "White-Asian": "nh_api",
    "White-Black": "nh_black",
    "White-HP": "other",
    "W-AI-A-HP": "other",
    "W-B-AI": "nh_black",
    "Asian-HP": "nh_api",
    "W-B-A": "nh_black",
    "W-A-HP": "nh_api",
    "Other 3 Race Combinations": "other",
    "W-B-AI-A": "nh_black",
    "AI-Asian": "nh_api",
    "W-AI-HP": "other",
    "W-B-HP": "nh_black",
    "Other 4 and 5 Race Combinations}": "other",
    "W-AI-A": "nh_api",
    "AI-HP": "other",
}

FIPS_TO_STATE = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}

# new dictionary with flipped keys and values
STATE_TO_FIPS = dict([(value, key) for key, value in FIPS_TO_STATE.items()])

# only columns used from cps microdata
CPS_COLS = ["state", "race", "reg_to_vote", "final_wt"]


def main():
    # compare the dataframe already written to file to the one created here
    year = 2010
    state = "fl"
    # df_file = pd.read_feather(f"generated_data/cps_{state}{year}_voter_true.feather")
    df_new = cps_geo_race(state, year=year, voters=True, load=False, save=False)
    # print(df_file)
    print(df_new)


def cps_geo_race(state, year, voters, load=True, save=False):
    """
    get the race distribution of a certain state (or the full USA)
    for all people surveyed or just for registered voters

    Parameters
    ----------
    state : string
        state, e.g., 'fl' or 'GA'
    year : int
        year
    voters : bool
        restrict population to registered voters
    load : bool, optional
        if this dataframe has already been created, load it
    save : bool, optional
        write the dataframe after creating it

    Returns
    -------
    pandas dataframe
        race distribution for state/national
    """

    # state abbreviation must be capitalized
    state = state.upper()

    # if this dataframe has already been created, read and return it
    file_out = (
        f"generated_data/cps_{state.lower()}{year}_voter_{str(voters).lower()}.feather"
    )
    if os.path.exists(file_out) and load:
        print(f"*loading dataframe from {file_out}")
        return pd.read_feather(file_out)

    # pull in cleaned microdata
    df_cps0 = read_cps_micro(year=year)

    # limit responses to a single state, unless otherwise specified (fl is 12)
    if state != "USA":
        mask = df_cps0["state"] == state
        df_cps0 = df_cps0[mask]
    if voters:
        df_cps0 = df_cps0[df_cps0["reg_to_vote"]]

    # take weighted average of final weights to get race distribution
    # for registered voters
    df = df_cps0[["race", "final_wt"]].groupby("race").sum("final_wt")
    df = df.T
    df = df.div(df.sum().sum())
    # in some states, some subpopulations are not represented. set those to 0.
    for race in fields.RACES:
        if race not in df.columns:
            df[race] = 0.0
    df = df[fields.RACES]
    df = df.reset_index(drop=True)

    # add state field and put it first
    df["state"] = state
    cols = list(df.columns)
    new_cols = [cols[-1]] + cols[:-1]
    df = df[new_cols]
    # write dataframe to feather, but first reset index as required by feather
    if save:
        # if this dataframe has already been created, read and return it
        dir_out = "generated_data"
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        df = df.reset_index(drop=True)
        df.to_feather(file_out)

    return df


def read_cps2010_micro() -> pd.DataFrame:
    """
    Pull in the 2010 voter supplement of the cps

    Returns
    -------
    pandas dataframe
        2010 cps voter supplement data
    """
    # read the field names and number of characters for each file
    df = pd.read_fwf(
        "cps_data/cps2010_fields.dat",
        header=None,
        names=["start", "field_name", "width"],
    )
    df["start"] = df["start"].str[1:].astype(np.int64) - 1
    # widths have a $ before some entries, and .x afterwards. remove both of these
    df["width"] = df["width"].str.replace("$", "", regex=False)
    df["width"] = df["width"].str.split('.').str[0]
    df["width"] = df["width"].astype(np.int64)
    widths = df["start"].diff().shift(-1).fillna(df["width"].values[-1]).astype(np.int64)
    usecols = ["pwsswgt", "gtcbsa", "gestfips", "pehspnon", "ptdtrace", "pes1", "pes2"]
    df_cps0 = pd.read_fwf(
        "cps_data/nov10pub.dat",
        widths=widths,
        names=df["field_name"],
        usecols=usecols,
        dtype=object,
    )
    return df_cps0


def read_cps_micro(year: int) -> pd.DataFrame:
    """
    read the microdata from the cps voter supplement and extract useful columns

    Parameters
    ----------
    year : int
        year

    Returns
    -------
    pandas dataframe
        cps survey responses
    """
    if year == 2020:
        filename = "cps_data/nov20pub.csv"
        df_cps0 = pd.read_csv(filename, dtype=str)
    elif year == 2010:
        df_cps0 = read_cps2010_micro()
    else:
        raise Exception(f"year {year} currently not supported")

    # make all column titles lowercase for convenience
    df_cps0.columns = df_cps0.columns.str.lower()

    # filter out final weights of 0 which can result from people who
    # couldn't be interviewed or people who refused to participate
    df_cps0 = df_cps0.rename(columns={"pwsswgt": "final_wt"})
    df_cps0 = df_cps0[df_cps0["final_wt"] != "0"]

    # filter out nan's
    df_cps0 = df_cps0[~df_cps0["final_wt"].isna()]

    # convert to float
    df_cps0["final_wt"] = df_cps0["final_wt"].astype(float)

    # make column names and entries interpretable
    df_cps0 = df_cps0.rename(columns={"gtcbsa": "metro_area"})
    # there are -1 entries in the following, i.e., neither
    # hispanic nor non-hispanic
    df_cps0["hispanic"] = df_cps0["pehspnon"].map(CPS_HISPANIC_CODES)
    df_cps0["race0"] = df_cps0["ptdtrace"].map(CPS_RACE_CODES)
    df_cps0["voted2020"] = df_cps0["pes1"].map(CPS_REG_TO_VOTE_CODES)
    df_cps0["reg_to_vote0"] = df_cps0["pes2"].map(CPS_REG_TO_VOTE_CODES)
    df_cps0["state"] = df_cps0["gestfips"].map(FIPS_TO_STATE)

    # registered to vote (either voted or didn't vote but registered to vote)
    df_cps0["reg_to_vote"] = (df_cps0["voted2020"] == "Yes") | (
        df_cps0["reg_to_vote0"] == "Yes"
    )

    # create the 6-group race/eth classification from the
    # cps classification using mcdonald's approach
    df_cps0 = mcdonald_race_eth(df_cps0)

    # remove extraneous columns
    df_cps0 = df_cps0[CPS_COLS]

    return df_cps0


def mcdonald_race_eth(df_cps0):
    """
    McDonald classification for converting CPS race to 6 categories. the
    classification is described in his 2007 paper: The True Electorate: A
    Cross-Validation of Voter Registration Files and Election Survey Demographics:
    " All CPS respondents reporting Hispanic ethnicity are scored as Hispanic.
    All non-hispanics reporting a single race only are reported as that race.
    Asian and Hawaiian-Pacific Islander are grouped into an Asian category.
    For multiple-race categories, non-Hispanics reporting black in any other combination are
    scored as black. Among the remainder, non-Hispanics reporting Asian or Hawaiian-Pacific
    Islander in combination with any other remaining race are identified as Asian.
    Those remaining are classified as Other"

    Parameters
    ----------
    df_cps0 : pandas dataframe
        cps microdata without the 6 race/ethnicity classifications

    Returns
    -------
    pandas dataframe
        the inputted dataframe with a 'race' column with 6 subpopulations

    """
    # classify everyone by starting with "unclassified"
    df_cps0["race"] = "unclassified"
    # all hispanics are listed as hispanic
    df_cps0.loc[df_cps0["hispanic"] == "hispanic", "race"] = "hispanic"
    # non-hispanics are mapped according to the dictionary
    mask = df_cps0["race"] == "unclassified"
    df_cps0.loc[mask, "race"] = df_cps0.loc[mask, "race0"].map(
        lambda x: CPS_RACE_MAP.get(x, x)
    )
    # set all remaining to other
    df_cps0.loc[df_cps0["race"].isna(), "race"] = "other"

    return df_cps0


if __name__ == "__main__":
    main()
