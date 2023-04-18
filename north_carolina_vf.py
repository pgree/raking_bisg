"""
this module converts the North Carolina voter file into a pandas dataframe.
voter files can be downloaded here -- https://www.ncsbe.gov/results-data/voter-registration-data
the 2010 and 2020 voter files are saved to:
'state_voter_files/north_carolina/VR_Snapshot_20101102.txt' and
'state_voter_files/north_carolina/VR_Snapshot_20201103.txt'

the definitions of regions we use are from here -- https://www.ncpedia.org/our-state-geography-snap-three

Functions
---------
nc_voter_file
    get the voter file in pandas dataframe form
"""
import numpy as np
import pandas as pd
import os


# dictionary whose keys are columns names in the voter file and
# values are more easily read
NC_RACE_DESC_DICT = {
    "WHITE": "white",
    "BLACK or AFRICAN AMERICAN": "black",
    "OTHER": "other",
    "UNDESIGNATED": "undesignated",
    "INDIAN AMERICAN or ALASKA NATIVE": "aian",  # the 2020 field name
    "TWO or MORE RACES": "other",
    "ASIAN": "api",
    "NATIVE HAWAIIAN or PACIFIC ISLANDER": "api",
    "AMERICAN INDIAN or ALASKA NATIVE": "aian",  # the 2010 field name
}

NC_COUNTY_REGIONS_DICT = {
    "west": [
        "cherokee",
        "graham",
        "clay",
        "swain",
        "macon",
        "jackson",
        "haywood",
        "transylvania",
        "madison",
        "buncombe",
        "henderson",
        "yancey",
        "mcdowell",
        "rutherford",
        "polk",
        "mitchell",
        "avery",
        "burke",
        "watauga",
        "caldwell",
        "ashe",
        "wilkes",
        "alleghany",
    ],
    "central": [
        "surry",
        "yadkin",
        "cleveland",
        "alexander",
        "catawba",
        "lincoln",
        "gaston",
        "iredell",
        "mecklenburg",
        "stokes",
        "forsyth",
        "davie",
        "rowan",
        "cabarrus",
        "union",
        "davidson",
        "stanly",
        "anson",
        "rockingham",
        "guilford",
        "randolph",
        "montgomery",
        "richmond",
        "caswell",
        "alamance",
        "chatham",
        "moore",
        "lee",
        "person",
        "orange",
        "durham",
        "granville",
        "wake",
        "vance",
        "franklin",
        "warren",
    ],
    "east": [
        "scotland",
        "robeson",
        "columbus",
        "brunswick",
        "hoke",
        "harnett",
        "cumberland",
        "bladen",
        "new hanover",
        "johnston",
        "sampson",
        "pender",
        "nash",
        "wilson",
        "wayne",
        "duplin",
        "halifax",
        "edgecombe",
        "greene",
        "lenoir",
        "jones",
        "onslow",
        "northampton",
        "hertford",
        "bertie",
        "martin",
        "pitt",
        "craven",
        "carteret",
        "beaufort",
        "pamlico",
        "gates",
        "currituck",
        "camden",
        "pasquotank",
        "perquimans",
        "chowan",
        "washington",
        "tyrrell",
        "dare",
        "hyde",
    ],
}

COUNTY_TO_REGION_DICT = {}
for key in NC_COUNTY_REGIONS_DICT:
    for county in NC_COUNTY_REGIONS_DICT[key]:
        COUNTY_TO_REGION_DICT[county] = key


def main():
    year = 2010
    df_vf = nc_voter_file(year=year, verbose=False, load=False, save=True)
    print(df_vf)

    year = 2020
    df_vf = nc_voter_file(year=year, verbose=False, load=False, save=True)
    print(df_vf)


def nc_voter_file(year, verbose=False, load=True, save=False):
    """
    construct a dataframe with name, county, race/ethnicity for
    registered voters in north carolina.

    Parameters
    ----------
    year : int
        one of 2010 and 2020
    verbose : bool, optional
        verbose
    load : bool, optional
        load the dataframe if it already exists
    save : bool, optional
        save the dataframe after creating it

    Returns
    -------
    pandas dataframe
        voter file
    """

    # if this dataframe has already been created, read and return it
    file_out = f"generated_data/nc{year}_vf.feather"
    if os.path.exists(file_out) and load:
        print(f"*loading dataframe from {file_out}")
        return pd.read_feather(file_out)

    # location of raw voter file, which is a txt file
    if year == 2020:
        filename = "state_voter_files/north_carolina/VR_Snapshot_20201103.txt"
    elif year == 2010:
        filename = "state_voter_files/north_carolina/VR_Snapshot_20101102.txt"
    nc_hisp_dict = {
        "NOT HISPANIC or NOT LATINO": "nh",
        "UNDESIGNATED": "undesignated",
        "HISPANIC or LATINO": "hispanic",
    }
    # columns to extract from the raw text file
    usecols = [
        "county_desc",
        "last_name",
        "first_name",
        "race_desc",
        "ethnic_desc",
        "voter_status_desc",
    ]

    # interpretable race/eth categories
    df_tmp = pd.read_csv(filename, sep="\t", encoding="utf-16", usecols=usecols)
    df_tmp["race_only"] = df_tmp["race_desc"].map(lambda x: NC_RACE_DESC_DICT.get(x, x))
    df_tmp["eth"] = df_tmp["ethnic_desc"].map(lambda x: nc_hisp_dict.get(x, x))
    if verbose:
        print(f"nc voter file total entries: {df_tmp.shape[0]}")

    # remove inactive registrations: "Voter registrations with a
    # voter_status_desc of 'Removed' are omitted whenever the most recent
    # last voted date is greater than 10 years."
    df_tmp = df_tmp[(df_tmp["voter_status_desc"] == "ACTIVE")]
    if verbose:
        print(f"total entries after removing inactive: {df_tmp.shape[0]}")

    # remove undesignated race/ethnicity
    df_tmp = df_tmp[~(df_tmp["race_only"] == "undesignated")]
    df_tmp = df_tmp[~(df_tmp["eth"] == "undesignated")]
    if verbose:
        print(f"total entries after removing undesignated: {df_tmp.shape[0]}")

    # 6 race/ethnicities
    df_tmp["race"] = np.nan
    # hispanic
    mask = df_tmp["eth"] == "hispanic"
    df_tmp.loc[mask, "race"] = "hispanic"
    # nh_aian
    mask = (df_tmp["eth"] == "nh") & (df_tmp["race_only"] == "aian")
    df_tmp.loc[mask, "race"] = "nh_aian"
    # nh_api
    mask = (df_tmp["eth"] == "nh") & (df_tmp["race_only"] == "api")
    df_tmp.loc[mask, "race"] = "nh_api"
    # nh_black
    mask = (df_tmp["eth"] == "nh") & (df_tmp["race_only"] == "black")
    df_tmp.loc[mask, "race"] = "nh_black"
    # nh_white
    mask = (df_tmp["eth"] == "nh") & (df_tmp["race_only"] == "white")
    df_tmp.loc[mask, "race"] = "nh_white"
    # other
    mask = (df_tmp["eth"] == "nh") & (df_tmp["race_only"] == "other")
    df_tmp.loc[mask, "race"] = "other"

    # county
    df_tmp["county"] = df_tmp["county_desc"].str.lower()

    # region
    df_tmp["region"] = df_tmp["county"].map(lambda x: COUNTY_TO_REGION_DICT.get(x, x))

    # name
    df_tmp["name"] = df_tmp["last_name"].str.lower()
    df_tmp = df_tmp[~df_tmp["name"].isnull()]

    # filter out names that don't appear frequently enough with minimum names
    # with 3, we have 74,000 unique names compared to 300,000. however, total
    # people in data set only decreases from ~.48 mil to ~.45 mil
    df_names = df_tmp.groupby(["name"]).size()
    df_names = df_names.to_frame("count").reset_index()
    min_names = 0
    df_names = df_names.loc[(df_names["count"] > min_names), :]
    names = df_names["name"].unique()
    df_tmp = df_tmp.loc[df_tmp["name"].isin(names), :]
    df_vf = df_tmp[["county", "name", "race", "region"]]

    # filter out  any names with numbers or any names with no letters
    # does name contain a digit
    mask1 = df_vf["name"].map(lambda x: any(char.isdigit() for char in x))
    # does name contain zero letters
    mask2 = ~df_vf["name"].map(lambda x: any(char.isalpha() for char in x))
    mask = ~mask1 & ~mask2
    df_vf = df_vf[mask]

    # write dataframe to feather, but first reset index as required by feather
    if save:
        df_vf = df_vf.reset_index(drop=True)
        df_vf.to_feather(file_out)

    return df_vf


if __name__ == "__main__":
    main()
