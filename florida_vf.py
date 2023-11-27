"""
This module contains functions for creating a pandas dataframe version of the
florida voter registration list. The florida voter file is available upon request
from -- https://dos.myflorida.com/elections/data-statistics/voter-registration-statistics/voter-extract-disk-request/

the voter file fields are included in 'state_voter_files/florida/voterfile_cols.txt' which
is a manually created text file from the documentation of the voter file. in florida, there
is one voter file per county and all county files are saved in the directory
'state_voter_files/florida/20201208_VoterDetail'

Functions
---------
fl_voter_file
    pandas dataframe version of the voter file
"""
import pandas as pd
import os

# the race/ethnicity categories used by the florida voter file
VOTER_RACES_STR = {
    "1": "nh_aian",
    "2": "nh_api",
    "3": "nh_black",
    "4": "hispanic",
    "5": "nh_white",
    "6": "other",
    "7": "multi-racial",
    "9": "unknown",
}

COUNTY_DICT = {
    "ALA": "Alachua",
    "BAK": "Baker",
    "BAY": "Bay",
    "BRA": "Bradford",
    "BRE": "Brevard",
    "BRO": "Broward",
    "CAL": "Calhoun",
    "CHA": "Charlotte",
    "CIT": "Citrus",
    "CLA": "Clay",
    "CLL": "Collier",
    "CLM": "Columbia",
    "DAD": "Miami-Dade",
    "DES": "Desoto",
    "DIX": "Dixie",
    "DUV": "Duval",
    "ESC": "Escambia",
    "FLA": "Flagler",
    "FRA": "Franklin",
    "GAD": "Gadsden",
    "GIL": "Gilchrist",
    "GLA": "Glades",
    "GUL": "Gulf",
    "HAM": "Hamilton",
    "HAR": "Hardee",
    "HEN": "Hendry",
    "HER": "Hernando",
    "HIG": "Highlands",
    "HIL": "Hillsborough",
    "HOL": "Holmes",
    "IND": "Indian River",
    "JAC": "Jackson",
    "JEF": "Jefferson",
    "LAF": "Lafayette",
    "LAK": "Lake",
    "LEE": "Lee",
    "LEO": "Leon",
    "LEV": "Levy",
    "LIB": "Liberty",
    "MAD": "Madison",
    "MAN": "Manatee",
    "MRN": "Marion",
    "MRT": "Martin",
    "MON": "Monroe",
    "NAS": "Nassau",
    "OKA": "Okaloosa",
    "OKE": "Okeechobee",
    "ORA": "Orange",
    "OSC": "Osceola",
    "PAL": "PalmBeach",
    "PAS": "Pasco",
    "PIN": "Pinellas",
    "POL": "Polk",
    "PUT": "Putnam",
    "SAN": "SantaRosa",
    "SAR": "Sarasota",
    "SEM": "Seminole",
    "STJ": "St.Johns",
    "STL": "St.Lucie",
    "SUM": "Sumter",
    "SUW": "Suwannee",
    "TAY": "Taylor",
    "UNI": "Union",
    "VOL": "Volusia",
    "WAK": "Wakulla",
    "WAL": "Walton",
    "WAS": "Washington",
}

FL_COUNTY_REGIONS_DICT = {
    "northwest": [
        "Escambia",
        "SantaRosa",
        "Okaloosa",
        "Walton",
        "Holmes",
        "Washington",
        "Bay",
        "Jackson",
        "Calhoun",
        "Gulf",
        "Franklin",
        "Liberty",
    ],
    "northcentral": [
        "Gadsden",
        "Leon",
        "Wakulla",
        "Jefferson",
        "Madison",
        "Taylor",
        "Dixie",
        "Lafayette",
        "Suwannee",
        "Hamilton",
        "Columbia",
        "Gilchrist",
        "Levy",
        "Alachua",
        "Bradford",
        "Union",
    ],
    "northeast": ["Baker", "Nassau", "Duval", "Clay", "St.Johns", "Putnam", "Flagler"],
    "centralwest": [
        "Citrus",
        "Hernando",
        "Pasco",
        "Hillsborough",
        "Pinellas",
        "Manatee",
        "Sarasota",
        "Desoto",
    ],
    "central": [
        "Marion",
        "Sumter",
        "Lake",
        "Orange",
        "Seminole",
        "Osceola",
        "Polk",
        "Hardee",
        "Highlands",
    ],
    "centraleast": ["Volusia", "Brevard", "Indian River", "Okeechobee", "St.Lucie"],
    "southwest": ["Glades", "Charlotte", "Lee", "Hendry", "Collier"],
    "southeast": ["Martin", "PalmBeach", "Broward", "Monroe", "Miami-Dade"],
}

FL_RURAL_COUNTIES = [
    'Walton',
    'Holmes',
    'Washington',
    'Jackson',
    'Calhoun',
    'Gulf',
    'Gadsden',
    'Liberty',
    'Franklin',
    'Wakulla',
    'Jefferson',
    'Madison',
    'Taylor',
    'Hamilton',
    'Suwannee',
    'Lafayette',
    'Dixie',
    'Columbia',
    'Gilchrist',
    'Levy',
    'Baker',
    'Union',
    'Bradford',
    'Hardee',
    'Desoto',
    'Highlands',
    'Okeechobee',
    'Glades',
    'Hendry',
    'Monroe'
]

COUNTIES_POP_ORDER = ['Lafayette', 'Liberty', 'Glades', 'Union', 'Hamilton', 'Calhoun',
       'Franklin', 'Jefferson', 'Gulf', 'Dixie', 'Holmes', 'Madison',
       'Gilchrist', 'Taylor', 'Hardee', 'Baker', 'Washington', 'Bradford',
       'Desoto', 'Hendry', 'Okeechobee', 'Wakulla', 'Suwannee', 'Jackson',
       'Levy', 'Gadsden', 'Columbia', 'Putnam', 'Monroe', 'Walton',
       'Highlands', 'Nassau', 'Flagler', 'Sumter', 'Citrus', 'Martin',
       'Indian River', 'Bay', 'Hernando', 'SantaRosa', 'Okaloosa', 'Charlotte',
       'Clay', 'Alachua', 'St.Johns', 'Leon', 'St.Lucie', 'Escambia',
       'Collier', 'Osceola', 'Lake', 'Marion', 'Manatee', 'Seminole',
       'Sarasota', 'Pasco', 'Volusia', 'Brevard', 'Polk', 'Lee', 'Duval',
       'Pinellas', 'Orange', 'Hillsborough', 'PalmBeach', 'Broward',
       'Miami-Dade']

COUNTY_TO_REGION_DICT = {}
for key in FL_COUNTY_REGIONS_DICT:
    for county in FL_COUNTY_REGIONS_DICT[key]:
        COUNTY_TO_REGION_DICT[county] = key


def main():
    df_vf = fl_voter_file(verbose=True, save=True, load=False)
    quit()
    df_vf['county_long'] = df_vf['county'].map(COUNTY_DICT)
    df_vf['rural'] = df_vf['county_long'].isin(FL_RURAL_COUNTIES)
    print(df_vf.head())

    rural_counties2 = df_vf[df_vf['rural']]['county_long'].unique()
    rural_counties2.sort()
    print(rural_counties2)

    FL_RURAL_COUNTIES.sort()
    print(FL_RURAL_COUNTIES)





def fl_voter_file(min_names=0, verbose=False, load=True, save=True):
    """
    first reads from a file the information about the format
    of Florida's detailed voter file (column names and number of characters
    in each column). Then, using that information, convert the voter file
    to a pandas dataframe.

    Parameters
    ----------
    min_names : int, optional
        exclude names that appear fewer than this many times
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
    file_out = f"generated_data/florida_vf.feather"
    if os.path.exists(file_out) and load:
        print(f"*loading dataframe from {file_out}")
        return pd.read_feather(file_out)

    # read in file with with information on columns of voterfile data
    filename = "state_voter_files/florida/voterfile_cols.txt"
    df_vf_format = pd.read_csv(
        filename, sep="\t", index_col=0, header=None, names=["name", "chars", "notes"]
    )

    # if the dataframe hasn't yet been created, then create it
    # names of columns to use
    usecols = [
        "County Code",
        "Name Last ",
        "Residence Address Line 1",
        "Residence City (USPS)",
        "Residence Zipcode",
        "Gender",
        "Race",
        "Party Affiliation",
        "Voter Status",
    ]
    # more convenient column names
    renamed_cols = [
        "county",
        "name",
        "address",
        "city",
        "zipcode",
        "gender",
        "race",
        "party",
        "voter_status",
    ]

    # the florida voter file consists of separate .txt files for each
    # county. go through to each county and read in the data
    dfs = []
    for filename in os.listdir("state_voter_files/florida/20201208_VoterDetail"):
        if filename[-12:] != "20201208.txt":
            continue
        fn = f"state_voter_files/florida/20201208_VoterDetail/{filename}"
        if verbose:
            print(f"reading {fn}")
        df = pd.read_csv(
            fn,
            sep="\t",
            header=None,
            dtype=str,
            names=df_vf_format["name"],
            usecols=usecols,
        )
        df.columns = renamed_cols
        dfs.append(df)
    df = pd.concat(dfs)

    # make all names lowercase
    df["name"] = df["name"].str.lower()

    # change race/ethnicity convention
    df["race"].replace(VOTER_RACES_STR, inplace=True)

    df = filter_vf(df, min_names=min_names, verbose=verbose)

    # add region
    df["region"] = df["county"].map(lambda x: COUNTY_DICT.get(x, x))
    df["region"] = df["region"].map(lambda x: COUNTY_TO_REGION_DICT.get(x, x))

    # add rural or urban designation
    df['county_long'] = df['county'].map(COUNTY_DICT)
    df['rural'] = df['county_long'].isin(FL_RURAL_COUNTIES)
    print(df)
    # write dataframe to feather, but first reset index as required by feather
    if save:
        df = df.reset_index(drop=True)
        df.to_feather(file_out)

    return df


def filter_vf(df_vf0, min_names=0, verbose=False):
    """
    the 12.2020 voter file has 15mil entries and 740K unique names,
    but some rows need cleaning

    Parameters
    ----------
    df_vf0 : pandas dataframe
        uncleaned voter file
    min_names : int, optional
        exclude names that appear fewer than this many times

    Returns
    -------
    pandas dataframe
        cleaned voter file
    """

    # first make a copy
    df_vf = df_vf0.copy()
    if verbose:
        print(f"fl voter file total entries: {df_vf0.shape[0]}")

    # remove those with missing and redacted names (245 rows)
    df_vf = df_vf[~df_vf["name"].isnull()]
    df_vf = df_vf.loc[df_vf["name"] != "*", :]
    if verbose:
        print(f"after removing missing names: {df_vf.shape[0]} entries")

    # filter out people of unknown races
    df_vf = df_vf.loc[df_vf["race"] != "unknown", :]
    if verbose:
        print(f"after removing unknown races: {df_vf.shape[0]} entries")

    # set multi-racial to "other"
    df_vf.loc[df_vf["race"] == "multi-racial", "race"] = "other"

    # filter out names that don't appear frequently enough (~1mil)
    df_tmp = df_vf0.groupby(["name"]).size()
    df_tmp = df_tmp.to_frame("count").reset_index()
    df_tmp = df_tmp.loc[(df_tmp["count"] > min_names), :]
    names = df_tmp["name"].unique()
    df_vf = df_vf.loc[df_vf["name"].isin(names), :]

    # only active registrations
    df_vf = df_vf.loc[(df_vf["voter_status"] == "ACT"), :]
    if verbose:
        print(f"after removing inactive registrations: {df_vf.shape[0]} entries")

    # filter out any names with numbers or any names with no letters
    # does name contain a digit
    mask1 = df_vf["name"].map(lambda x: any(char.isdigit() for char in str(x)))
    # does name contain zero letters
    mask2 = ~df_vf["name"].map(lambda x: any(char.isalpha() for char in str(x)))
    mask = ~mask1 & ~mask2
    df_vf = df_vf[mask]

    return df_vf


if __name__ == "__main__":
    main()
