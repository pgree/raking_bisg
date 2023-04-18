"""
this module converts the state of oklahoma voter file into a pandas dataframe.
the voter file is available for download from https://www6.ohiosos.gov/ords/f?p=VOTERFTP:HOME
the voter file is divided into 4 files, which are saved to the directory
'state_voter_files/ohio'

a voter status of "CONFIRMATION" means "...your county board of elections has mailed you a 
confirmation notice, but you have not yet responded to that notice to confirm or update your information.
Voters who have received a confirmation notice and fail to respond to that notice or participate in any 
voter activity (such as voting or updating their voter registration information) after a period of four
years are subject to cancellation."
This information comes from looking up on this page -- https://voterlookup.ohiosos.gov/voterlookup.aspx --
a voter with voter status "confirmation".

There are two possible entries in the "voter_status" field, "ACTIVE" and "CONFIRMATION".
We include both registration statuses.

Functions
---------
oh_voter_file
    get the voter file in pandas dataframe form
"""
import os
import numpy as np
import pandas as pd


def main():
    oh_voter_file(load=False, save=True)


def oh_voter_file(load=True, save=True):
    """
    construct the ohio voter file

    Parameters
    ----------
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
    file_out = f"generated_data/oh2020_vf.feather"
    if os.path.exists(file_out) and load:
        print(f"*loading dataframe from {file_out}")
        return pd.read_feather(file_out)

    # pull in map from "county_number" to county name
    filename = "state_voter_files/ohio/ohio_counties.dat"
    df_counties = pd.read_csv(filename, header=None, names=["county"])
    df_counties["county_number"] = np.arange(1, 89)
    df_counties = df_counties.set_index("county_number")
    code_to_name_dict = df_counties.to_dict()["county"]

    # the 88 counties of ohio are split up into 4 different files
    files = ["SWVF_1_22.txt", "SWVF_23_44.txt", "SWVF_45_66.txt", "SWVF_67_88.txt"]
    dfs = []
    usecols = ["COUNTY_NUMBER", "LAST_NAME"]
    for file in files:
        filename = "state_voter_files/ohio/" + file
        df = pd.read_csv(filename, usecols=usecols)
        dfs.append(df)
    df = pd.concat(dfs)
    # make columns and entries lowercase
    df.columns = df.columns.str.lower()
    df["county"] = df["county_number"].map(lambda x: code_to_name_dict.get(x, x))
    df["county"] = df["county"].str.lower()
    df["name"] = df["last_name"].str.lower().astype(str)
    df = df[["name", "county"]]

    # filter out  any names with numbers or any names with no letters
    # does name contain a digit
    mask1 = df["name"].map(lambda x: any(char.isdigit() for char in x))
    # does name contain zero letters
    mask2 = ~df["name"].map(lambda x: any(char.isalpha() for char in x))
    mask = ~mask1 & ~mask2
    df = df[mask]

    # write dataframe to feather, but first reset index as required by feather
    if save:
        df = df.reset_index(drop=True)
        df.to_feather(file_out)

    return df


if __name__ == "__main__":
    main()
