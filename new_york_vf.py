"""
This module contains functions for creating a pandas dataframe version of the
new york state voter registration list. The data from the voter file comes from a
disk that was requested from new york state via a freedom of information
request here -- https://www.elections.ny.gov/FoilRequests.html.
the text file 'ny_county_raw.dat' contains the map from county code to 
county name and 'ny_fields.dat' contains the field (column) names of the 
voter file in 'AllNYSVoters_20220801.txt'. the files 'ny_county_raw.dat' and
'ny_fields.dat' were made manually by copy-paste from the voter file 
documentation. 

Voter Status Codes:
-------------------
A = Active
AM = Active Military
AF = Active Special Federal
AP = Active Special Presidential 
AU = Active UOCAVA
I = Inactive
P = Purged
17 = Prereg – Older than 16 years but younger than 18 years

we include voter statuses other than inactive, purged, and preregistered

Functions
---------
ny_voter_file
    pandas dataframe version of the voter file
"""
import os
import pandas as pd


def main():
    ny_voter_file()


def ny_voter_file(load=True, save=True):
    """
     construct a dataframe with name, county, race/ethnicity for
     registered voters in new york.

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
    # after creating the dataframe, we'll save it as a feather
    file_out = "generated_data/new_york_vf.feather"
    # ...but if the file has already been created, load it and return
    if os.path.exists(file_out) and load:
        print(f"*loading dataframe from {file_out}")
        return pd.read_feather(file_out)

    # read in the county code to county name map
    filename = "state_voter_files/new_york/ny_county_raw.dat"
    df = pd.read_csv(filename, names=["county_name", "county_code"], dtype=object)
    df["county_code"] = df["county_code"].str.strip()
    code_to_name_dict = df.set_index("county_code").to_dict()["county_name"]

    # read in the field names of the voter file
    filename = "state_voter_files/new_york/ny_fields.dat"
    df = pd.read_csv(filename, header=None, names=["field"])
    col_names = df["field"].str.lower().values

    # read the voter file
    usecols = ["lastname", "countycode", "status"]  # voter registration status
    filename = "state_voter_files/new_york/AllNYSVoters_20220801.txt"
    df = pd.read_csv(
        filename, header=None, names=col_names, usecols=usecols, dtype=object
    )

    # remove inactive, purged, and preregistered voters
    df = df.loc[~df["status"].isin(["I", "P", "17"])]
    # map countycode onto county names
    df["county"] = df["countycode"].map(lambda x: code_to_name_dict.get(x, x))
    df["county"] = df["county"].str.lower()
    # convert surnames to lowercase
    df["name"] = df["lastname"].str.lower()
    # drop columns other than name and county
    df = df[["name", "county"]]
    # write the file as a feather so it can be loaded in the future
    if save:
        df.reset_index(drop=True).to_feather(file_out)

    return df


if __name__ == "__main__":
    main()
