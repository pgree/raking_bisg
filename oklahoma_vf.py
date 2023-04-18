"""
this module converts the state of oklahoma voter file into a pandas dataframe.
a voter registration list can be requested here: 
https://oklahoma.gov/content/dam/ok/en/elections/ok-election-data-warehouse/data-warehouse-request-form.pdf
There is one file per county and files are saved to
f'state_voter_files/oklahoma/CTY{county_num}_vr.csv' for county_num from 01 to 78.
n.b. i left the filenames unchanged from how it was sent

the files used here were the voter registration lists downloaded on 8/8/2022
and appears to be up to date as of that same date, or something close to it
since it has voting records from 6/2022. 

in the OK voter file, voter status of "A" is actuve and "I" inactive and 
there's a different file for each county.

Functions
---------
ok_voter_file
    get the voter file in pandas dataframe form
"""
import os
import pandas as pd

VF_TO_CEN_COUNTY = {"leflore": "le flore"}


def main():
    ok_voter_file(load=False, save=True)


def ok_voter_file(load=True, save=True):
    """
    pull the oklahoma voter file

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
    file_out = f"generated_data/ok2020_vf.feather"
    if os.path.exists(file_out) and load:
        print(f"*loading dataframe from {file_out}")
        return pd.read_feather(file_out)

    county_nums = [f"{n:02}" for n in range(1, 78)]
    # pull in map from "county_number" to county name
    usecols = ["LastName", "County_Desc", "Status"]
    dfs = []
    for county_num in county_nums:
        filename = f"state_voter_files/oklahoma/CTY{county_num}_vr.csv"
        df = pd.read_csv(filename, usecols=usecols)
        dfs.append(df)
    df = pd.concat(dfs)
    df.columns = df.columns.str.lower()
    # filter out inactive registrations
    mask = df["status"] == "A"
    df = df.loc[mask, :]
    # make names and counties lowercase
    df["name"] = df["lastname"].str.lower()
    df["county"] = df["county_desc"].str.lower()
    df["county"] = df["county"].map(lambda x: VF_TO_CEN_COUNTY.get(x, x))

    # filter out  any names with numbers or any names with no letters
    # does name contain a digit
    mask1 = df["name"].map(lambda x: any(char.isdigit() for char in str(x)))
    # does name contain zero letters
    mask2 = ~df["name"].map(lambda x: any(char.isalpha() for char in str(x)))
    mask = ~mask1 & ~mask2
    df = df[mask]

    # keep only useful columns
    df = df[["name", "county"]]

    # write dataframe to feather, but first reset index as required by feather
    if save:
        df.reset_index(drop=True).to_feather(file_out)

    return df


if __name__ == "__main__":
    main()
