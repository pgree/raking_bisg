"""
this module converts the state of washington voter file into a pandas dataframe. 
in order to get the washington voter file, fill out the form here 
https://www.sos.wa.gov/elections/vrdb/extract-requests.aspx
and the state sends you an email with the voter file. The file is saved 
to ./state_voter_files/washington/20220801_VRDB_Extract.txt
n.b. i left the filename unchanged from how it was sent

Functions
---------
wa_voter_file
    get the voter file in pandas dataframe form
"""
import os
import pandas as pd

COUNTY_CODE_TO_NAME = {
    "AD": "Adams",
    "AS": "Asotin",
    "BE": "Benton",
    "CH": "Chelan",
    "CM": "Clallam",
    "CR": "Clark",
    "CU": "Columbia",
    "CZ": "Cowlitz",
    "DG": "Douglas",
    "FE": "Ferry",
    "FR": "Franklin",
    "GA": "Garfield",
    "GR": "Grant",
    "GY": "Grays Harbor",
    "IS": "Island",
    "JE": "Jefferson",
    "KI": "King",
    "KP": "Kitsap",
    "KS": "Kittitas",
    "KT": "Klickitat",
    "LE": "Lewis",
    "LI": "Lincoln",
    "MA": "Mason",
    "OK": "Okanogan",
    "PA": "Pacific",
    "PE": "Pend Oreille",
    "PI": "Pierce",
    "SJ": "San Juan",
    "SK": "Skagit",
    "SM": "Skamania",
    "SN": "Snohomish",
    "SP": "Spokane",
    "ST": "Stevens",
    "TH": "Thurston",
    "WK": "Wahkiakum",
    "WL": "Walla Walla",
    "WM": "Whatcom",
    "WT": "Whitman",
    "YA": "Yakima",
}


def main():
    wa_voter_file(load=False, save=True)


def wa_voter_file(load=True, save=True):
    """
    pull the state of washington voter file

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
    file_out = f"generated_data/wa2020_vf.feather"
    if os.path.exists(file_out) and load:
        print(f"*loading dataframe from {file_out}")
        return pd.read_feather(file_out)
    # file send by the state of washington
    filename = "state_voter_files/washington/20220801_VRDB_Extract.txt"
    usecols = ["LName", "CountyCode", "StatusCode"]
    df = pd.read_csv(filename, sep="|", usecols=usecols)
    df = df.loc[df["StatusCode"] == "Active", :]
    df["name"] = df["LName"].str.lower()
    df["county"] = (
        df["CountyCode"].map(lambda x: COUNTY_CODE_TO_NAME.get(x, x)).str.lower()
    )
    df = df[["name", "county"]]

    # filter out  any names with numbers or any names with no letters
    # does name contain a digit
    mask1 = df["name"].map(lambda x: any(char.isdigit() for char in str(x)))
    # does name contain zero letters
    mask2 = ~df["name"].map(lambda x: any(char.isalpha() for char in str(x)))
    mask = ~mask1 & ~mask2
    df = df[mask]

    # write dataframe to feather, but first reset index as required by feather
    if save:
        df.reset_index(drop=True).to_feather(file_out)

    return df


if __name__ == "__main__":
    main()
