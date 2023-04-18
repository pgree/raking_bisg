"""
this module converts the state of vermont voter file into a pandas dataframe.
in order to get the vermont voter file, i emailed a form found here --
https://outside.vermont.gov/dept/sos/Elections%20Division/voters/2019-sworn-affidavit-checklist-request.pdf
-- to the vermont secretary of state and they added me to monthly distributions of the voter file.
The file is saved to state_voter_files/vermont/9.7.2022Statewidevoters (38).txt.
n.b. i left the filename unchanged from how it was sent

Functions
---------
vt_voter_file
    get the voter file in pandas dataframe form
"""
import os
import pandas as pd


def main():
    vt_voter_file(load=False, save=True)


def vt_voter_file(load=True, save=True):
    """
    pull the state of vermont voter file

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
    file_out = f"generated_data/vt2020_vf.feather"
    if os.path.exists(file_out) and load:
        print(f"*loading dataframe from {file_out}")
        return pd.read_feather(file_out)

    filename = "state_voter_files/vermont/9.7.2022Statewidevoters (38).txt"
    usecols = ["Last Name", "County"]
    df = pd.read_csv(filename, sep="|", usecols=usecols)
    df["name"] = df["Last Name"].str.lower()
    df["county"] = df["County"].str.lower()

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
