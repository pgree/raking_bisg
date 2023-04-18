import os
import requests
import zipfile
import io
from unlzw import unlzw

from plotting import (
    subsampled_figures_tables,
    calib_map_figures_tables,
    self_contained_figures_tables,
)


def main():
    download_plot_nc2010(dir_out="./")
    download_plot_nc2020(dir_out="./")
    download_plot_fl2020(dir_out="./")


def download_plot_fl2020(dir_out):
    # make sure that the voter file exists
    filename = "state_voter_files/florida/20201208_VoterDetail"
    assert os.path.exists(filename), f"Florida voter file {filename} not found"

    # download and unzip state census data
    zip_dir = f"census_data/fl2020.pl/"
    if not os.path.exists(zip_dir):
        census_zip_url = (
            "https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94"
            "-171/Florida/fl2020.pl.zip"
        )
        print(f"*downloading Florida census data")
        download_extract_zip(census_zip_url, zip_dir)

    # download and unzip usa census data
    zip_dir = f"census_data/us2020.npl/"
    if not os.path.exists(zip_dir):
        census_zip_url = (
            "https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94"
            "-171/National/us2020.npl.zip"
        )
        print(f"*downloading usa census data")
        download_extract_zip(census_zip_url, zip_dir)

    # download cps csv
    filename = f"cps_data/nov20pub.csv"
    if not os.path.exists(filename):
        cps_csv_url = "https://www2.census.gov/programs-surveys/cps/datasets/2020/supp/nov20pub.csv"
        print(f"*downloading cps data")
        download_csv(cps_csv_url, filename)

    # plot
    state = "fl"
    year = 2020
    print(f"plotting for {state} {year}")
    subsampled_figures_tables(state, year, dir_out, load=True)
    calib_map_figures_tables(state, year, dir_out, load=True)
    self_contained_figures_tables(state, year, dir_out, load=True)


def download_plot_nc2010(dir_out):
    # download and unzip voterfile
    if not os.path.exists("state_voter_files/north_carolina/VR_Snapshot_20101102.txt"):
        voterfile_zip_url = "https://s3.amazonaws.com/dl.ncsbe.gov/data/Snapshots/VR_Snapshot_20101102.zip"
        voterfile_dir = f"state_voter_files/north_carolina/"
        print(f"*downloading north carolina voterfile")
        download_extract_zip(voterfile_zip_url, voterfile_dir)

    # download and unzip state census data
    zip_dir = f"census_data/nc2010.pl/"
    if not os.path.exists(zip_dir):
        census_zip_url = (
            "https://www2.census.gov/census_2010/01-Redistricting_File--PL_94-171/North_Carolina/nc2010"
            ".pl.zip"
        )
        print(f"*downloading north carolina census data")
        download_extract_zip(census_zip_url, zip_dir)

    # download and unzip usa census data
    zip_dir = f"census_data/us2010.npl/"
    if not os.path.exists(zip_dir):
        census_zip_url = "https://www2.census.gov/census_2010/01-Redistricting_File--PL_94-171/National/us2010.npl.zip"
        print(f"*downloading usa census data")
        download_extract_zip(census_zip_url, zip_dir)

    # download cps csv
    filename = f"cps_data/nov10pub.dat"
    if not os.path.exists(filename):
        cps_zip_file = "https://www2.census.gov/programs-surveys/cps/datasets/2010/supp/nov10pub.dat.Z"
        download_uncompress_z(cps_zip_file, filename)

    # plot
    state = "nc"
    year = 2010
    print(f"*plotting for {state} {year}")
    subsampled_figures_tables(state, year, dir_out, load=True)
    calib_map_figures_tables(state, year, dir_out, load=True)
    self_contained_figures_tables(state, year, dir_out, load=True)


def download_plot_nc2020(dir_out):
    # download and unzip voterfile
    if not os.path.exists("state_voter_files/north_carolina/VR_Snapshot_20201103.txt"):
        voterfile_zip_url = "https://s3.amazonaws.com/dl.ncsbe.gov/data/Snapshots/VR_Snapshot_20201103.zip"
        voterfile_dir = f"state_voter_files/north_carolina/"
        print(f"*downloading north carolina voterfile")
        download_extract_zip(voterfile_zip_url, voterfile_dir)

    # download and unzip state census data
    zip_dir = f"census_data/nc2020.pl/"
    if not os.path.exists(zip_dir):
        census_zip_url = (
            "https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94"
            "-171/North_Carolina/nc2020.pl.zip"
        )
        print(f"*downloading north carolina census data")
        download_extract_zip(census_zip_url, zip_dir)

    # download and unzip usa census data
    zip_dir = f"census_data/us2020.npl/"
    if not os.path.exists(zip_dir):
        census_zip_url = (
            "https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94"
            "-171/National/us2020.npl.zip"
        )
        print(f"*downloading usa census data")
        download_extract_zip(census_zip_url, zip_dir)

    # download cps csv
    filename = f"cps_data/nov20pub.csv"
    if not os.path.exists(filename):
        cps_csv_url = "https://www2.census.gov/programs-surveys/cps/datasets/2020/supp/nov20pub.csv"
        print(f"*downloading cps data")
        download_csv(cps_csv_url, filename)

    # plot
    state = "nc"
    year = 2020
    print(f"plotting for {state} {year}")
    subsampled_figures_tables(state, year, dir_out, load=True)
    calib_map_figures_tables(state, year, dir_out, load=True)
    self_contained_figures_tables(state, year, dir_out, load=True)


def download_uncompress_z(z_url, filename):
    # create output directory if it doesn't exist
    path, _ = os.path.split(filename)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'*directory "{path}" created')
    # pull data
    r = requests.get(z_url)
    with open(filename, "wb") as fh:
        fh.write(unlzw(r.content))


def download_extract_zip(zip_url, extract_dir):
    # download zip file
    r = requests.get(zip_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    # if the output directory doesn't exist, create it
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
        print(f'*directory "{extract_dir}" created')
    z.extractall(extract_dir)


def download_csv(csv_url, filename):
    # create output directory if it doesn't exist
    path, _ = os.path.split(filename)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'*directory "{path}" created')

    r = requests.get(csv_url)
    csv_file = open(filename, "wb")
    csv_file.write(r.content)
    csv_file.close()


if __name__ == "__main__":
    main()
