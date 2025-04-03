# Race/ethnicity predictions via raking and BISG

This repository includes all software for reproducing the figures and tables of the paper 
"BISG: When inferring race or ethnicity, does it matter that people often live near 
their relatives?" by Philip Greengard and Andrew Gelman 
([pre-print](https://arxiv.org/abs/2304.09126)). 

The code in this repository constructs race/ethnicity (subpopulation) predictions for registered voters in 
several states. 

## Pandas dataframe of predictions
The main output of the codes of this repository is a Pandas dataframe named `df_agg`, 
that is built in `raking_bisg.py`.  Each row of `df_agg` contains race/ethnicity predictions corresponding to one (surname, county) pair in one state. 
For example, the New York dataframe contains a row for (Smith, Queens) which includes 
race/ethnicity predictions for a registered voter named "Smith" in Queens County, New York. 

The codes in this repository construct these dataframes for a number of predictions 
and states. Dataframes are available to download 
[here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FQIM4UF). 
The states and predictions included (as of 4/7/2023) are as follows. 

Predictions:
Census race given county,
Census 18+ race given county,
Census race given surname, 
Voter-adjusted race given county, 
Voter-adjusted race given surname,
BISG (full US population),
Voter BISG (registered voter population in state),
Raking (registered voter population in state)

States: 
Florida (race/ethnicity labeled),
North Carolina (race/ethnicity labeled),
New York,
Ohio,
Oklahoma,
Vermont,
Washington State

After downloading the dataframes 
[here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FQIM4UF) 
in pickle form, the dataframes can 
be loaded into memory with the Python commands

```
> import pandas as pd 
> df_agg = pd.read_pickle('path/to/df_agg_{state}{year}_dataverse.pkl')
```

## Data Dictionary

Below is a description of the fields included in the `df_agg` dataframe:

### Core Identification Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Surname of voter, converted to lowercase |
| `county` | string | County code or identifier where the voter is registered |
| `county_long` | string | Full county name (only in some state datasets, e.g., Florida) |
| `region` | string | Geographic region within the state (when available) |
| `rural` | boolean | Flag indicating if the county is classified as rural (when available) |
| `in_cen_surs` | boolean | Flag indicating if the surname appears in the Census surname list |
| `vf_tot` | integer | Total number of voters with the given surname in the county |

### Race/Ethnicity Categories

The package uses six racial/ethnic categories throughout. For each prediction type, there will be six fields, one for each category:

- `nh_white`: Non-Hispanic White
- `nh_black`: Non-Hispanic Black
- `nh_api`: Non-Hispanic Asian and Pacific Islander
- `nh_aian`: Non-Hispanic American Indian and Alaska Native
- `hispanic`: Hispanic (any race)
- `other`: Other races and multi-racial

### Field Patterns

For each prediction type or data source, there is a set of six fields (one for each race/ethnicity category) following consistent naming patterns:

- Voter File Race Counts (labeled files only), pattern `vf_{race}`: Count of voters of each race/ethnicity for the given surname-county pair.

- Census Data Probabilities:
  - Pattern `cen_r_given_geo_{race}`: Census-based probability of being of a race/ethnicity given the county.
  - Pattern `cen18_r_given_geo_{race}`: Census-based probability for the 18+ population given the county.
  - Pattern `cen_r_given_sur_{race}`: Census-based probability given the surname.

- CPS Voter-Adjusted Probabilities:
  - Pattern `cps_bayes_r_given_geo_{race}`: CPS-adjusted probability given the county for voters.
  - Pattern `cps_bayes_r_given_sur_{race}`: CPS-adjusted probability given the surname for voters.

- Prediction Fields:
  - Pattern `bisg_cen_county_{race}`: BISG prediction using Census data for the full US population.
  - Pattern `bisg_bayes_{race}`: Voter-BISG prediction adjusted for registered voters using CPS data.
  - Pattern `rake_{race}`: Raking prediction matching the CPS voter race distribution margin.

- Additional Voter File Fields (labeled files only):
  - Pattern `vf_r_given_geo_{race}`: Voter file-based probability given the county.
  - Pattern `vf_r_given_sur_{race}`: Voter file-based probability given the surname.
  - Pattern `vf_bisg_{race}`: Voter file BISG prediction.
  - Pattern `vf_bayes_opt_{race}`: Actual probability (ground truth from labeled voter files).
  - Pattern `vf_r_geo_tot_{race}`: Raw count of voters of each race in a county.
  - Pattern `vf_r_sur_tot_{race}`: Raw count of voters of each race with a particular surname.
  - Pattern `vf_bisg_tot_{race}`: Unnormalized BISG scores.
  - Pattern `vf_rake3_count_{race}`: Counts from three-way raking procedure (when implemented).

## Unit test
A unit test can be run from the home directory of this repository via the command 
```
$ python3 ./test.py
``` 
that checks that if data is generated according to the BISG assumption, then BISG is exact, 
and when data is not generated with the BISG assumption, raking predictions still have the 
correct race/ethnicity margin.

## Reproducing figures and tables
The figures and tables of [the paper](https://arxiv.org/abs/2304.09126) from North Carolina can be reproduced
by running the script
```
$ python3 ./download_data_and_run.py
``` 
This script i) downloads available voter file, census, and cps data, ii) builds dataframes 
of predictions, iii) constructs figures and latex tables of the paper. Warning: voter files 
downloaded by this script can be very large. For example, the 2020 North Carolina voter 
file is nearly 15GB. 

In order to generate figures and tables for Florida, the Florida voter file must be 
saved to a location specified in the above script. The Florida voter file is 
sent by disk to those who request it. 
The North Carolina voter files are available for download online and are
downloaded using the above script.  
 

## Python files

`{state}_vf.py`: reads in a voter file from a particular state

`raking_bisg.py` : the main statistical codes, including building `df_agg` which involves computing 
raking and BISG predictions

`fields.py` : defines column names (mainly predictions) of `df_agg`

`process_census.py` : reads in census data

`process_cps.py` : reads in data from the voter supplement of cps

`plotting.py` : plot figures for paper and construct LaTeX tables

`test.py` : unit test

`download_data_and_run.py` : script that downloads available data and reproduces figures and tables for the paper
