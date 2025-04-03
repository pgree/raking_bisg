# Race/ethnicity predictions via raking and BISG

This repository includes all software for reproducing the figures and tables of the paper 
"A calibrated BISG for inferring race from surname and geolocation" by Philip Greengard and Andrew Gelman:

Philip Greengard, Andrew Gelman, A calibrated BISG for inferring race from surname and geolocation, Journal of the Royal Statistical Society Series A: Statistics in Society, 2025;, qnaf003, https://doi.org/10.1093/jrsssa/qnaf003

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

The columns of the dataframe are explained in the file `./fields.py`.

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

`download_data_and_run.py` : script that 
