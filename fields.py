"""
this file contains fields for the various predictions that are used throughout this repository.
df_agg, the dataframe of predictions for (surname, geolocation) pairs that is constructed by
the codes of this repo contains all the prediction fields in this file in addition to
the fields 'vf_tot' (total number of appearances of (surname, geolocation) pair in the state)
and 'in_cen_surs' (a 1 indicates the surname is in the census surnames list and 0 means it's not).
"""

# subpopulations by increasing population size
RACES = ["nh_aian", "nh_api", "nh_black", "hispanic", "nh_white", "other"]
RACES_TO_PRETTY = {
    "nh_white": "White",
    "nh_black": "Black",
    "nh_api": "API",
    "nh_aian": "AIAN",
    "hispanic": "Hispanic",
    "other": "Other",
}
PRETTY_PRINT = [RACES_TO_PRETTY[race] for race in RACES]
PROB_COLS = [f"prob_{race}" for race in RACES]
PROB18_RACES = [f"prob18_{race}" for race in RACES]
RACES18 = [f"{race}18" for race in RACES]

# voter file-only columns. these columns only appear in Florida and North Carolina,
# states with labeled race
VF_RACES = [f"vf_{race}" for race in RACES]
VF_R_GIVEN_GEO_COLS = [f"vf_r_given_geo_{race}" for race in RACES]
VF_R_GIVEN_SUR_COLS = [f"vf_r_given_sur_{race}" for race in RACES]
VF_R_SUR_TOT_COLS = [f'vf_r_sur_tot_{race}' for race in RACES]
VF_R_GEO_TOT_COLS = [f'vf_r_geo_tot_{race}' for race in RACES]
VF_BISG_TOT = [f'vf_bisg_tot_{race}' for race in RACES]
VF_BAYES_OPT_COLS = [f"vf_bayes_opt_{race}" for race in RACES]
VF_RAKE2_COLS = [f"vf_rake_{race}" for race in RACES]
VF_BISG = [f"vf_bisg_{race}" for race in RACES]
VF_RAKE3_COLS = [f"vf_rake3_{race}" for race in RACES]
VF_RAKE3_COUNTS = [f"vf_rake3_count_{race}" for race in RACES]
VF_R_SUR_OTHER = [f'vf_r_sur_other_{race}' for race in RACES]

# census data
CEN18_R_GIVEN_GEO_COLS = [f"cen18_r_given_geo_{race}" for race in RACES]
CEN_R_GIVEN_GEO_COLS = [f"cen_r_given_geo_{race}" for race in RACES]
CEN_R_GIVEN_SUR_COLS = [f"cen_r_given_sur_{race}" for race in RACES]

# cps-calibrated data
CPS_BAYES_R_GIVEN_GEO_COLS = [f"cps_bayes_r_given_geo_{race}" for race in RACES]
CPS_BAYES_R_GIVEN_SUR_COLS = [f"cps_bayes_r_given_sur_{race}" for race in RACES]

# predictions
BISG_CEN_COUNTY_COLS = [f"bisg_cen_county_{race}" for race in RACES]
BISG_BAYES_COLS = [f"bisg_bayes_{race}" for race in RACES]
RAKE_COLS = [f"rake_{race}" for race in RACES]
