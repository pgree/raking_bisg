"""
this module contains a unittest for the raking and bisg algorithms of this repository.
the test check that i) if data is generated according to the BISG assumption, then BISG
and raking are both exact, and ii) if data is not generated with the BISG assumption,
raking predictions have the correct race/ethnicity margin and BISG predictions do not.
"""
import pandas as pd
import numpy as np
import fields
from itertools import product
from raking_bisg import append_voter_bisg, append_raking_preds, subpopulation_preds
import unittest

# for the unit test, use the counties in the northwest region of florida
COUNTIES = [
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
]
# ...and some of the most common names in florida
NAMES = [
    "rodriguez",
    "williams",
    "johnson",
    "diaz",
    "thompson",
    "taylor",
    "wright",
    "davis",
    "wilson",
    "king",
    "hill",
    "charles",
    "jones",
    "green",
    "marin",
    "turner",
    "adams",
    "jenkins",
    "fernandez",
    "kelly",
]


class TestRakingBISG(unittest.TestCase):
    def conditionally_ind_data(self):
        """
        generate tabular data across surname, geolocation, race, voter status that satisfies
        the conditional independence assumptions of bisg. that is,

           x_{rgsv} \propto \tau_{gr} \tau_{sr} \tau_{rv}

        """
        nnames = len(NAMES)
        ncounties = len(COUNTIES)
        nraces = 6
        nvoter = 2
        tabv = np.random.randint(0, 10, size=(nnames, ncounties, nraces, nvoter))
        gr = np.sum(tabv, axis=(0, 3))
        sr = np.sum(tabv, axis=(1, 3))
        vr = np.sum(tabv, axis=(0, 1))

        # generate conditionally independent tabular data
        tab = np.ones(shape=(nnames, ncounties, nraces, nvoter))
        tab = tab * gr[np.newaxis, :, :, np.newaxis]
        tab = tab * sr[:, np.newaxis, :, np.newaxis]
        tab = tab * vr[np.newaxis, np.newaxis, :, :]
        tab = tab / 100000
        return tab

    def random_data(self):
        """
        generate iid random data for each entry of x_{sgrv}
        """
        nnames = len(NAMES)
        ncounties = len(COUNTIES)
        nraces = 6
        nvoter = 2
        tab = np.random.pareto(a=2, size=(nnames, ncounties, nraces, nvoter))
        return tab

    def test_cond_ind_preds(self):
        """
        check that for data that satisfies the conditional independence assumption of bisg,
        the appropriate marginals of the predictions are correct
        """
        # construct data
        tab = self.conditionally_ind_data()
        nnames, ncounties, nraces, nvoter = tab.shape

        # construct df_agg, this is equivalent to creating df_vf and runnning df_agg_init
        df = pd.DataFrame(list(product(NAMES, COUNTIES)))
        df.columns = ["name", "county"]
        name_county_voters = np.sum(tab[:, :, :, 0], axis=2).reshape((-1, 1))
        df["vf_tot"] = name_county_voters
        name_county_race_voters = tab[:, :, :, 0].reshape((ncounties * nnames, nraces))
        df[fields.VF_RACES] = name_county_race_voters
        df_agg = df.copy()

        # race distribution for voters
        df_cps_reg_voters = np.sum(tab[:, :, :, 0], axis=(0, 1)).reshape((1, -1))
        df_cps_reg_voters = pd.DataFrame(data=df_cps_reg_voters, columns=fields.RACES)
        df_cps_reg_voters = df_cps_reg_voters.div(df_cps_reg_voters.sum().sum())

        # race by county
        df_cen_counties = np.sum(tab[:, :, :, :], axis=(0, 3))
        df_cen_counties = pd.DataFrame(data=df_cen_counties, columns=fields.RACES18)
        df_cen_counties["county"] = COUNTIES
        df_cen_counties[fields.PROB18_RACES] = df_cen_counties[fields.RACES18].div(
            df_cen_counties[fields.RACES18].sum(axis=1), axis=0
        )
        df_cen_counties["total"] = df_cen_counties[fields.RACES18].sum(axis=1)

        # race by surname
        df_cen_surs = np.sum(tab[:, :, :, :], axis=(1, 3))
        df_cen_surs = pd.DataFrame(data=df_cen_surs, columns=fields.CEN_R_GIVEN_SUR_COLS)
        df_cen_surs["count"] = df_cen_surs[fields.CEN_R_GIVEN_SUR_COLS].sum(axis=1)
        df_cen_surs[fields.CEN_R_GIVEN_SUR_COLS] = df_cen_surs[fields.CEN_R_GIVEN_SUR_COLS].div(
            df_cen_surs["count"], axis=0
        )
        df_cen_surs["name"] = NAMES

        # voter bisg
        df_agg = append_voter_bisg(
            df_agg, df_cps_reg_voters, df_cen_counties, df_cen_surs
        )
        df_agg = append_raking_preds(df_agg, df_cps_reg_voters)

        # make raking and bisg predictions
        preds1, true_pops1, true_probs1 = subpopulation_preds(
            df_agg, fields.RAKE_COLS, region="county", calib_map=None
        )
        preds2, true_pops2, true_probs2 = subpopulation_preds(
            df_agg, fields.BISG_BAYES_COLS, region="county", calib_map=None
        )
        np.testing.assert_allclose(preds1, true_pops1)
        np.testing.assert_allclose(preds2, true_pops2)

    def test_random_preds(self):
        """
        check that for data that does not satisfy the conditional independence assumption of bisg,
        the marginals of the raking predictions coincide with the exact marginals
        """
        # construct data
        tab = self.random_data()
        nnames, ncounties, nraces, nvoter = tab.shape

        # construct df_agg, this is equivalent to creating df_vf and runnning df_agg_init
        df = pd.DataFrame(list(product(NAMES, COUNTIES)))
        df.columns = ["name", "county"]
        name_county_voters = np.sum(tab[:, :, :, 0], axis=2).reshape((-1, 1))
        df["vf_tot"] = name_county_voters
        name_county_race_voters = tab[:, :, :, 0].reshape((ncounties * nnames, nraces))
        df[fields.VF_RACES] = name_county_race_voters
        df_agg = df.copy()

        # race distribution for voters
        df_cps_reg_voters = np.sum(tab[:, :, :, 0], axis=(0, 1)).reshape((1, -1))
        df_cps_reg_voters = pd.DataFrame(data=df_cps_reg_voters, columns=fields.RACES)
        df_cps_reg_voters = df_cps_reg_voters.div(df_cps_reg_voters.sum().sum())

        # race by county
        df_cen_counties = np.sum(tab[:, :, :, :], axis=(0, 3))
        df_cen_counties = pd.DataFrame(data=df_cen_counties, columns=fields.RACES18)
        df_cen_counties["county"] = COUNTIES
        df_cen_counties[fields.PROB18_RACES] = df_cen_counties[fields.RACES18].div(
            df_cen_counties[fields.RACES18].sum(axis=1), axis=0
        )
        df_cen_counties["total"] = df_cen_counties[fields.RACES18].sum(axis=1)

        # race by surname
        df_cen_surs = np.sum(tab[:, :, :, :], axis=(1, 3))
        df_cen_surs = pd.DataFrame(data=df_cen_surs, columns=fields.CEN_R_GIVEN_SUR_COLS)
        df_cen_surs["count"] = df_cen_surs[fields.CEN_R_GIVEN_SUR_COLS].sum(axis=1)
        df_cen_surs[fields.CEN_R_GIVEN_SUR_COLS] = df_cen_surs[fields.CEN_R_GIVEN_SUR_COLS].div(
            df_cen_surs["count"], axis=0
        )
        df_cen_surs["name"] = NAMES

        # voter bisg
        df_agg = append_voter_bisg(
            df_agg, df_cps_reg_voters, df_cen_counties, df_cen_surs
        )
        df_agg = append_raking_preds(df_agg, df_cps_reg_voters)

        # make raking and bisg predictions
        preds1, true_pops1, true_probs1 = subpopulation_preds(
            df_agg, fields.RAKE_COLS, region="county", calib_map=None
        )
        preds2, true_pops2, true_probs2 = subpopulation_preds(
            df_agg, fields.BISG_BAYES_COLS, region="county", calib_map=None
        )
        # assert that raking recovers the correct race margins
        np.testing.assert_allclose(np.sum(preds1, axis=0), np.sum(true_pops1, axis=0))
        # ...and bisg doesn't
        np.testing.assert_(
            np.linalg.norm(np.sum(preds2, axis=0) - np.sum(true_pops2, axis=0)) > 0.01
        )


if __name__ == "__main__":
    unittest.main()
