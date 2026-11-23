"""Project settings: paths, sample windows, column lists.

Everything that defines what the analysis is run on lives here, so the design
choices can be read in one place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "accepted_2007_to_2018Q4.csv"
EXTRACT = DATA_DIR / "loans_36m_2007_2015.parquet"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

SEED = 42

# --- Sample design ---------------------------------------------------------
# Only 36-month loans issued up to December 2015 are used. The data snapshot
# ends in early 2019, so every one of these loans has reached maturity and its
# outcome is known. Including younger loans would bias default rates downward.
TERM = "36 months"
TERM_MONTHS = 36
LAST_ISSUE_DATE = "2015-12-31"

TRAIN = ("2012-01-01", "2013-12-31")   # fit the models
CALIB = ("2014-01-01", "2014-12-31")   # recalibrate PD levels
TEST = ("2015-01-01", "2015-12-31")    # evaluate; this is "the 2015 book"

# --- Point-in-time robustness checks ----------------------------------------
# The main design trains on the final outcome of 2012-13 loans, which was not
# fully known until 2016. Two checks ask what could have been built with only
# the information available on POINT_IN_TIME_DATE, the day before the first
# test loan was issued.
POINT_IN_TIME_DATE = "2014-12-31"
# Check 1 (matured loans only): 36-month loans issued by December 2011 had all
# reached maturity by the point-in-time date.
POINT_IN_TIME_TRAIN = ("2007-07-01", "2011-12-31")
# Check 2 (early performance): the same 2012-13 loans as the main model, but
# labelled only with what had been observed by the point-in-time date. A loan
# counts as an early default if it was 120+ days past due within 12 months of
# issue (and was later charged off).
EARLY_WINDOW_MONTHS = 12
DELINQUENCY_LAG_MONTHS = 4

# Vintages used to estimate asset correlation (nothing from the test year).
RHO_WINDOW = ("2007-07-01", "2014-12-31")

# --- Columns ---------------------------------------------------------------
ID_COLS = ["id", "issue_d", "term", "application_type", "loan_status", "grade"]

# Known when the loan is funded. Only these fields may be used by a model.
CORE_NUMERIC = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
    "fico_range_low", "inq_last_6mths", "mths_since_last_delinq",
    "mths_since_last_record", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "collections_12_mths_ex_med", "acc_now_delinq",
    "pub_rec_bankruptcies", "tax_liens", "chargeoff_within_12_mths",
    "delinq_amnt",
]
# Extra credit-bureau attributes that Lending Club began publishing in 2012.
BUREAU_NUMERIC = [
    "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim", "acc_open_past_24mths",
    "avg_cur_bal", "bc_open_to_buy", "bc_util", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl",
    "mort_acc", "mths_since_recent_bc", "mths_since_recent_inq",
    "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl",
    "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts",
    "num_rev_tl_bal_gt_0", "num_sats", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75",
    "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit", "mths_since_last_major_derog",
    "mths_since_recent_bc_dlq", "mths_since_recent_revol_delinq",
]
RAW_TEXT_FEATURES = ["sub_grade", "emp_length", "earliest_cr_line"]
CATEGORICAL = ["home_ownership", "verification_status", "purpose"]

# Recorded after funding. Used to build the target and to measure realised
# losses and profit. Never used as model inputs.
OUTCOME_COLS = [
    "funded_amnt", "total_pymnt", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d",
]

RAW_USECOLS = (
    ID_COLS + CORE_NUMERIC + BUREAU_NUMERIC + RAW_TEXT_FEATURES + CATEGORICAL
    + OUTCOME_COLS
)

# Built at load time from origination fields only.
ENGINEERED = [
    "sub_grade_num", "emp_length_years", "credit_history_years",
    "loan_to_income", "payment_to_income", "revol_bal_to_income",
]

# Fields that carry Lending Club's own risk assessment (the grade, and the
# interest rate that follows from it; the instalment is a function of the
# rate). Kept as a separate list so the model can be run with and without them.
LENDER_SCORE = ["sub_grade_num", "int_rate", "installment", "payment_to_income"]

NUMERIC_FEATURES = CORE_NUMERIC + BUREAU_NUMERIC + ENGINEERED
FEATURES = NUMERIC_FEATURES + CATEGORICAL
# Subset that exists for every vintage back to 2007 (used by the strict check).
CORE_FEATURES = CORE_NUMERIC + ENGINEERED + CATEGORICAL

# Derived outcome fields created in data.clean(). Listed so tests can assert
# that none of them ever reaches a model.
DERIVED_OUTCOMES = [
    "default", "loss_amount", "severity", "net_pnl", "realized_return",
    "months_to_last_payment", "early_default",
]

# Deliberately excluded from FEATURES: addr_state and zip_code (geography can
# act as a proxy for protected characteristics), free-text fields, and
# everything in OUTCOME_COLS.
GRADES = list("ABCDEFG")
