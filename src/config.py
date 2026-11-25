
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# RAW DATASET PATHS
# ============================================================

RAW_DIR = PROJECT_ROOT

ARTICLES_FILE = RAW_DIR / "articles.csv"

CUSTOMERS_FILE = RAW_DIR / "customers.csv"

TRANSACTIONS_FILE = RAW_DIR / "transactions_train.csv"


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

PROCESSED_DIR = DATA_DIR / "processed"

TRANSACTIONS_OUTPUT = (
    PROCESSED_DIR / "transactions"
)

CUSTOMER_FEATURES_OUTPUT = (
    PROCESSED_DIR / "customer_features"
)

ARTICLE_FEATURES_OUTPUT = (
    PROCESSED_DIR / "article_features"
)

CONTEXT_OUTPUT = (
    PROCESSED_DIR / "contextual_features"
)

TRAIN_TEST_OUTPUT = (
    PROCESSED_DIR / "train_test"
)

VOCAB_DIR = DATA_DIR / "vocabularies"


# ============================================================
# OUTPUT AND LOG DIRECTORIES
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

LOG_DIR = PROJECT_ROOT / "logs"


# ============================================================
# SPARK CONFIGURATION
# ============================================================

APP_NAME = "HM_Personalized_Fashion_Recommendations"

MASTER = "local[*]"

SHUFFLE_PARTITIONS = 8

TRAIN_RATIO = 0.80


# ============================================================
# HADOOP CONFIGURATION (WINDOWS)
# ============================================================

HADOOP_HOME = Path("C:/hadoop")

WINUTILS_PATH = HADOOP_HOME / "bin" / "winutils.exe"


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

REQUIRED_DIRECTORIES = [

    DATA_DIR,

    PROCESSED_DIR,

    TRANSACTIONS_OUTPUT,

    CUSTOMER_FEATURES_OUTPUT,

    ARTICLE_FEATURES_OUTPUT,

    CONTEXT_OUTPUT,

    TRAIN_TEST_OUTPUT,

    VOCAB_DIR,

    OUTPUT_DIR,

    LOG_DIR
]


for directory in REQUIRED_DIRECTORIES:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# VALIDATE RAW DATASET FILES
# ============================================================

DATASET_FILES = {

    "Articles": ARTICLES_FILE,

    "Customers": CUSTOMERS_FILE,

    "Transactions": TRANSACTIONS_FILE
}


def validate_dataset_files():

    missing_files = []

    for name, file_path in DATASET_FILES.items():

        if not file_path.exists():

            missing_files.append(
                f"{name}: {file_path}"
            )

    if missing_files:

        raise FileNotFoundError(
            "Missing H&M dataset files:\n"
            + "\n".join(missing_files)
        )

    print("All H&M dataset files found successfully.")


# ============================================================
# PROJECT CONFIGURATION SUMMARY
# ============================================================

def display_config():

    print("\n" + "=" * 60)

    print("H&M PROJECT CONFIGURATION")

    print("=" * 60)

    print("Project root:", PROJECT_ROOT)

    print("Python project:", PROJECT_ROOT.name)

    print("Articles:", ARTICLES_FILE.exists())

    print("Customers:", CUSTOMERS_FILE.exists())

    print("Transactions:", TRANSACTIONS_FILE.exists())

    print("Hadoop home:", HADOOP_HOME)

    print("Winutils exists:", WINUTILS_PATH.exists())

    print("Spark application:", APP_NAME)

    print("Spark master:", MASTER)

    print("Shuffle partitions:", SHUFFLE_PARTITIONS)

    print("=" * 60)
