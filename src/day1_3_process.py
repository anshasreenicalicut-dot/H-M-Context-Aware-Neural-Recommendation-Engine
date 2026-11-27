
import json
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType
)

from .config import (
    TRANSACTIONS_CSV,
    CUSTOMERS_CSV,
    ARTICLES_CSV,
    TX_OUT,
    COLD_OUT,
    OUTPUTS,
    CSV_OPTIONS
)

from .spark_session import get_spark


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def validate_file(file_path, file_name):

    path = Path(file_path)

    if not path.exists():

        raise FileNotFoundError(
            f"{file_name} not found: {path}"
        )

    if not path.is_file():

        raise ValueError(
            f"{file_name} is not a valid file: {path}"
        )

    print(f"{file_name} found: {path}")


def create_directory(directory):

    Path(directory).mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def run():

    spark = get_spark("Week1_Days1to3")

    try:

        print("\n" + "=" * 65)
        print("WEEK 1: DAYS 1-3")
        print("DISTRIBUTED DATA PROCESSING")
        print("=" * 65)

        # --------------------------------------------------
        # STEP 1: VALIDATE DATASET FILES
        # --------------------------------------------------

        print("\nSTEP 1: Validating dataset files...")

        validate_file(
            TRANSACTIONS_CSV,
            "Transactions dataset"
        )

        validate_file(
            CUSTOMERS_CSV,
            "Customers dataset"
        )

        validate_file(
            ARTICLES_CSV,
            "Articles dataset"
        )

        # --------------------------------------------------
        # STEP 2: DEFINE TRANSACTION SCHEMA
        # --------------------------------------------------

        print("\nSTEP 2: Defining transaction schema...")

        schema = StructType([

            StructField(
                "t_dat",
                StringType(),
                True
            ),

            StructField(
                "customer_id",
                StringType(),
                True
            ),

            StructField(
                "article_id",
                StringType(),
                True
            ),

            StructField(
                "price",
                DoubleType(),
                True
            ),

            StructField(
                "sales_channel_id",
                IntegerType(),
                True
            )

        ])

        # --------------------------------------------------
        # STEP 3: LOAD TRANSACTIONS
        # --------------------------------------------------

        print("\nSTEP 3: Loading transaction dataset...")

        raw = (
            spark.read
            .options(**CSV_OPTIONS)
            .schema(schema)
            .csv(str(TRANSACTIONS_CSV))
        )

        raw_count = raw.count()

        print(
            "Total raw transactions:",
            f"{raw_count:,}"
        )

        if raw_count == 0:

            raise ValueError(
                "Transaction dataset is empty."
            )

        # --------------------------------------------------
        # STEP 4: DATA CLEANING
        # --------------------------------------------------

        print("\nSTEP 4: Cleaning transaction data...")

        df = (
            raw

            .withColumn(
                "customer_id",
                F.trim(F.col("customer_id"))
            )

            .withColumn(
                "article_id",
                F.trim(F.col("article_id"))
            )

            .withColumn(
                "t_dat",
                F.to_date(
                    F.trim(F.col("t_dat")),
                    "yyyy-MM-dd"
                )
            )
        )

        # Valid interaction records require:
        # customer_id, article_id and transaction date.

        valid = df.filter(

            F.col("customer_id").isNotNull() &

            (F.col("customer_id") != "") &

            F.col("article_id").isNotNull() &

            (F.col("article_id") != "") &

            F.col("t_dat").isNotNull()

        )

        valid_count = valid.count()

        invalid_count = raw_count - valid_count

        print(
            "Valid transactions:",
            f"{valid_count:,}"
        )

        print(
            "Invalid transactions:",
            f"{invalid_count:,}"
        )

        # --------------------------------------------------
        # STEP 5: IDENTIFY MISSING VALUES
        # --------------------------------------------------

        print("\nSTEP 5: Checking missing values...")

        missing_stats = valid.agg(

            F.sum(
                F.when(
                    F.col("price").isNull() |
                    F.isnan(F.col("price")),
                    1
                ).otherwise(0)
            ).alias("missing_price"),

            F.sum(
                F.when(
                    F.col("sales_channel_id").isNull(),
                    1
                ).otherwise(0)
            ).alias("missing_sales_channel")

        ).first()

        missing_price_count = (
            missing_stats["missing_price"] or 0
        )

        missing_channel_count = (
            missing_stats["missing_sales_channel"] or 0
        )

        print(
            "Missing prices:",
            f"{missing_price_count:,}"
        )

        print(
            "Missing sales channels:",
            f"{missing_channel_count:,}"
        )

        # --------------------------------------------------
        # STEP 6: REMOVE EXACT DUPLICATES
        # --------------------------------------------------

        print("\nSTEP 6: Removing duplicate records...")

        dedup = valid.dropDuplicates([

            "t_dat",
            "customer_id",
            "article_id",
            "price",
            "sales_channel_id"

        ])

        dedup_count = dedup.count()

        duplicate_count = valid_count - dedup_count

        print(
            "Duplicates removed:",
            f"{duplicate_count:,}"
        )

        print(
            "Transactions after deduplication:",
            f"{dedup_count:,}"
        )

        # --------------------------------------------------
        # STEP 7: HANDLE MISSING PRICE VALUES
        # --------------------------------------------------

        print("\nSTEP 7: Calculating median price...")

        valid_prices = dedup.filter(

            F.col("price").isNotNull() &

            (~F.isnan(F.col("price"))) &

            (F.col("price") >= 0)

        )

        price_values = valid_prices.approxQuantile(

            "price",

            [0.5],

            0.001

        )

        median_price = (

            float(price_values[0])

            if price_values

            else 0.0

        )

        print(
            "Median price:",
            median_price
        )

        # --------------------------------------------------
        # STEP 8: HANDLE MISSING SALES CHANNEL
        # --------------------------------------------------

        print("\nSTEP 8: Calculating sales channel mode...")

        mode_row = (

            dedup

            .filter(
                F.col("sales_channel_id").isNotNull()
            )

            .groupBy("sales_channel_id")

            .count()

            .orderBy(
                F.desc("count"),
                F.asc("sales_channel_id")
            )

            .first()

        )

        mode_channel = (

            int(mode_row["sales_channel_id"])

            if mode_row is not None

            else -1

        )

        print(
            "Most frequent sales channel:",
            mode_channel
        )

        # --------------------------------------------------
        # STEP 9: APPLY MISSING VALUE TREATMENT
        # --------------------------------------------------

        print("\nSTEP 9: Applying missing value treatment...")

        clean = (

            dedup

            .withColumn(

                "price",

                F.when(

                    F.col("price").isNull() |

                    F.isnan(F.col("price")) |

                    (F.col("price") < 0),

                    F.lit(median_price)

                )

                .otherwise(F.col("price"))

                .cast("double")

            )

            .withColumn(

                "sales_channel_id",

                F.coalesce(

                    F.col("sales_channel_id"),

                    F.lit(mode_channel)

                )

            )

        )

        # --------------------------------------------------
        # STEP 10: CONTEXTUAL FEATURE ENGINEERING
        # --------------------------------------------------

        print("\nSTEP 10: Creating contextual features...")

        clean = (

            clean

            .withColumn(

                "transaction_year",

                F.year("t_dat")

            )

            .withColumn(

                "transaction_month",

                F.month("t_dat")

            )

            .withColumn(

                "transaction_day",

                F.dayofmonth("t_dat")

            )

            .withColumn(

                "day_of_week",

                F.dayofweek("t_dat")

            )

            .withColumn(

                "day_of_year",

                F.dayofyear("t_dat")

            )

            .withColumn(

                "transaction_quarter",

                F.quarter("t_dat")

            )

            .withColumn(

                "is_weekend",

                F.when(

                    F.dayofweek("t_dat").isin(1, 7),

                    1

                ).otherwise(0)

            )

        )

        print(
            "Contextual date features created successfully."
        )

        # --------------------------------------------------
        # STEP 11: SAVE CLEANED TRANSACTIONS
        # --------------------------------------------------

        print("\nSTEP 11: Saving cleaned transactions...")

        create_directory(TX_OUT)

        (

            clean.write

            .mode("overwrite")

            .partitionBy(

                "transaction_year",

                "transaction_month"

            )

            .parquet(str(TX_OUT))

        )

        print(
            "Cleaned transactions saved successfully!"
        )

        print(
            "Output location:",
            TX_OUT
        )

        # --------------------------------------------------
        # STEP 12: LOAD CUSTOMER MASTER DATA
        # --------------------------------------------------

        print("\nSTEP 12: Loading customer master data...")

        customers = (

            spark.read

            .options(**CSV_OPTIONS)

            .option("inferSchema", "false")

            .csv(str(CUSTOMERS_CSV))

            .select(

                F.trim(
                    F.col("customer_id")
                ).alias("customer_id")

            )

            .filter(

                F.col("customer_id").isNotNull() &

                (F.col("customer_id") != "")

            )

            .dropDuplicates(["customer_id"])

        )

        # --------------------------------------------------
        # STEP 13: LOAD ARTICLE MASTER DATA
        # --------------------------------------------------

        print("\nSTEP 13: Loading article master data...")

        articles = (

            spark.read

            .options(**CSV_OPTIONS)

            .option("inferSchema", "false")

            .csv(str(ARTICLES_CSV))

            .select(

                F.trim(
                    F.col("article_id")
                ).alias("article_id")

            )

            .filter(

                F.col("article_id").isNotNull() &

                (F.col("article_id") != "")

            )

            .dropDuplicates(["article_id"])

        )

        customer_count = customers.count()

        article_count = articles.count()

        print(
            "Total unique customers:",
            f"{customer_count:,}"
        )

        print(
            "Total unique articles:",
            f"{article_count:,}"
        )

        # --------------------------------------------------
        # STEP 14: COLD-START ANALYSIS
        # --------------------------------------------------

        print("\nSTEP 14: Performing cold-start analysis...")

        active_users = (

            clean

            .select("customer_id")

            .distinct()

        )

        active_items = (

            clean

            .select("article_id")

            .distinct()

        )

        # Customers without observed transaction history.

        cold_users = customers.join(

            active_users,

            on="customer_id",

            how="left_anti"

        )

        # Articles without observed transaction history.

        cold_items = articles.join(

            active_items,

            on="article_id",

            how="left_anti"

        )

        # --------------------------------------------------
        # STEP 15: SAVE COLD-START DATA
        # --------------------------------------------------

        print("\nSTEP 15: Saving cold-start datasets...")

        create_directory(COLD_OUT)

        cold_users_path = (
            Path(COLD_OUT) /
            "customers_without_transactions"
        )

        cold_items_path = (
            Path(COLD_OUT) /
            "articles_without_transactions"
        )

        (

            cold_users.write

            .mode("overwrite")

            .parquet(str(cold_users_path))

        )

        (

            cold_items.write

            .mode("overwrite")

            .parquet(str(cold_items_path))

        )

        cold_user_count = cold_users.count()

        cold_item_count = cold_items.count()

        print(
            "Customers without transactions:",
            f"{cold_user_count:,}"
        )

        print(
            "Articles without transactions:",
            f"{cold_item_count:,}"
        )

        # --------------------------------------------------
        # STEP 16: GENERATE PROCESSING SUMMARY
        # --------------------------------------------------

        print("\nSTEP 16: Generating processing summary...")

        summary = {

            "project":
                "Context-Aware Neural Recommendation Engine",

            "week":
                1,

            "stage":
                "Days 1-3: Distributed Data Processing",

            "raw_transaction_rows":
                raw_count,

            "valid_transaction_rows":
                valid_count,

            "invalid_transaction_rows":
                invalid_count,

            "missing_price_values":
                missing_price_count,

            "missing_sales_channel_values":
                missing_channel_count,

            "duplicate_transactions_removed":
                duplicate_count,

            "cleaned_transaction_rows":
                dedup_count,

            "median_price_imputation":
                median_price,

            "sales_channel_mode_imputation":
                mode_channel,

            "master_customers":
                customer_count,

            "master_articles":
                article_count,

            "customers_without_transactions":
                cold_user_count,

            "articles_without_transactions":
                cold_item_count,

            "transaction_output":
                str(TX_OUT),

            "cold_start_output":
                str(COLD_OUT)

        }

        create_directory(OUTPUTS)

        summary_file = (
            Path(OUTPUTS) /
            "day1_3_summary.json"
        )

        with open(

            summary_file,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                summary,

                file,

                indent=4

            )

        # --------------------------------------------------
        # STEP 17: DISPLAY FINAL RESULTS
        # --------------------------------------------------

        print("\n" + "=" * 65)

        print("WEEK 1: DAYS 1-3 COMPLETED SUCCESSFULLY")

        print("=" * 65)

        print(
            json.dumps(
                summary,
                indent=4
            )
        )

        print(
            "\nSummary saved:",
            summary_file
        )

        print("\nSample cleaned transactions:")

        clean.show(
            5,
            truncate=False
        )

        print("\nSample cold-start customers:")

        cold_users.show(
            5,
            truncate=False
        )

        print("\nSample cold-start articles:")

        cold_items.show(
            5,
            truncate=False
        )

    except Exception as error:

        print("\n" + "=" * 65)

        print("ERROR IN WEEK 1 DAYS 1-3")

        print("=" * 65)

        print(type(error).__name__, ":", str(error))

        raise

    finally:

        print("\nStopping Spark session...")

        spark.stop()

        print("Spark session stopped.")


# ==========================================================
# EXECUTION
# ==========================================================

if __name__ == "__main__":

    run()
