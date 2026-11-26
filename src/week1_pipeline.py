
import json
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import Row

from src.config import (
    ARTICLES_FILE,
    CUSTOMERS_FILE,
    TRANSACTIONS_FILE,
    TRANSACTIONS_OUTPUT,
    CUSTOMER_FEATURES_OUTPUT,
    ARTICLE_FEATURES_OUTPUT,
    CONTEXT_OUTPUT,
    TRAIN_TEST_OUTPUT,
    VOCAB_DIR,
    OUTPUT_DIR
)


# ============================================================
# HELPER: SAVE JSON REPORT
# ============================================================

def save_json_report(data, filename):

    output_file = OUTPUT_DIR / filename

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)

    print(f"Report saved: {output_file}")


# ============================================================
# DAY 1: LOAD COMPLETE H&M DATASET USING PYSPARK
# ============================================================

def load_datasets(spark):

    print("\n" + "=" * 60)
    print("DAY 1: LOADING H&M DATASET")
    print("=" * 60)

    required_files = [
        ARTICLES_FILE,
        CUSTOMERS_FILE,
        TRANSACTIONS_FILE
    ]

    for file_path in required_files:

        if not Path(file_path).exists():

            raise FileNotFoundError(
                f"Dataset file not found: {file_path}"
            )

    # Load articles

    articles = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .option("mode", "FAILFAST")
        .csv(str(ARTICLES_FILE))
    )

    # Load customers

    customers = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .option("mode", "FAILFAST")
        .csv(str(CUSTOMERS_FILE))
    )

    # Load transactions

    transactions = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .option("mode", "FAILFAST")
        .csv(str(TRANSACTIONS_FILE))
    )

    # Validate required columns

    required_transaction_columns = [
        "t_dat",
        "customer_id",
        "article_id",
        "price",
        "sales_channel_id"
    ]

    required_customer_columns = [
        "customer_id",
        "age",
        "club_member_status",
        "fashion_news_frequency"
    ]

    required_article_columns = [
        "article_id",
        "prod_name",
        "product_type_name",
        "product_group_name",
        "colour_group_name",
        "garment_group_name",
        "index_name"
    ]

    for column in required_transaction_columns:

        if column not in transactions.columns:
            raise ValueError(
                f"Missing transaction column: {column}"
            )

    for column in required_customer_columns:

        if column not in customers.columns:
            raise ValueError(
                f"Missing customer column: {column}"
            )

    for column in required_article_columns:

        if column not in articles.columns:
            raise ValueError(
                f"Missing article column: {column}"
            )

    # Convert transaction data types

    transactions = (
        transactions
        .withColumn(
            "t_dat",
            F.to_date("t_dat", "yyyy-MM-dd")
        )
        .withColumn(
            "price",
            F.col("price").cast("double")
        )
        .withColumn(
            "sales_channel_id",
            F.col("sales_channel_id").cast("integer")
        )
        .withColumn(
            "event_day",
            F.unix_timestamp("t_dat")
        )
    )

    # Convert customer age

    customers = customers.withColumn(
        "age",
        F.col("age").cast("integer")
    )

    print("Articles loaded successfully")
    print("Customers loaded successfully")
    print("Transactions loaded successfully")

    print("\nDataset columns:")

    print("Articles:", len(articles.columns))
    print("Customers:", len(customers.columns))
    print("Transactions:", len(transactions.columns))

    return articles, customers, transactions


# ============================================================
# DAY 1-2: DATA QUALITY AND MISSING VALUE HANDLING
# ============================================================

def clean_datasets(
    articles,
    customers,
    transactions
):

    print("\n" + "=" * 60)
    print("DAY 1-2: DATA CLEANING")
    print("=" * 60)

    # --------------------------------------------------------
    # TRANSACTION CLEANING
    # --------------------------------------------------------

    print("\nCleaning transactions...")

    original_transaction_count = transactions.count()

    transactions = transactions.filter(
        F.col("customer_id").isNotNull()
        & F.col("article_id").isNotNull()
        & F.col("t_dat").isNotNull()
        & F.col("price").isNotNull()
        & F.col("sales_channel_id").isNotNull()
        & (F.col("price") > 0)
        & F.col("sales_channel_id").isin(1, 2)
    )

    cleaned_transaction_count = transactions.count()

    # We preserve identical transaction rows.
    # The source does not contain a unique transaction ID,
    # so identical rows cannot safely be assumed duplicates.

    # --------------------------------------------------------
    # CUSTOMER CLEANING
    # --------------------------------------------------------

    print("\nCleaning customers...")

    original_customer_count = customers.count()

    customers = customers.dropDuplicates(
        ["customer_id"]
    )

    # Handle invalid age values

    customers = customers.withColumn(
        "age",
        F.when(
            F.col("age").between(1, 100),
            F.col("age")
        ).otherwise(F.lit(None).cast("integer"))
    )

    # Handle missing categorical information

    customers = (
        customers
        .withColumn(
            "club_member_status",
            F.coalesce(
                F.col("club_member_status"),
                F.lit("Unknown")
            )
        )
        .withColumn(
            "fashion_news_frequency",
            F.coalesce(
                F.col("fashion_news_frequency"),
                F.lit("Unknown")
            )
        )
    )

    # Age bucket generation

    customers = customers.withColumn(
        "age_bucket",

        F.when(
            F.col("age").isNull(),
            "Unknown"
        )
        .when(F.col("age") < 20, "Under_20")
        .when(F.col("age") < 30, "20_29")
        .when(F.col("age") < 40, "30_39")
        .when(F.col("age") < 50, "40_49")
        .when(F.col("age") < 60, "50_59")
        .otherwise("60_plus")
    )

    cleaned_customer_count = customers.count()

    # --------------------------------------------------------
    # ARTICLE CLEANING
    # --------------------------------------------------------

    print("\nCleaning articles...")

    original_article_count = articles.count()

    articles = (
        articles
        .filter(F.col("article_id").isNotNull())
        .dropDuplicates(["article_id"])
    )

    article_text_columns = [
        "prod_name",
        "product_type_name",
        "product_group_name",
        "colour_group_name",
        "garment_group_name",
        "index_name",
        "department_name",
        "detail_desc"
    ]

    for column in article_text_columns:

        if column in articles.columns:

            articles = articles.withColumn(
                column,
                F.coalesce(
                    F.col(column),
                    F.lit("Unknown")
                )
            )

    cleaned_article_count = articles.count()

    # --------------------------------------------------------
    # SAVE CLEANING REPORT
    # --------------------------------------------------------

    report = {

        "transactions": {
            "original": original_transaction_count,
            "cleaned": cleaned_transaction_count,
            "removed": (
                original_transaction_count
                - cleaned_transaction_count
            )
        },

        "customers": {
            "original": original_customer_count,
            "cleaned": cleaned_customer_count
        },

        "articles": {
            "original": original_article_count,
            "cleaned": cleaned_article_count
        }
    }

    save_json_report(
        report,
        "data_cleaning_report.json"
    )

    # --------------------------------------------------------
    # SAVE CLEANED DATASETS
    # --------------------------------------------------------

    print("\nSaving cleaned datasets...")

    transactions.write.mode(
        "overwrite"
    ).parquet(str(TRANSACTIONS_OUTPUT))

    customers.write.mode(
        "overwrite"
    ).parquet(str(CUSTOMER_FEATURES_OUTPUT))

    articles.write.mode(
        "overwrite"
    ).parquet(str(ARTICLE_FEATURES_OUTPUT))

    print("\nData cleaning completed!")

    print("Clean transactions:", cleaned_transaction_count)
    print("Clean customers:", cleaned_customer_count)
    print("Clean articles:", cleaned_article_count)

    return articles, customers, transactions


# ============================================================
# DAY 3: CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

def cold_start_analysis(
    spark,
    transactions,
    customers,
    articles
):

    print("\n" + "=" * 60)
    print("DAY 3: COLD START ANALYSIS")
    print("=" * 60)

    # Calculate temporal cutoff using transaction dates

    cutoff_values = (
        transactions
        .select("event_day")
        .where(F.col("event_day").isNotNull())
        .stat.approxQuantile(
            "event_day",
            [0.80],
            0.001
        )
    )

    if not cutoff_values:

        raise ValueError(
            "Unable to calculate temporal split."
        )

    cutoff_timestamp = int(cutoff_values[0])

    cutoff_date = F.to_date(
        F.from_unixtime(
            F.lit(cutoff_timestamp)
        )
    )

    print("Temporal cutoff date:")

    cutoff_result = (
        spark.range(1)
        .select(cutoff_date.alias("cutoff_date"))
        .first()["cutoff_date"]
    )

    print(cutoff_result)

    # Training transactions

    train = transactions.filter(
        F.col("t_dat") <= cutoff_date
    )

    # Testing transactions

    test = transactions.filter(
        F.col("t_dat") > cutoff_date
    )

    # --------------------------------------------------------
    # IDENTIFY KNOWN USERS AND ITEMS
    # --------------------------------------------------------

    train_users = (
        train
        .select("customer_id")
        .distinct()
    )

    train_items = (
        train
        .select("article_id")
        .distinct()
    )

    # --------------------------------------------------------
    # COLD START USERS
    # --------------------------------------------------------

    cold_users = (
        test
        .select("customer_id")
        .distinct()
        .join(
            train_users,
            on="customer_id",
            how="left_anti"
        )
    )

    # --------------------------------------------------------
    # COLD START ITEMS
    # --------------------------------------------------------

    cold_items = (
        test
        .select("article_id")
        .distinct()
        .join(
            train_items,
            on="article_id",
            how="left_anti"
        )
    )

    # --------------------------------------------------------
    # UNKNOWN CUSTOMER METADATA
    # --------------------------------------------------------

    unknown_customers = (
        transactions
        .select("customer_id")
        .distinct()
        .join(
            customers.select("customer_id"),
            on="customer_id",
            how="left_anti"
        )
    )

    # --------------------------------------------------------
    # UNKNOWN ARTICLE METADATA
    # --------------------------------------------------------

    unknown_articles = (
        transactions
        .select("article_id")
        .distinct()
        .join(
            articles.select("article_id"),
            on="article_id",
            how="left_anti"
        )
    )

    # --------------------------------------------------------
    # SAVE TRAIN AND TEST DATA
    # --------------------------------------------------------

    train.write.mode(
        "overwrite"
    ).parquet(
        str(TRAIN_TEST_OUTPUT / "train")
    )

    test.write.mode(
        "overwrite"
    ).parquet(
        str(TRAIN_TEST_OUTPUT / "test")
    )

    cold_users.write.mode(
        "overwrite"
    ).parquet(
        str(TRAIN_TEST_OUTPUT / "cold_start_users")
    )

    cold_items.write.mode(
        "overwrite"
    ).parquet(
        str(TRAIN_TEST_OUTPUT / "cold_start_items")
    )

    unknown_customers.write.mode(
        "overwrite"
    ).parquet(
        str(TRAIN_TEST_OUTPUT / "unknown_customers")
    )

    unknown_articles.write.mode(
        "overwrite"
    ).parquet(
        str(TRAIN_TEST_OUTPUT / "unknown_articles")
    )

    # --------------------------------------------------------
    # GENERATE REPORT
    # --------------------------------------------------------

    metrics = {

        "cutoff_date": str(cutoff_result),

        "train_transactions": train.count(),

        "test_transactions": test.count(),

        "train_users": train_users.count(),

        "train_items": train_items.count(),

        "cold_start_users": cold_users.count(),

        "cold_start_items": cold_items.count(),

        "unknown_customers": unknown_customers.count(),

        "unknown_articles": unknown_articles.count()
    }

    save_json_report(
        metrics,
        "cold_start_report.json"
    )

    print("\nCold-start analysis completed!")

    for key, value in metrics.items():

        print(f"{key}: {value}")

    return train, test


# ============================================================
# DAY 4-6: CONTEXTUAL FEATURE ENGINEERING
# ============================================================

def generate_contextual_features(
    transactions,
    customers,
    articles
):

    print("\n" + "=" * 60)
    print("DAY 4-6: CONTEXTUAL FEATURE ENGINEERING")
    print("=" * 60)

    # --------------------------------------------------------
    # FEATURE 1: PURCHASE DATE FEATURES
    # --------------------------------------------------------

    print("\nGenerating purchase date features...")

    contextual = (
        transactions
        .withColumn(
            "purchase_day",
            F.dayofweek("t_dat")
        )
        .withColumn(
            "purchase_month",
            F.month("t_dat")
        )
        .withColumn(
            "purchase_day_of_month",
            F.dayofmonth("t_dat")
        )
        .withColumn(
            "purchase_year",
            F.year("t_dat")
        )
        .withColumn(
            "purchase_quarter",
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

    # --------------------------------------------------------
    # FEATURE 2: CUSTOMER HISTORICAL PURCHASES
    # --------------------------------------------------------

    print("\nGenerating customer history features...")

    # Aggregate customer purchases by day.
    # This prevents arbitrary ordering of same-day purchases.

    customer_daily = (
        transactions
        .groupBy(
            "customer_id",
            "t_dat"
        )
        .agg(
            F.count("*").alias("daily_customer_purchases")
        )
        .withColumn(
            "event_day",
            F.unix_timestamp("t_dat")
        )
    )

    customer_window = (
        Window
        .partitionBy("customer_id")
        .orderBy("event_day")
    )

    previous_customer_window = (
        customer_window
        .rangeBetween(
            Window.unboundedPreceding,
            -1
        )
    )

    customer_history = (
        customer_daily
        .withColumn(
            "historical_purchase_count",
            F.coalesce(
                F.sum(
                    "daily_customer_purchases"
                ).over(previous_customer_window),
                F.lit(0)
            )
        )
        .withColumn(
            "last_purchase_day",
            F.max("event_day").over(
                previous_customer_window
            )
        )
        .withColumn(
            "days_since_previous_purchase",
            F.when(
                F.col("last_purchase_day").isNull(),
                -1
            ).otherwise(
                F.floor(
                    (
                        F.col("event_day")
                        - F.col("last_purchase_day")
                    ) / 86400
                ).cast("integer")
            )
        )
        .withColumn(
            "is_first_purchase",
            F.when(
                F.col("last_purchase_day").isNull(),
                1
            ).otherwise(0)
        )
        .select(
            "customer_id",
            "t_dat",
            "historical_purchase_count",
            "days_since_previous_purchase",
            "is_first_purchase"
        )
    )

    contextual = contextual.join(
        customer_history,
        on=["customer_id", "t_dat"],
        how="left"
    )

    # --------------------------------------------------------
    # FEATURE 3: PRODUCT POPULARITY OVER TIME
    # --------------------------------------------------------

    print("\nGenerating product popularity features...")

    daily_popularity = (
        transactions
        .groupBy(
            "article_id",
            "t_dat"
        )
        .agg(
            F.count("*").alias("daily_sales")
        )
        .withColumn(
            "event_day",
            F.unix_timestamp("t_dat")
        )
    )

    popularity_window = (
        Window
        .partitionBy("article_id")
        .orderBy("event_day")
    )

    seven_day_window = (
        popularity_window
        .rangeBetween(
            -7 * 86400,
            -1
        )
    )

    thirty_day_window = (
        popularity_window
        .rangeBetween(
            -30 * 86400,
            -1
        )
    )

    daily_popularity = (
        daily_popularity
        .withColumn(
            "popularity_last_7_days",
            F.coalesce(
                F.sum("daily_sales").over(
                    seven_day_window
                ),
                F.lit(0)
            )
        )
        .withColumn(
            "popularity_last_30_days",
            F.coalesce(
                F.sum("daily_sales").over(
                    thirty_day_window
                ),
                F.lit(0)
            )
        )
        .withColumn(
            "popularity_before_today",
            F.coalesce(
                F.sum("daily_sales").over(
                    popularity_window.rangeBetween(
                        Window.unboundedPreceding,
                        -1
                    )
                ),
                F.lit(0)
            )
        )
        .select(
            "article_id",
            "t_dat",
            "daily_sales",
            "popularity_last_7_days",
            "popularity_last_30_days",
            "popularity_before_today"
        )
    )

    contextual = contextual.join(
        daily_popularity,
        on=["article_id", "t_dat"],
        how="left"
    )

    # --------------------------------------------------------
    # FEATURE 4: CUSTOMER DEMOGRAPHICS
    # --------------------------------------------------------

    print("\nJoining customer features...")

    customer_columns = [
        "customer_id",
        "age",
        "age_bucket",
        "club_member_status",
        "fashion_news_frequency"
    ]

    contextual = contextual.join(
        customers.select(*customer_columns),
        on="customer_id",
        how="left"
    )

    # --------------------------------------------------------
    # FEATURE 5: PRODUCT CHARACTERISTICS
    # --------------------------------------------------------

    print("\nJoining article features...")

    article_columns = [
        "article_id",
        "product_type_name",
        "product_group_name",
        "colour_group_name",
        "garment_group_name",
        "index_name"
    ]

    available_article_columns = [
        column
        for column in article_columns
        if column in articles.columns
    ]

    contextual = contextual.join(
        articles.select(*available_article_columns),
        on="article_id",
        how="left"
    )

    # --------------------------------------------------------
    # FEATURE 6: SALES CHANNEL
    # --------------------------------------------------------

    contextual = contextual.withColumn(
        "sales_channel",
        F.when(
            F.col("sales_channel_id") == 1,
            "Online"
        )
        .when(
            F.col("sales_channel_id") == 2,
            "Offline"
        )
        .otherwise("Unknown")
    )

    # --------------------------------------------------------
    # HANDLE MISSING CONTEXTUAL CATEGORIES
    # --------------------------------------------------------

    categorical_columns = [
        "club_member_status",
        "fashion_news_frequency",
        "age_bucket",
        "product_type_name",
        "product_group_name",
        "colour_group_name",
        "garment_group_name",
        "index_name",
        "sales_channel"
    ]

    for column in categorical_columns:

        if column in contextual.columns:

            contextual = contextual.withColumn(
                column,
                F.coalesce(
                    F.col(column),
                    F.lit("Unknown")
                )
            )

    # --------------------------------------------------------
    # SAVE CONTEXTUAL DATASET
    # --------------------------------------------------------

    print("\nSaving contextual features...")

    contextual.write.mode(
        "overwrite"
    ).parquet(str(CONTEXT_OUTPUT))

    print("\nContextual feature engineering completed!")

    print("\nGenerated feature schema:")

    contextual.printSchema()

    print(
        "\nTotal contextual records:",
        contextual.count()
    )

    return contextual


# ============================================================
# DAY 7: FEATURE VOCABULARY GENERATION
# ============================================================

def create_vocabulary(
    dataframe,
    column,
    output_name
):

    print(
        f"\nCreating vocabulary: {output_name}"
    )

    if column not in dataframe.columns:

        raise ValueError(
            f"Column not found: {column}"
        )

    # Convert feature values to strings

    unique_values = (
        dataframe
        .select(
            F.col(column)
            .cast("string")
            .alias("raw_id")
        )
        .filter(
            F.col("raw_id").isNotNull()
        )
        .distinct()
        .sort("raw_id")
    )

    # Reserve 0 for unknown / OOV values.
    # Known vocabulary values start from 1.

    vocabulary_rdd = (
        unique_values
        .rdd
        .map(lambda row: row["raw_id"])
        .zipWithIndex()
        .map(
            lambda item: Row(
                raw_id=item[0],
                embedding_id=int(item[1]) + 1
            )
        )
    )

    vocabulary = (
        dataframe.sparkSession
        .createDataFrame(vocabulary_rdd)
    )

    output_path = VOCAB_DIR / output_name

    vocabulary.write.mode(
        "overwrite"
    ).parquet(str(output_path))

    vocabulary_count = vocabulary.count()

    print(
        f"Vocabulary created: {output_name}"
    )

    print(
        f"Number of unique values: {vocabulary_count}"
    )

    return vocabulary


# ============================================================
# DAY 7: GENERATE ALL USER AND ITEM VOCABULARIES
# ============================================================

def generate_all_vocabularies(
    train,
    customers,
    articles
):

    print("\n" + "=" * 60)
    print("DAY 7: USER AND ITEM FEATURE VOCABULARIES")
    print("=" * 60)

    # --------------------------------------------------------
    # USER ID VOCABULARY
    # --------------------------------------------------------

    print("\nGenerating user ID vocabulary...")

    create_vocabulary(
        train,
        "customer_id",
        "user_customer_id"
    )

    # --------------------------------------------------------
    # ITEM ID VOCABULARY
    # --------------------------------------------------------

    print("\nGenerating item ID vocabulary...")

    create_vocabulary(
        train,
        "article_id",
        "item_article_id"
    )

    # --------------------------------------------------------
    # USER CATEGORICAL VOCABULARIES
    # --------------------------------------------------------

    user_features = [
        "club_member_status",
        "fashion_news_frequency",
        "age_bucket"
    ]

    for column in user_features:

        if column in customers.columns:

            create_vocabulary(
                customers,
                column,
                f"user_{column}"
            )

    # --------------------------------------------------------
    # ITEM CATEGORICAL VOCABULARIES
    # --------------------------------------------------------

    item_features = [
        "product_type_name",
        "product_group_name",
        "colour_group_name",
        "garment_group_name",
        "index_name"
    ]

    for column in item_features:

        if column in articles.columns:

            create_vocabulary(
                articles,
                column,
                f"item_{column}"
            )

    print("\n" + "=" * 60)
    print("ALL FEATURE VOCABULARIES GENERATED")
    print("=" * 60)
