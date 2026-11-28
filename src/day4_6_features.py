import json
from pyspark.sql import functions as F
from .config import *
from .spark_session import get_spark

def run():
    spark = get_spark("Week1_Days4to6")
    tx = spark.read.parquet(str(TX_OUT))
    articles = spark.read.options(**CSV_OPTIONS).option("inferSchema", "false").csv(str(ARTICLES_CSV))
    customers = spark.read.options(**CSV_OPTIONS).option("inferSchema", "false").csv(str(CUSTOMERS_CSV))

    # H&M t_dat is date-only: no reliable hour-of-day feature exists.
    ref_date = tx.agg(F.max("t_dat").alias("d")).first()["d"]
    print("Recency reference date:", ref_date)

    context = (tx
        .withColumn("purchase_year", F.year("t_dat"))
        .withColumn("purchase_month", F.month("t_dat"))
        .withColumn("purchase_day", F.dayofmonth("t_dat"))
        .withColumn("purchase_day_of_week", F.dayofweek("t_dat"))
        .withColumn("purchase_quarter", F.quarter("t_dat"))
        .withColumn("purchase_is_weekend", F.when(F.dayofweek("t_dat").isin(1, 7), 1).otherwise(0))
    )

    user = (context.groupBy("customer_id").agg(
        F.count("*").alias("user_transaction_count"),
        F.countDistinct("article_id").alias("user_unique_articles"),
        F.countDistinct("t_dat").alias("user_active_days"),
        F.min("t_dat").alias("user_first_purchase_date"),
        F.max("t_dat").alias("user_last_purchase_date"),
        F.avg("price").alias("user_avg_price"),
        F.sum("price").alias("user_total_spend"),
        F.avg("purchase_is_weekend").alias("user_weekend_purchase_ratio"),
        F.countDistinct("sales_channel_id").alias("user_channel_count")
    ).withColumn("user_recency_days", F.datediff(F.lit(ref_date), "user_last_purchase_date"))
     .join(customers, "customer_id", "left"))

    USER_OUT.parent.mkdir(parents=True, exist_ok=True)
    user.write.mode("overwrite").parquet(str(USER_OUT))

    daily = context.groupBy("article_id", "t_dat").agg(
        F.count("*").alias("daily_purchase_count"),
        F.countDistinct("customer_id").alias("daily_unique_buyers")
    )
    DAILY_POP_OUT.parent.mkdir(parents=True, exist_ok=True)
    daily.write.mode("overwrite").partitionBy("t_dat").parquet(str(DAILY_POP_OUT))

    d7, d30 = F.date_sub(F.lit(ref_date), 6), F.date_sub(F.lit(ref_date), 29)
    pop = context.groupBy("article_id").agg(
        F.count("*").alias("article_total_purchases"),
        F.countDistinct("customer_id").alias("article_unique_buyers"),
        F.max("t_dat").alias("article_last_purchase_date"),
        F.sum(F.when(F.col("t_dat") >= d7, 1).otherwise(0)).alias("article_purchases_last_7d"),
        F.sum(F.when(F.col("t_dat") >= d30, 1).otherwise(0)).alias("article_purchases_last_30d"),
        F.avg("price").alias("article_avg_selling_price")
    ).withColumn("article_recency_days", F.datediff(F.lit(ref_date), "article_last_purchase_date"))

    item = (articles.dropDuplicates(["article_id"]).join(pop, "article_id", "left")
        .fillna({
            "article_total_purchases": 0, "article_unique_buyers": 0,
            "article_purchases_last_7d": 0, "article_purchases_last_30d": 0
        })
        .withColumn("has_purchase_history", F.when(F.col("article_total_purchases") > 0, 1).otherwise(0))
    )
    ITEM_OUT.parent.mkdir(parents=True, exist_ok=True)
    item.write.mode("overwrite").parquet(str(ITEM_OUT))

    summary = {
        "reference_date": str(ref_date), "user_feature_rows": user.count(),
        "article_feature_rows": item.count(), "daily_popularity_rows": daily.count(),
        "time_feature_note": "t_dat has date but no hour/minute; hour-of-day is not inferred."
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "day4_6_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("User feature sample:"); user.show(5, truncate=False)
    print("Article feature sample:")
    item.select("article_id", "article_total_purchases", "article_purchases_last_7d",
                "article_purchases_last_30d", "has_purchase_history").show(5, truncate=False)
    spark.stop()

if __name__ == "__main__":
    run()
