
import os

from pyspark.sql import SparkSession

from src.config import (
    APP_NAME,
    MASTER,
    SHUFFLE_PARTITIONS,
    HADOOP_HOME,
    WINUTILS_PATH
)


def create_spark_session():

    # Windows Hadoop environment

    if WINUTILS_PATH.exists():

        os.environ["HADOOP_HOME"] = str(HADOOP_HOME)

        os.environ["hadoop.home.dir"] = str(HADOOP_HOME)

        os.environ["PATH"] = (
            str(HADOOP_HOME / "bin")
            + os.pathsep
            + os.environ.get("PATH", "")
        )

    else:

        print(
            "WARNING: winutils.exe not found at:",
            WINUTILS_PATH
        )

    spark = (

        SparkSession.builder

        .appName(APP_NAME)

        .master(MASTER)

        .config(
            "spark.sql.shuffle.partitions",
            str(SHUFFLE_PARTITIONS)
        )

        .config(
            "spark.sql.adaptive.enabled",
            "true"
        )

        .config(
            "spark.sql.adaptive.coalescePartitions.enabled",
            "true"
        )

        .config(
            "spark.sql.parquet.compression.codec",
            "snappy"
        )

        .getOrCreate()

    )

    spark.sparkContext.setLogLevel("WARN")

    print("\nSPARK SESSION INITIALIZED")

    print("Spark version:", spark.version)

    print("Master:", spark.sparkContext.master)

    return spark
