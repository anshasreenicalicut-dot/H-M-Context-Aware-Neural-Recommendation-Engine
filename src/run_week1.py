
from src.spark_utils import create_spark_session

from src.week1_pipeline import (
    load_datasets,
    clean_datasets,
    cold_start_analysis,
    generate_contextual_features,
    generate_all_vocabularies
)


def main():

    spark = create_spark_session()

    try:

        print("\n")
        print("=" * 70)
        print("CONTEXT-AWARE NEURAL RECOMMENDATION ENGINE")
        print("H&M PERSONALIZED FASHION RECOMMENDATIONS")
        print("WEEK 1: DISTRIBUTED DATA PROCESSING")
        print("=" * 70)

        # ------------------------------------------------
        # DAY 1: LOAD DATA
        # ------------------------------------------------

        articles, customers, transactions = load_datasets(
            spark
        )

        # ------------------------------------------------
        # DAY 1-2: CLEAN DATA
        # ------------------------------------------------

        articles, customers, transactions = clean_datasets(
            articles,
            customers,
            transactions
        )

        # ------------------------------------------------
        # DAY 3: COLD START ANALYSIS
        # ------------------------------------------------

        train, test = cold_start_analysis(
            spark,
            transactions,
            customers,
            articles
        )

        # ------------------------------------------------
        # DAY 4-6: CONTEXTUAL FEATURE ENGINEERING
        # ------------------------------------------------

        contextual = generate_contextual_features(
            transactions,
            customers,
            articles
        )

        # ------------------------------------------------
        # DAY 7: FEATURE VOCABULARIES
        # ------------------------------------------------

        generate_all_vocabularies(
            train,
            customers,
            articles
        )

        # ------------------------------------------------
        # FINAL SUMMARY
        # ------------------------------------------------

        print("\n")
        print("=" * 70)
        print("WEEK 1 PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print("\nGenerated outputs:")

        print("1. Cleaned transaction dataset")
        print("2. Cleaned customer dataset")
        print("3. Cleaned article dataset")
        print("4. Chronological training dataset")
        print("5. Chronological testing dataset")
        print("6. Cold-start user analysis")
        print("7. Cold-start item analysis")
        print("8. Customer historical features")
        print("9. Product popularity features")
        print("10. Purchase date features")
        print("11. Customer demographic features")
        print("12. Product contextual features")
        print("13. User feature vocabularies")
        print("14. Item feature vocabularies")

        print("\nAll Week 1 operations finished.")

    except Exception as error:

        print("\n" + "=" * 70)
        print("WEEK 1 PIPELINE FAILED")
        print("=" * 70)

        print("Error type:", type(error).__name__)
        print("Error details:", str(error))

        raise

    finally:

        spark.stop()

        print("\nSpark session stopped.")


if __name__ == "__main__":

    main()
