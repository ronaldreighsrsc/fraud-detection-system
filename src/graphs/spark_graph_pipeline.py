"""
Pipeline Distribuido PySpark para Databricks (Cálculo Batch de Métricas de Grafo).
Calcula In-Degree, Out-Degree y Ratios de Cuentas Mula a escala masiva sobre Delta Lake.
"""
from typing import Optional


def run_pyspark_graph_feature_pipeline(delta_table_path: str, output_feature_store_path: str,
                                       spark_session: Optional[object] = None):
    """
    Ejecuta el pipeline distribuido sobre el histórico de transferencias en Delta Lake.
    Crea la tabla consolidada de métricas de red para sincronización en Redis.
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
    except ImportError:
        print("⚠️ PySpark no está instalado en este entorno. Este job está diseñado para ejecutarse en Databricks.")
        return None

    spark = spark_session or SparkSession.builder \
        .appName("Bci-PLAFT-GraphFeatureEngineering") \
        .getOrCreate()

    # 1. Leer histórico de transacciones desde Delta Lake
    df_tx = spark.read.format("delta").load(delta_table_path)

    # 2. Calcular In-Degree masivo (Cuentas que reciben transferencias de distintos orígenes)
    df_in_degree = df_tx.groupBy("destination_account") \
        .agg(
            F.countDistinct("origin_account").alias("pyspark_in_degree"),
            F.sum("transaction_amount").alias("total_amount_received"),
            F.count("transaction_id").alias("tx_received_count")
        ) \
        .withColumnRenamed("destination_account", "account_id")

    # 3. Calcular Out-Degree masivo (Cuentas que dispersan fondos a otros destinos)
    df_out_degree = df_tx.groupBy("origin_account") \
        .agg(
            F.countDistinct("destination_account").alias("pyspark_out_degree"),
            F.sum("transaction_amount").alias("total_amount_sent")
        ) \
        .withColumnRenamed("origin_account", "account_id")

    # 4. Join y Detección Distribuida de Cuentas Mula
    df_graph_features = df_in_degree.join(df_out_degree, on="account_id", how="outer").na.fill(0)

    df_graph_features = df_graph_features.withColumn(
        "is_mule_candidate",
        F.when((F.col("pyspark_in_degree") >= 5) & (F.col("pyspark_out_degree") <= 2), 1.0).otherwise(0.0)
    )

    # 5. Guardar en Delta Lake para consulta en tiempo real desde FastAPI
    df_graph_features.write.format("delta").mode("overwrite").save(output_feature_store_path)
    print("✅ Características de grafo calculadas y almacenadas exitosamente con PySpark.")
    return df_graph_features
