import pandas as pd


def align_to_hopsworks_schema(df: pd.DataFrame, fg) -> pd.DataFrame:
    """
    Convert dataframe columns to the exact types expected by
    the existing Hopsworks Feature Group.

    This prevents errors such as:
        expected double, derived from input bigint
    """

    result = df.copy()

    schema = {
        feature.name: str(feature.type).lower()
        for feature in fg.columns
    }

    for col, hops_type in schema.items():

        if col not in result.columns:
            raise ValueError(
                f"Missing required V2 feature column: {col}"
            )

        # Floating point types
        if hops_type in {
            "double",
            "float",
            "float32",
            "float64",
            "real",
        }:
            result[col] = pd.to_numeric(
                result[col],
                errors="coerce"
            ).astype("float64")

        # Integer types
        elif hops_type in {
            "bigint",
            "int",
            "integer",
            "long",
            "int64",
            "int32",
        }:
            result[col] = pd.to_numeric(
                result[col],
                errors="raise"
            ).astype("int64")

        # Date
        elif hops_type == "date":
            result[col] = pd.to_datetime(
                result[col]
            ).dt.date

        # Timestamp
        elif "timestamp" in hops_type:
            result[col] = pd.to_datetime(
                result[col],
                utc=True
            )

        # Strings
        elif (
            "string" in hops_type
            or "varchar" in hops_type
        ):
            result[col] = result[col].astype(object)

    return result