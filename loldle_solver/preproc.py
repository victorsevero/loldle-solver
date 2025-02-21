import polars as pl

df = pl.read_csv("loldle.csv")
df = df.rename({"Range type": "Range"})
df = df.with_columns(
    Name=pl.col("Name").str.strip_chars().str.replace("&amp;", "&"),
    Position=pl.col("Position").str.split(",\n"),
    Species=pl.col("Species").str.split(",\n"),
    Range=pl.col("Range").str.split(",\n"),
    Region=pl.col("Region").str.split(",\n"),
)
df = df.drop("Range type", "")
df.columns = [col.lower() for col in df.columns]
df.write_parquet("preproc.parquet")
