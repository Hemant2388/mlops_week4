import pandas as pd

def test_data_validity():
    df = pd.read_csv("data/iris.csv")
    expected_cols = {'sepal_length','sepal_width','petal_length','petal_width','species'}
    
    # Print info for CML
    with open("test_output.txt", "a") as f:
        f.write("\n📁 Data Validation\n")
        f.write(f"✅ Total rows: {len(df)}\n")
        f.write(f"✅ Columns present: {', '.join(df.columns)}\n")
        f.write(f"✅ Null values present: {df.isnull().values.any()}\n")
        f.write(f"✅ Unique species count: {df['species'].nunique()}\n")

    # Validation assertions
    assert expected_cols.issubset(df.columns), "Missing columns in data"
    assert not df.isnull().values.any(), "Data contains null values"
    assert df['species'].nunique() == 3, "Unexpected number of species"
