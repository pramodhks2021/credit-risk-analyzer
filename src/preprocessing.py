import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Drop columns with excessive missing values
    columns_to_drop = [
        'COMMONAREA_MEDI', 'COMMONAREA_AVG', 'COMMONAREA_MODE',
        'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAPARTMENTS_MEDI',
        'FONDKAPREMONT_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAPARTMENTS_AVG', 'LIVINGAPARTMENTS_MEDI'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    # Fill remaining missing values with median
    df = df.fillna(df.median(numeric_only=True))

    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    return df
