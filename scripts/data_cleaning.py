
def clean_missing_data(df):
    # Example: Drop rows with missing values in essential columns
    # Handle competition-related columns
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)
    df['PromoInterval'] = df['PromoInterval'].fillna("None")
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].median())
    
    return df

def handle_outliers(df):
    # Example: Remove rows where 'Sales' is an outlier
    Q1 = df['Sales'].quantile(0.25)
    Q3 = df['Sales'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Sales'] >= (Q1 - 1.5 * IQR)) & (df['Sales'] <= (Q3 + 1.5 * IQR))]
    return df

def encode_categorical(df):
    df['StoreType'] = df['StoreType'].map({'a': 0, 'b': 1, 'c': 2, 'd': 3})
    return df
