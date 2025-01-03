
def get_basic_stats(df):
    return df.describe()

def get_correlation_matrix(df):
    return df.corr()

def analyze_promo_effect(df):
    promo_sales = df[df['Promo'] == 1]['Sales']
    non_promo_sales = df[df['Promo'] == 0]['Sales']
    return promo_sales.mean(), non_promo_sales.mean()


def store_type_analysis(df):
    return df.groupby('StoreType')['Sales'].mean()
