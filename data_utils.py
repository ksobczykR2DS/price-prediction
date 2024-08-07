import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def _read_data():
    df = pd.read_csv('baysian-search/data/train.csv')
    df = df.drop('Id', axis='columns')
    return df


def _fill_missing_values(df):
    """Handling missing values"""
    columns_to_fill = ['MasVnrArea', 'LotFrontage', 'Electrical']
    for c in columns_to_fill:
        mode_value = df[c].mode()[0]
        df[c] = df[c].fillna(mode_value)

    columns_to_mark_as_absence = ['BsmtFinType1', 'BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType2', 'GarageQual',
                                  'GarageFinish', 'GarageType', 'GarageCond', 'FireplaceQu', 'MasVnrType', 'Fence',
                                  'Alley', 'MiscFeature', 'PoolQC']
    for c in columns_to_mark_as_absence:
        df[c] = df[c].fillna('Absence')

    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    assert df.isna().sum().sum() == 0, "There are some missing values in df"

    return df


def _feature_engineering(df):
    # Calculate the age of the property
    df['AgeOfProperty'] = df['YrSold'] - df['YearBuilt']

    # Calculate the total square footage of the property
    df['TotalSqFt'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] +
                       df['1stFlrSF'] + df['2ndFlrSF'] + df['GarageArea'])

    # Calculate the number of years the garage was built after the house
    df['GarageBuildYearsAfterHouse'] = df['GarageYrBlt'] - df['YearBuilt']

    # Calculate the total number of bathrooms
    df['TotalBathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) +
                            df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))

    # Calculate the total porch square footage
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['3SsnPorch'] +
                          df['EnclosedPorch'] + df['ScreenPorch'] +
                          df['WoodDeckSF'])

    # Create binary features indicating the presence of specific amenities
    df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    # Drop the original columns that have been transformed
    df = df.drop(columns=['PoolArea', '2ndFlrSF', 'GarageArea', 'TotalBsmtSF', 'Fireplaces'])

    return df


def _one_hot_encoding(df):
    """Categorical Data Handling"""
    encoder = OneHotEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns
    encoded_data = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(
        encoded_data.toarray(),
        columns=encoder.get_feature_names_out(categorical_columns)
    )
    df = pd.concat([df, encoded_df], axis=1)
    df = df.drop(columns=categorical_columns)
    assert not df.index.duplicated().any(), "Found duplicated indexes"

    return df


def load_raw_data(feature_engineering):
    df = _read_data()
    df = _fill_missing_values(df=df)

    target_col = 'SalePrice'
    y = df.pop(target_col)
    x = df.copy()

    if feature_engineering:
        x = _feature_engineering(x)

    x = _one_hot_encoding(x)

    return x, y


def load_training_data(feature_engineering):
    """Load data and split it into training and test"""
    x, y = load_raw_data(feature_engineering=feature_engineering)

    # Split into training and testing datasets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, shuffle=True)

    print(f"{x_train.shape[0]=}, {x_test.shape[0]=}\n{y_train.shape[0]=}, {y_test.shape[0]=}")

    return x_train, x_test, y_train, y_test
