import pandas as pd 

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors = "coerce")
    df = df.dropna(subset=['Start_Time'])


    df['Hour'] = df['Start_Time'].dt.hour
    df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
    
    #helps calculate rush hour traffic
    df['rush_hour'] = df['Hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0)

    #weekend
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    #day/night
    df['is_night'] = ((df['hour'] >= 18) | (df['hour'] < 6)).astype(int)
    df['is_day'] = ((df['hour'] >= 6) & (df['hour'] < 18)).astype(int)
    return df

def encode(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = ['Weather_Condition']
    existing_cols = [col for col in categorical_cols if col in df.columns]

    if existing_cols:
        df = pd.get_dummies(df, columns=existing_cols, drop_first=True)
    return df
    