import pandas as pd
from src.utils import check_columns, timer
 
@timer
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df = df.dropna(subset=['Start_Time'])
 
    df['Hour'] = df['Start_Time'].dt.hour
    df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
 
    # Rush hour: 7-9am and 4-6pm
    df['rush_hour'] = df['Hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0)
 
    # Weekend: Saturday=5, Sunday=6
    df['is_weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)  # FIX: was df['day_of_week']
 
    # Day / Night
    df['is_night'] = ((df['Hour'] >= 18) | (df['Hour'] < 6)).astype(int)   # FIX: was df['hour']
    df['is_day']   = ((df['Hour'] >= 6)  & (df['Hour'] < 18)).astype(int)  # FIX: was df['hour']
 
    return df  # FIX: was missing after dropna block
 
 
@timer
def encode(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = ['Weather_Condition', 'Wind_Direction', 'Sunrise_Sunset']
    existing_cols = check_columns(df, categorical_cols)
 
    if existing_cols:
        # Fill nulls before encoding so rows aren't silently dropped
        for col in existing_cols:
            df[col] = df[col].fillna('Unknown')
        df = pd.get_dummies(df, columns=existing_cols, drop_first=True)
 
    return df