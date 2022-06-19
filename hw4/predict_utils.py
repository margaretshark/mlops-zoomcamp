import pickle
import pandas as pd
import os 
def read_data(filename):
    df = pd.read_parquet(filename)
    categorical = ['PUlocationID', 'DOlocationID']
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df, categorical

def predict(df, categorical, output_df):
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

        dicts = df[categorical].to_dict(orient='records')
        X_val = dv.transform(dicts)
        y_pred = lr.predict(X_val)

    df['ride_id'] = '{:04d}/{:02d}_'.format(2021, 2) + df.index.astype('str')
    df['prediction'] = y_pred
    df[['ride_id', 'prediction']].to_parquet(
        output_df,
        engine='pyarrow',
        compression=None,
        index=False
    )
    return round(y_pred.mean(), 2), os.path.getsize(output_df)/(1024*1024)