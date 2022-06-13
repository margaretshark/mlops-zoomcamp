import datetime
import pandas as pd
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prefect import flow, get_run_logger, task
import pickle


from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    
    logger = get_run_logger()
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    logger.info
    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    
    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)

    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")

    return



@task
def get_paths(date):
    # eщё надо цикл с перестановкой года
    if date == None:
        m = int(datetime.datetime.now().strftime("%m"))
    else:
        m = int(datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%m"))
        
    if m == 2:
        val_m, train_m = 1, 12
    elif m == 1:
        val_m, train_m = 12, 11
    else: 
        val_m, train_m = m - 1, m - 2
        
    def f(val):

        if val >= 10:
            return str(val)
        else:
            return '0' + str(val)
        
    return  f"./data/fhv_tripdata_2021-{f(train_m)}.parquet", f"./data/fhv_tripdata_2021-{f(val_m)}.parquet"

@flow
def main(date=None):
    train_path, val_path = get_paths(date = date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    logger = get_run_logger()
    df_train, df_val = read_data(train_path), read_data(val_path)
    logger.info("Data was sucessfully loaded")
    
    df_train_processed = prepare_features(df_train, categorical)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    # save model wheights and preprocessing
    if date == None:
        date = datetime.datetime.now().strftime("%Y-%m-%d")

        
    with open(f"./models/model-{date}.bin", "wb") as file:
        pickle.dump(lr, file)
    with open(f"./models/dv-{date}.b", "wb") as file:
        pickle.dump(dv, file)
        
        
    file_name = f"./models/dv-{date}.b"
    

    logger.info(f"Size of dict vectorizer: {os.path.getsize(file_name)}")
    
    run_model(df_val_processed, categorical, dv, lr)

# main()
# main(date="2021-03-15")
# main(date="2021-08-15")

DeploymentSpec(
    flow = main,
    name = "every_15th_day_retraining_model",
    schedule = CronSchedule(cron="0 9 15 * *"),
    flow_runner = SubprocessFlowRunner(), # shows that it is a casual script( not a docker container/Kuber) 
    tags = ["ml", "9am15"]
)