from predict_utils import * 
import sys

def run():
    year, month = int(sys.argv[1]), int(sys.argv[2])

    df, cat = read_data('https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{:04d}-{:02d}.parquet'.format(year, month))
    output_df = "output/predictions_df_fhv_tripdata_{:04d}-{:02d}.parquet".format(year, month)

    mean_val, size = predict(df, cat, output_df)
    print("MEAN predicted duration for Feb 2021 - {}".format(mean_val))
    print(f"Size of output file {size} MB")

if __name__ == '__main__':
    print("Started")
    run()
    print("Running comleted")
