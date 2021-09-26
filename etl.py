import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import IntegerType

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config["AWS"]['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config["AWS"]['AWS_SECRET_ACCESS_KEY']
os.environ['S3_LOCATION_OUT']= config["AWS"]['S3_LOCATION_OUT']
os.environ['S3_LOCATION_IN']= config["AWS"]['S3_LOCATION_IN']

def create_spark_session():
    """
    Description:
    The function creates a spark session
        
    Returns: None
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Description:
    The function executes the following steps:
    
    - Dowload the data from the song_data files
    
    - Create the dataframes for songs and artists
    
    - Stores the dataframes into the selected location
    
    Arguments:
        spark: the SparkSession object
        input_data: base location of the input raw data
        output_data: base location of the output transformed data
        
    Returns: None
    """
    # get filepath to song data file
    song_data = input_data + "song_data/*/*/*/*.json"
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select(df.song_id,
                            df.title,
                            df.artist_id,
                            df.year,
                            df.duration).dropDuplicates()
     
 
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.mode("overwrite").partitionBy("year","artist_id").parquet(output_data + "songs.parquet") 
 
    # extract columns to create artists table
    artists_table = df.select(df.artist_id,
                              df.artist_name.alias("name"),
                              df.artist_location.alias("location"),
                              df.artist_latitude.alias("latitude"),
                              df.artist_longitude.alias("longitude")).dropDuplicates()
    
    # write artists table to parquet files
    artists_table.write.mode("overwrite").parquet(output_data + "artists.parquet") 

def process_log_data(spark, input_data, output_data):
    """
    Description:
    The function executes the following steps:
    
    - Dowload the data from the log_data files
    
    - Create the dataframes for users, time, and song plays
    
    - Stores the dataframes into the selected location
    
    Arguments:
        spark: the SparkSession object
        input_data: base location of the input raw data
        output_data: base location of the output transformed data
        
    Returns: None
    """
    # get filepath to log data file
    log_data = input_data + "/log_data/*.json"

    # read log data file
    df = spark.read.json(log_data)

    # filter by actions for song plays
    df = df.filter(df.page == "NextSong")

    # extract columns for users table    
    users_table =  df.select(df.userId.alias("user_id"),
                              df.firstName.alias("first_name"),
                              df.lastName.alias("last_name"),
                              df.gender,
                              df.level).dropDuplicates()
    
    # write users table to parquet files
    users_table.write.mode("overwrite").parquet(output_data + "users.parquet") 

    # create timestamp column from original timestamp column 
    transform_timestamp = udf(lambda time: time/1000)
    df =df.withColumn('timestamp', transform_timestamp(col('ts')))
    
    get_dateTime = udf(lambda time: datetime.fromtimestamp(time))
    # create datetime column from original timestamp column

    df = df.withColumn("date_time",get_dateTime(col('timestamp')))

     # extract columns to create time table
    get_hour =  udf(lambda dt : int(dt.hour), IntegerType())
    get_day =  udf(lambda dt : int(dt.day), IntegerType())
    get_week =  udf(lambda dt : int(dt.isocalendar()[1]), IntegerType())
    get_month =  udf(lambda dt : int(dt.month), IntegerType())
    get_year =  udf(lambda dt : int(dt.year), IntegerType())
    get_weekday =  udf(lambda dt : int(dt.weekday()), IntegerType())
    time_table = df.select([df.date_time, df.timestamp])
    time_table = time_table.withColumn("hour",get_hour(col("date_time"))) \
                           .withColumn("day",get_day(col("date_time"))) \
                           .withColumn("week",get_week(col("date_time"))) \
                           .withColumn("month",get_month(col("date_time"))) \
                           .withColumn("year",get_year(col("date_time"))) \
                           .withColumn("weekday",get_weekday(col("date_time"))).dropDuplicates()
    
    # write time table to parquet files partitioned by year and month
    time_table.write.mode("overwrite").partitionBy("year","month").parquet(output_data + "time.parquet") 

    # read in song data to use for songplays table
    song_data = input_data + "song_data/*/*/*/*.json"
    song_df = spark.read.json(song_data)


    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = song_df.join(df,
                                   [song_df.title == df.song,song_df.artist_name == df.artist],
                                   "inner").drop("year")
    songplays_table = songplays_table.join(time_table,
                                           songplays_table.timestamp == time_table.timestamp.alias("t_timestamp"),
                                           "left").drop("t_timestamp")

    songplays_table = songplays_table.select(songplays_table.ts.alias("start_time"),
                                             songplays_table.userId.alias("user_id"),
                                             songplays_table.song_id,
                                             songplays_table.artist_id,
                                             songplays_table.sessionId.alias("session_id"),
                                             songplays_table.location,
                                             songplays_table.userAgent.alias("user_agent"),
                                             songplays_table.year,
                                             songplays_table.month)\
    .withColumn("songplay_id",monotonically_increasing_id())
    
    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.mode("overwrite").partitionBy("year","month").parquet(output_data + "songplays.parquet") 


def main():
    """
    Description:
    The function executes the following steps
    - Creates a spark session
    
    - Loads all the data from the S3 buckets into spark
    
    - Inserts all the data into dataframes stored as parquets saved either locally or on S3 bucket(s)
        
    Returns: None
    """
    spark = create_spark_session()
    input_data = os.environ['S3_LOCATION_IN']
    output_data = os.environ['S3_LOCATION_OUT'] 
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
