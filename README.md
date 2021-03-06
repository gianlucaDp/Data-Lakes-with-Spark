# Data Lakes with Spark
### Scope
The project scope is to extract data from S3 containers containing logs about songs and webpage usage, process the data
into a Spark cluster and save the processed data into parquet files stored into a S3 container
### Execution
![project](https://user-images.githubusercontent.com/36500094/134797865-4dcaa434-1fff-4300-98d0-9f1cdb36d291.jpg)
This is a simple visualization of the project, with the flow of the data from the S3 containers to redshift and then into another S3 container
#### Input data
The input data consists of two S3 containers containing two different datasets
##### Song Dataset
The first dataset is a subset of real data from the Million Song Dataset. Each file is in JSON format and contains metadata about a song and the artist of that song. The files are partitioned by the first three letters of each song's track ID. For example, here are filepaths to two files in this dataset.
>song_data/A/B/C/TRABCEI128F424C983.json
>song_data/A/A/B/TRAABJL12903CDCF1A.json

And below is an example of what a single song file, TRAABJL12903CDCF1A.json, looks like.
>{"num_songs": 1, "artist_id": "ARJIE2Y1187B994AB7", "artist_latitude": null, "artist_longitude": null, "artist_location": "", "artist_name": "Line Renaud", "song_id": "SOUPIRU12A6D4FA1E1", "title": "Der Kleine Dompfaff", "duration": 152.92036, "year": 0}
##### Log Dataset
The second dataset consists of log files in JSON format generated by this event simulator based on the songs in the dataset above. These simulate app activity logs from an imaginary music streaming app based on configuration settings.

The log files in the dataset you'll be working with are partitioned by year and month. For example, here are filepaths to two files in this dataset.

>log_data/2018/11/2018-11-12-events.json
>log_data/2018/11/2018-11-13-events.json

And below is an example of what the data in a log file, 2018-11-12-events.json, looks like.
![log-data](https://user-images.githubusercontent.com/36500094/134797838-10dc57e5-4dbd-4087-a283-15fc54305f44.png)


#### Database
The processed database contains 4 dimensional table users, songs, artists and time and a fact table songplays.
This is the ER schema of these tables:


<img width="570" alt="schema" src="https://user-images.githubusercontent.com/36500094/134797883-1e13cc7e-a855-427b-a996-05f933ca21d5.png">


### Project content
* etl.py : Script to connect to the S3 containers, move the data into the spark cluster, process the data and load the processed data as parquets into another S3 container
* dl.cfg : Configuration file with the AWS credentials and input/output paths (to be filled)
* data folder: A folder with a smaller sample of the input data
### Launch the project
To execute the project first fill the dl.cfg file with valid AWS credentials and a valid output S3 bucket. Then execute etl.py. If no AWS account is available leave the config file as it is and the script will be executed on local data.
