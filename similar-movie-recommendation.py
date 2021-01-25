
"""

PROJECT BY : Akhil Kumar Dundigalla
PROJECT NAME: "Similar Movie Recomendation using Item based Collaborative Filtering"
DATASET USED: ml-1m dataset is used for this project, which is a Movie Lens dataset provided by movielens.com

Entire dataset is distributed into 3 files, they are: users.dat, ratings.dat, and movies.dat.

The movie id  = 260 was given as input, which corresponds to => Star Wars: Episode IV - A New Hope (1977) movie.

"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType
import sys


# defining a function used for creating a dictionary that maps movie id to movie name
def loadMovieNames():
    movieNames = {}

    with open("movies.dat", encoding="ISO-8859-1") as f:

        for line in f:

            fields = line.split("::")
            movieNames[int(fields[0])] = fields[1]

    return movieNames


# creating spark session
spark = SparkSession.builder.appName("MovieSimilarities-1m").getOrCreate()

# defining the schema for ratings.dat dataset
ratingSchema = StructType([StructField("userID", IntegerType(), True), \
                     StructField("movieID", IntegerType(), True), \
                     StructField("rating", IntegerType(), True), \
                     StructField("timeStamp", LongType(), True)])

# calling the above defined function 
movieNames = loadMovieNames()

# loading the ratings.dat dataset into driver program
ratingsData = spark.read.option("sep", ":").option("charset", "ISO-8859-1").schema(ratingSchema).csv("s3n://sample/pyspark-practice/ratings.dat")

# selecting the only required columns
ratings = ratingsData.select("userID", "movieID", "rating")

# partitioned the above dataframe into 100 partitions for properly spreading the data across all the executors
ratingsPartitioned = ratings.repartition(100)

# Performing Self-join on user id column to find every combination of movie pairs that each user has watched and rated.
# Also removed duplicate rows in the resulting dataframe
ratingsJoined = ratingsPartitioned.alias("ratings1").join(ratingsPartitioned.alias("ratings2"), (func.col("ratings1.userID") == func.col("ratings2.userID")) \
    & (func.col("ratings1.movieID") < func.col("ratings2.movieID")))

# Select movie pairs and rating pairs
ratingsJoinedSelected = ratingsJoined.select(func.col("ratings1.movieID").alias("movieID1"), func.col("ratings2.movieID").alias("movieID2"), \
    func.col("ratings1.rating").alias("rating1"), func.col("ratings2.rating").alias("rating2"))

# adding the extra columns that are required to calculate cosine similarity and repartitioning into 100 partitions before grouping.
pairScores = ratingsJoinedSelected.withColumn("x2", func.col("rating1") * func.col("rating1")).withColumn("y2", func.col("rating2") * func.col("rating2")) \
    .withColumn("xy", func.col("rating1")*func.col("rating2")).repartition(100)

# grouping of all the rows having same movie pairs and performing the necessary aggregations
pairScoreAggregate = pairScores.groupBy("movieID1", "movieID2").agg(func.sum(func.col("xy")).alias("numerator"), \
    (func.sqrt(func.sum(func.col("x2"))) * func.sqrt(func.sum(func.col("y2")))).alias("denominator"), func.count(func.col("xy")).alias("noSamples"))

# calculating the cosine similarity for every unique movie pairs 
cosineSimilarity = pairScoreAggregate.withColumn("cosineSimilarityScore", func.when(func.col("denominator") != 0, \
    func.col("numerator") / func.col("denominator")).otherwise(0)).select("movieID1", "movieID2", "cosineSimilarityScore", "noSamples").cache()

if len(sys.argv) > 1:
    # setting the threshold values
    scoreThreshold = 0.97  # the minimum threshold for cosine similarity value
    noSamplesThreshold = 100 #the minimum threshold for the number of samples to be considered into result.
    
    # reading and assigning the selected movie id for which top 10 similar movies are calculated.
    inputMovieID = int(sys.argv[1])

    # Now filtering out only the movies that qualify the above two mentioned thresholds.
    resultFiltered = cosineSimilarity.filter(((func.col("movieID1") == inputMovieID) | (func.col("movieID2") == inputMovieID)) \
        & (func.col("cosineSimilarityScore") > scoreThreshold) & (func.col("noSamples") > noSamplesThreshold))
    
    # now sorting the results by similarity score in descending order and then selecting the top 10 results
    resultSorted = resultFiltered.sort(func.col("cosineSimilarityScore").desc()).cache()
    
    # writing the results into csv file
    resultSorted.repartition(1).write.format('csv').option('header',True).mode('overwrite').option('sep',',').save('s3n://sample/pyspark-practice/output')

    # getting back the results into the driver program
    results = resultSorted.take(10)

    
    print("\n\n")
    print("Top 10 similar movies for " + movieNames[inputMovieID])
    print("\n")

    # Now displaying top 10 similar movies for the given input movie id 
    for result in results:
        similarMovieID = result.movieID1
        if similarMovieID == inputMovieID :
            similarMovieID = result.movieID2
        
        # printing to top 10 similar movies
        print(movieNames[similarMovieID] + "\tSimilarityScore: " + str(result.cosineSimilarityScore) + "\tSamples: " + str(result.noSamples))
    print("\n\n")
        

spark.stop()











    