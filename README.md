# Similar-Movie-Recommendation

PROJECT BY : Akhil Kumar Dundigalla
PROJECT NAME: "Similar Movie Recomendation using Item based Collaborative Filtering"

This is my PySpark dataframe project to perform similar movie recommendation system using item based collaborative filtering.

In Item based collaborative filtering technique, the similarity is found between items i.e; similarity between movies based on ratings provided by users. I used Cosine Similarity index as the similarity index to calculate the similarities between every pairs of movies. 

Here i found every pair of movies that were watched by same person. Then i measured the similarity of the ratings of each pair of movies across all users who watched and rated the same pair of movies. Generally if we take a movie pair that a person watched together and compare it with how everyone else rated that pair of movies, we can compute how similar those two movies are to each other based on aggregated user behaviour.

DATASET USED: ml-1m dataset is used for this project, which is a Movie Lens dataset provided by movielens.com

Entire dataset is distributed into 3 files, they are: users.dat, ratings.dat, and movies.dat.

The movie id  = 260 was given as input, which corresponds to => Star Wars: Episode IV - A New Hope (1977) movie.

The results of the project are saved in output.csv file, which has four columns of output in it. The 4 columns gives the information about movie id 1 , movie id 2, Cosine similarity score, and no of samples respectively for each pair of movies which qualify the thresholds set in the program. The output in this file are not sorted.

Finally the results were sorted in descending order of similarity score and top 10 similar movies were displayed as output in EMR terminal. The screenshot of this output is also uploaded in this repository with the name similar-movie-recommendation-output-screenshot.png


