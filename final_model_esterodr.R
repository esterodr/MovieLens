### This file is part of the MovieLens Project for the last
### course of the EDX/HarvardX Data Science Specialization
###
### Important Notes:
###
### Here is just the final model. The explanations and the 
### tuning of the parameters are in the report:
### "movielens_esterodr.pdf" or "movielens_esterodr.Rmd".
### Both files can be found in the Github repository:
###
### https://github.com/esterodr/MovieLens
###
### If you are going to reproduce this code, I strongly
### recommend you to do it in more than one session and
### saving the important files in your local drive, as is
### indicated in the report.
###
### The execution of the code takes time and could be
### intensive in the use of computing resources
###
### If you already have the "edx" and "validation" dataframes,
### save them in an object "edx.Rda", or the code will try
### to download the files again.

### Construction of the datasets

if (file.exists("edx.Rda")) {
  load("edx.Rda")
} else {
  # The following code was provided by the instructors:
  
  #############################################################
  # Create edx set, validation set, and submission file
  #############################################################
  
  # Note: this process could take a couple of minutes
  
  if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
  if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
  
  # MovieLens 10M dataset:
  # https://grouplens.org/datasets/movielens/10m/
  # http://files.grouplens.org/datasets/movielens/ml-10m.zip
  
  dl <- tempfile()
  download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
  
  ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                        col.names = c("userId", "movieId", "rating", "timestamp"))
  
  movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
  colnames(movies) <- c("movieId", "title", "genres")
  movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                             title = as.character(title),
                                             genres = as.character(genres))
  
  movielens <- left_join(ratings, movies, by = "movieId")
  
  # Validation set will be 10% of MovieLens data
  
  set.seed(1)
  test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
  edx <- movielens[-test_index,]
  temp <- movielens[test_index,]
  
  # Make sure userId and movieId in validation set are also in edx set
  
  validation <- temp %>% 
    semi_join(edx, by = "movieId") %>%
    semi_join(edx, by = "userId")
  
  # Add rows removed from validation set back into edx set
  
  removed <- anti_join(temp, validation)
  edx <- rbind(edx, removed)
  
  rm(dl, ratings, movies, test_index, temp, movielens, removed)
  
  ## The next line was introduced by me to avoid creating the datasets every time
  save(edx, validation, file = "edx.Rda")
  
}

### Loading packages

library(tidyverse)
library(caret)

### I will save the number of rating per user
### and the number of ratings received by each movie
### for future use:

ratings_per_user <- edx %>%
  group_by(userId) %>%
  summarise(ratings_per_user = n())
ratings_per_movie <- edx %>%
  group_by(movieId) %>%
  summarise(ratings_per_movie = n())

### Function for estimate the RMSE of the predictions

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

### Baseline model + time Bins in item-effects
### See the report for justifications and tunning

## Mean of all the ratings
mu <- mean(edx$rating)

## Item effect
b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+2))

## Convert the variable "timestamp" to daily frequency
edx$time_day <- as.Date(as.POSIXct(edx$timestamp , origin="1970-01-01"))
validation$time_day <- as.Date(as.POSIXct(validation$timestamp , origin="1970-01-01"))

## Range of the dates of ratings, to calculate the Bins
l1 <- min(edx$time_day)
l2 <- max(edx$time_day)

## 6 bins from min(date) to max(date)
bins <- seq(l1, l2, length.out = 6)

## One aditional term to the item-effect for each bin
b_it <- edx %>%
  left_join(b_i, by="movieId") %>%
  mutate(interval = findInterval(time_day, bins)) %>%
  group_by(movieId, interval) %>%
  summarise(b_it = sum(rating-mu-b_i)/(n()+2), b_i=first(b_i)) %>%
  ungroup() %>%
  mutate(b_it = b_i + b_it) %>%
  select(-b_i)

## Delete unnecesary variables
rm(b_i, l1, l2)

## User effect
b_u <- edx %>%
  mutate(interval = findInterval(time_day, bins)) %>%
  left_join(b_it, by=c("movieId", "interval")) %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_it - mu)/(n()+5))


### Matrix factorization

## Resid after the Baseline model
## Considering only movies with at least 1500 ratings and
## users with at least 100 ratings.
resid <- edx %>%
  left_join(ratings_per_movie, by="movieId") %>%
  left_join(ratings_per_user, by="userId") %>%
  filter(ratings_per_movie >= 1500) %>%
  filter(ratings_per_user >= 100) %>%
  left_join(b_u, by="userId") %>%
  mutate(interval = findInterval(time_day, bins)) %>%
  left_join(b_it, by = c("movieId", "interval")) %>%
  replace_na(list(b_it=0)) %>%
  mutate(resid = rating - mu - b_it - b_u)

## Resid as a matrix
resid_matrix <- resid %>% 
  select(userId, movieId, resid) %>%
  spread(movieId, resid) %>%
  as.matrix()

## Delete the resid dataframe to save memory
rm(resid)

## The first column of the matrix are the userIds, so
## I move them to rownames and delete the first column
rownames(resid_matrix)<- resid_matrix[,1]
resid_matrix <- resid_matrix[,-1]

## Is not necessary, but I'll put movie titles as
## column names
movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()
colnames(resid_matrix) <- with(movie_titles, title[match(colnames(resid_matrix), movieId)])

## The empty spaces of the matrix are set to 0
resid_matrix[is.na(resid_matrix)] <- 0

## Calculate Principal Components
## The excecution could take time
pca <- prcomp(resid_matrix, center = TRUE, scale = FALSE)

## Calculate the cumulative variance explained by the Principal Components
var_explained <- cumsum(pca$sdev^2/sum(pca$sdev^2))

## As explained in the report, I will only use the Principal Components
## that explains at least the 15% of the variance of the resids matrix.
## (37 Principal Components)
pc.used <- which(var_explained>=0.15)[1]

## Aproximate the resid matrix with the 37 Principal Components
resid_pc <- pca$x[,1:pc.used] %*% t(pca$rotation[,1:pc.used])

## As the pca was run with the parameter "center = TRUE", I need to
## recenter the matrix again
resid_pc <- scale(resid_pc, center = -1 * pca$center, scale=FALSE)

## Convert the resid_pc matrix to a dataframe that can be "leftjoined"
## to the other factors of the model.
pc_df <- as.data.frame(resid_pc)
pc_df$userId <- as.numeric(row.names(pc_df))
pc_df <- pc_df %>%
  gather(title, b_pc, -userId)

## Join all the parameters to predict the ratings for the "validation" set
predicted_ratings <-
  validation %>%
  left_join(b_u, by="userId") %>%
  mutate(interval = findInterval(time_day, bins)) %>%
  left_join(b_it, by = c("movieId", "interval")) %>%
  left_join(pc_df, by=c("userId", "title")) %>%
  replace_na(list(b_it=0, b_pc=0)) %>%
  mutate(pred = mu + b_it + b_u + b_pc) %>%
  .$pred

### Print the RMSE of the prediction

print("The RMSE of the predictions on the validation set is:")
RMSE(validation$rating, predicted_ratings)

### The End
