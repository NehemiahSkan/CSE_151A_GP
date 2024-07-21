# Milestone 3
## Preprocessing
To clean up the data, we got rid of some features that we decided we were not going to use, such as url, author, token_count.
We also decided to use only english comments so we used the language feature to remove the rows that were not classified as english.
We removed punctuation from our data and also made all the comments lowercase. 
We removed common words like it, the, a, and, to for the purpose of finding patterns with the words in the comments. These words are known as stopwords.

## Questions
### Where does your model fit in the fitting graph?
The linear regression model has a decent fit for lower scores that are close to zero. At these low scores, the model predicts well since the points are close to the generated diagonal line. However, with higher values, the model has a very poor fit and the predicted values deviate significant from the actual scores. Thus, there is a negative R-squared which indicates poor generalizability for the given model.

### What are the next models you are thinking of and why?
The next model that we would like to consider is an RNN or a CNN to create a better predictive model based on the nature of the keyword, the context of the subreddit, and the upvote/downvote score.

## Data Exploration
[Data Source](https://huggingface.co/datasets/OpenCo7/UpVoteWeb)

We made a chart of some frequently used words in comments and the scores they had. We used a smaller version of the dataframe for our initial data exploration due to the sheer size of our data set.

We started by exploring the most common words.
![word_freq](DataCleaning/graphs/word_frequencies_from_640k_samples.png)

We then discovered some more insights on threads with the most "enthusiastic" comment participation.
![score_per_comment](DataCleaning/graphs/score_per_comment_by_subreddit_from_640k_samples.png)

We also looked at the relation of score with words as a different overall metric.
![score_and_words](DataCleaning/graphs/score_and_words.png)