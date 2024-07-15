# Milestone 2
## Preprocessing
To clean up the data, we got rid of some features that we decided we were not going to use, such as url, author, token_count.
We also decided to use only english comments so we used the language feature to remove the rows that were not classified as english.
We removed punctuation from our data and also made all the comments lowercase. 
We removed common words like it, the, a, and, to for the purpose of finding patterns with the words in the comments. These words are known as stopwords.
## Data Exploration
[Data Source](https://huggingface.co/datasets/OpenCo7/UpVoteWeb)
We made a chart of some frequently used words in comments and the scores they had. We used a smaller version of the dataframe for our initial data exploration due to the sheer size of our data set.


![word_freq](DataCleaning\graphs\word_frequencies_from_640k_samples.png)

![score_per_comment](DataCleaning\graphs\score_per_comment_by_subreddit_from_640k_samples.png)

![score_and_words](DataCleaning\graphs\score_and_words.png)