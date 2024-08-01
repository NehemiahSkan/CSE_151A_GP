## Results
### RNN Model 1 (Score Prediction)
For the first RNN score prediction model, we were ultimately unable to yield a good prediction result. With a moderately sized model, we were able to only achieve a test MSE of 6337, and an MAE of 14.08.

We only ran the training for one epoch after seeing the loss not changing significantly towards the end of the epoch.

```
Training MAE: 14.323173522949219
Training MSE: 8630.5068359375
Test MAE: 14.083123207092285
Test MSE: 6337.0185546875
```
![actual vs predict](/DataCleaning/graphs/small_RNN_pred_vs_actual.png)

### RNN Model 2 (Subreddit Classification)
For the RNN model for subreddit classification, we used a fairly complex model and was able to get decent result for the predictions of from which of the top 10 subreddits the comments came from.

![report](/DataCleaning/graphs/classification_report.png)

We got an overall accuracy on our classification predictions of 0.55. The training only ran for 2 epochs, where we used early_stopping to make sure that the model didn't overfit.

![loss graph](/DataCleaning/graphs/test_train_loss.png)

## Discussion
### RNN Model 1 (Score Prediction)
For this first model, I believe that the main issue that it had was the simple fact that most comments do not receive a lot of upvotes. In hindsight, we should have realized that this would be an issue during our data cleaning and exploration.

```
               score
count  600000.000000
mean       10.697337
std        81.798644
min      -448.000000
25%         1.000000
50%         2.000000
75%         6.000000
max     18085.000000
```

We had found that even at the 75 percentile, the upvote count was still at only 6, which meant that an overwhelming amount of comments were between 0 and 6 upvotes. However, we were still trying to use it to predict those special comments that would've gotten the 18085 upvotes. However, for our dataset, those comments are essentially outliers, which would be impossibly hard to make the model predict. Thus, predictably, our model settled on making very low upvote count predictions for most comments and chose to ignore the outliers, as can be seen from the graph below:

![actual vs predict graph](/DataCleaning/graphs/small_RNN_pred_vs_actual.png)

In conclusion, for this model, we tried to make a model predict something impossible. Which is the question of, out of millions of comments, which ones are going to go viral. If our model worked sucessfully, it probably would've been able to be used to make a lot of money.

### RNN Model 2 (Subreddit Classification)
For this classification model, we ended up with rather good results. Out of the 10 possible choices for subreddits, the model was able to select the right one 55% of the time.

![report](/DataCleaning/graphs/classification_report.png)

From the classification report, we can isolate some of the individual subreddits to see what may have cause the difference in precision and recall for them. 

Firstly, we can see that one of the best performing subreddits was CFB (College football). Firstly, we ended up with 16184 comments out of the total 62860 for the top 10 subreddits, meaning it stood for about 25.7% of our data, which may have helped predictions for it, but clearly these comments also had very distinctive characteristics that was able to make the precision go up to 0.77.

After browsing through the subreddit, I found that there were many comments of well written theories about players and teams. These long comments were not very present in other subreddits, and as a result, made them stand out a lot more, and able to be identified by the RNN. Another main identifier for these comments may have been the player and team names that get mentioned in the comments.

On the other hand, we can also see there are those such as the NoStupidQuestions subreddit which had an abysmal precision score of 0.04. This is most likely due to the much less focused topics of this subreddit. Whereas all the comments in the CFB subreddit were mainly about football, this subreddit allowed for questions of any kind from any domain. As a result, the comments in this subreddit have no clearly distinctive formats or topics, rendering it very difficult to predict whether or not a comment came from this subreddit or not.

