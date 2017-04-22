# Recommendation Engines

The purpose of this project was to implement content-based and hybrid recommendation methods for a question-and-answer website.

## Getting Started

The notebook is divided into 3 parts: 

1. Content-based filtering engine containing: 
	1. Simple Unary method
	2. Unit Weight method
	3. IDF method
2. Hybrid switching method
3. Hybrid method:
In this method, I used a trust-based recommendation method

Below, I will give a small description about each method

## Dataset Structure
Below is a snapshot of the structure of each of the dataframes:

#### Questions

questions	    |	Sports	        |	Books	        |	Leadership		|	Philosophy
-------------	|	-------------	|	-------------	|	-------------	|	-------------
question1	    |	1	            |	0	            |	1				|	0
question2	    |	0	            |	1				|	1				|	1
question3	    |	0	            |	0				|	0				|	1

#### Feedback
		
questions	    |	User 1			|	User 2			|	User 3			|	User 4
-------------	|	-------------	|	-------------	|	-------------	|	-------------
question1		|	1				|	-1				|					|	
question1		|	-1				|	1				|					|	
question3		|					|					|					|	

## Method Implementation
### Content-based filtering 
#### Simple Unary method

In this part, we will calculate the user profile based on his previous likes and dislikes, and then get the top 5 questions to be predicted for each user. 

To get the user profile for each topic, we have to calculate the dot product of their feedback (their likes and dislikes on specific questions) and their questions (in what topics are their questions related to)

```
def sumproductDF(df1, df2):
    sumprDF = df2.apply(lambda x: x.dot(df1), axis = 0)
    return sumprDF

sumproductDF(feedback, questions)
```

The User Profile dataframe would look something like this:

User Profile	|	Sports			|	Books			|	Leadership		|	Philosophy
-------------	|	-------------	|	-------------	|	-------------	|	-------------
User 1			|	3				|	-2				|	-1				|	0
User 2			|	-2				|	2				|	2				|	3
User 3			|	-2				|	1				|	1				|	0

Then, to get the predictions, we will apply the cosine similarity function between two vectors.

The cosine similarity function is given by: 

![picture alt](https://alexn.org/assets/img/cosine-similarity-34eaf5ab.png)

First, the below function will compute the squared sum of squares of each item in the dataframe
```
def SQSumSq(df):
    return (df**2).sum(axis=1)**0.5
```

```
Then, the below function will compute the cosine similarity between the dataframes by first calcualting the squared sum of squares for each dataframe and then applying the dot product. At the end, we will get the predictions dataframe with the computed cosine similarity for each entry. 

"""
#df1 is the users' questions dataframe
#df2 is the usersprofiles dataframe
"""

def DFPred(df1, df2):
    df1SQSumSq = SQSumSq(df1)
    df2SQSumSq = SQSumSq(df2)
    preds = df1.apply(lambda x: df2.dot(x), axis=1)
    return preds/np.outer(df1SQSumSq, df2SQSumSq)
```

The dataframe will look something like:

questions		|	User 1			|	User 2			|	User 3
-------------	|	-------------	|	-------------	|	-------------
question1		|	0.39			|	-0.298			|	-0.293
question2		|	-0.436			|	0.833			|	0
question3		|	0.252			|	0				|	-0.378


#### Unit Weight Method

Here, the only difference with regards to the method above is how we calculate the user profiles.
Instead of calculating the dot product on the questions dataframe, we will create a new dataframe called ratios, that will store the topic frequency for question. 

```
#Sum the number of topics in the questions dataframe per question, then divide the questions dataframe by the result to get the frequency for each topic
totalsUW = questions.sum(axis=1)
ratios = questions.div(totalsUW, axis = 0)

userProfileUW = sumproductDF(feedback, ratios)
```


#### IDF Method
Here, we will use tf–idf (term frequency–inverse document frequency) which lets us get the relevance of a topic. In that sense, the more the number of questions for a topic, the less it will be relevant. Topics with a low number of questions will therefore be more relevant in the final predicition.

We calculate DF (document frequency) by summing the number of questions per topic 
Then, we calculate IDF using the below formula:
![picture alt](https://wikimedia.org/api/rest_v1/media/math/render/svg/ac67bc0f76b5b8e31e842d6b7d28f8949dab7937)

We will divide the total number of questions (in our case it's 20) by the DF (document frequency) for each topic and then take the log.

```
#IDF Formula
IDF = np.log10(20/questions.sum(axis=0))
````

###Comments
The methods we have implemented above are useful if we are already aware of the user (it only works for users who have rated and asked questions). 
In order to have some recommendations for new users, we need to use other non-personalized methods, like the one I have implemented below. 

### Hybrid Switching
This method consists of 'switching' recommendation depending on the user profile. In case of an active user, we will use the IDF method we defined above. In case of a new user, we will switch to a non-personalized recommendation method.

We want to get the top 5 questions for each user baed on the hybrid switching method: 
1- We need to make sure in the case of an active user not to recommend him questions he has already rated.
2- For new users, we will recommend them the questions that had the greatest number of likes among the active users.

The full code is available in the notebook

### Hybrid Switching
In this part, I decided to use a trust-based recommender system to retrieve the top 5 questions for each user. This method works by calculating the trust between 2 users and the trust of a user based on his previous predictions in comparison with other users' predictions.

__Note__: In our example of questions and answers dataframe, we are getting the trust based on implicit trust generation methods because we do not have any data about users explicitly rating others. 

### Reference & Inspiration
The following documents helped me a lot to understand the methods for a trust-based recommendation system:
* [Trust In Recommender Sytems](https://csiweb.ucd.ie/files/Trust%20in%20Recommender%20Systems.pdf)
* [Implicit Trust Recommendation Methods](https://pdfs.semanticscholar.org/8dce/d6f630d27be6b1f759a74d2caf2a09a61842.pdf)




