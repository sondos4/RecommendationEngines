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


## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc

