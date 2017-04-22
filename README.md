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
Below is a small 

questions	    |	Sports	        |	Books	        |	Leadership		|	Philosophy
-------------	|	-------------	|	-------------	|	-------------	|	-------------
question1	    |	1	            |	0	            |	1				|	0
question1	    |	0	            |	1				|	1				|	1
question3	    |	0	            |	0				|	0				|	1

questions     | Sports
------------- | -------------
question1     | Content Cell
question1     | Content Cell

	Sports	Books	Leadership	Philosophy
question1	1	0	1	0
question2	0	1	1	1
question3	0	0	0	1

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

Then, to get the predictions, we will apply the cosine similarity function between two vectors:


```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

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

