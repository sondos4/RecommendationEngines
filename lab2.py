import pandas as pd
import numpy as np


xl = pd.ExcelFile('exerciseCB.xlsx')
sheets = xl.sheet_names


"""
CB - Simple Unary Analysis
"""

xlSU = xl.parse(sheets[0], skiprows=1)
suDF = pd.DataFrame(xlSU)

"""
Same Dataframes to be used along all exercises
"""
questions = suDF.iloc[:20, :11].rename(columns={"Unnamed: 0": "questions"}).set_index('questions')
userFeedback = suDF.iloc[:20, 14:18].fillna(0)
questsTitles = suDF.iloc[:20, 0]
feedback = pd.concat([questsTitles, userFeedback], axis=1).rename(columns={"Unnamed: 0": "questions"}).set_index('questions')
userAnswers =  suDF.iloc[:20, 19:23].fillna(0)
predictions =  suDF.iloc[:20, 24:28]
userProfile = suDF.iloc[25:29, :11].rename(columns={"Unnamed: 0": "User Profile"}).set_index('User Profile')
questionLikes = pd.concat([questsTitles, userAnswers], axis = 1).rename(columns={"Unnamed: 0": "questions"}).set_index('questions')



"""
Functions to be used throughout the exercise
"""

"""
Function for Sum Product
df1 is the users' Feedback dataframe
df2 is the users' questions dataframe
"""

def sumproductDF(df1, df2):
    sumprDF = df2.apply(lambda x: x.dot(df1), axis = 0)
    return sumprDF


"""
Function to compute the square sum of squares
"""
def SQSumSq(df):
    return (df**2).sum(axis=1)**0.5

"""
Function for the predictions dataframe
df1 is the users' questions dataframe
df2 is the users' usersprofiles dataframe
"""

def DFPred(df1, df2):
    df1SQSumSq = SQSumSq(df1)
    df2SQSumSq = SQSumSq(df2)
    preds = df1.apply(lambda x: df2.dot(x), axis=1)
    return preds/np.outer(df1SQSumSq, df2SQSumSq)

"""
Function to calculate the number of likes, dislikes and neutral votes
df is the predictions dataframe
"""
def likes(df):
    likesdf = pd.DataFrame(index=['likes', 'dislikes', 'neutral'], columns=df.columns)
    for i in likesdf.index:
        if i == 'likes':
            likesdf.loc[i] = [(df[c] > 0).sum() for c in df.columns]
        elif i == 'dislikes':
            likesdf.loc[i] = [(df[c] < 0).sum() for c in df.columns]
        else:
            likesdf.loc[i] = [(df[c] == 0).sum() for c in df.columns]
    return likesdf


"""
CB - SIMPLE UNARY
"""

userProfileSU = sumproductDF(feedback, questions)

predictionsSU = DFPred(questions, userProfileSU)

likesSU = likes(predictionsSU)


"""
CB - Unit Weight
"""

xlUW = xl.parse(sheets[1], skiprows=1)

unitw = pd.DataFrame(xlUW)

totalsUW = questions.sum(axis=1)

ratios = questions.div(totalsUW, axis = 0)

userProfileUW = sumproductDF(feedback, ratios)

predictionsUW = DFPred(questions, userProfileUW)

likesUW = likes(predictionsUW)


"""
CB - IDF
"""
xlIDF = xl.parse(sheets[2], skiprows=1)
cbIDF = pd.DataFrame(xlIDF)
userProfileIDF = cbIDF.iloc[25:29, :11].rename(columns={"Unnamed: 0": "User Profile"}).set_index('User Profile')

IDF = np.log10(20/questions.sum(axis=0))

userProfileIDF = userProfileUW * IDF

predictionsIDF = DFPred(questions, userProfileIDF)

likesIDF = likes(predictionsIDF)


"""
HYBRID SWITCHING
"""

"""
Get the list of unanswered questions per user
"""
def unAnsweredQ(df, col):
    nonAnswered = df[col] == 0
    return df.loc[nonAnswered, col].index

"""
Get the list of answered questions per user
"""
def answeredQ(df, col):
    answered = df[col] != 0
    return df.loc[answered, col].index

"""
Given two dataframes, return that top 5 non common movies
df is the predictions dataframe with the
df1 is the dataframe with all the unanswered questions per all users
the column order is the same in both dataframes
"""
def nonCommon(df, df1, c):
    if len(df1) > 1:
        nonComDF = df.loc[df[c].isin(df1[df.columns.get_loc(c)]), c]
    else:
        nonComDF = df.loc[df[c].isin(df1[0]), c]
    nonComDF.reset_index(drop=True, inplace=True)
    return nonComDF

"""
Method to sort the questions by the number of likes
"""
def genericTop(df = questionLikes):
    sums = pd.DataFrame(df.sum(axis=1))
    top = sums.sort_values(0, ascending=False).index
    return top


"""
Mehtod to predict the top 5 questions per user depending on their profile:
If it is a new user - the generic method will be applied (top 5 questions will the ones with the highest number of likes)
If it is an old user, we will be based on the prediction done in CB IDF (the highest predicted AND not already rated will be displayed)
"""
def top5PerUser(df, predictionsdf):
    top5 = pd.DataFrame(columns = predictionsdf.columns)
    unAnswered = [unAnsweredQ(df, col) for col in df.columns]
    predictions = predictionsdf.apply(lambda x: x.sort_values(ascending=False).index, axis =0)
    for c in predictions.columns:
        if predictionsdf[c].isnull().all():
            userTop5 = genericTop()
        else:
            userTop5 = nonCommon(predictions, unAnswered, c)
        top5[c] = userTop5[:5]
    return top5


predictionsSwitching = top5PerUser(feedback, predictionsIDF)


"""
HYBRID CHALLENGE
"""

userCorrs = feedback.corr()
predicitionsDF = pd.DataFrame(index = feedback.index, columns=feedback.columns)


"""
Method to get the common questions between two users
"""
def commonQuests(user1, user2):
    AU1 = pd.DataFrame(answeredQ(feedback, user1), columns=['questions'])
    AU2 = pd.DataFrame(answeredQ(feedback, user2), columns=['questions'])
    common = pd.merge(AU1, AU2, on='questions', how='inner')
    return common

"""
Method to create a dataframe for a specific user with the predictions made by him (only applicable to questions that were also rated by other users)
This will later be used to calculate the trust of the user's ratings
"""

def trustItemsPerUser(user, df = feedback):
    trustItemsDF = pd.DataFrame()
    #for every user
    for u in df.columns:
        #For all users different than this user
        if u!= user:
            com = commonQuests(user, u)
            listcom = list(com['questions'])
            expected = df[df.index.map(lambda x: x in listcom)][[user, u]]
            expected[user] = [averageUser(u) + (expected.loc[r, user] - averageUser(user)) for r in expected.index]
            trustItemsDF = pd.concat([trustItemsDF, expected])
    return trustItemsDF


"""
Method to calculate the trust for an item i and a user u.
For the item, we will compare the predicted value with all the other ratings by the other users for the same item.
If the difference between the prediction and the actual rating of the item is between -0.5 and 0.5, we will set the trust to be 1.
Else it will be -1.
We will them sum the trust for each difference for this item and divide it by the number of attempted predictions.

If the item is not in the list of items that can be predicted by user u, the trust will be 0.
"""
def trustPerItem(user, item, df = feedback):
    trustItems = trustItemsPerUser(user, df)
    if item in trustItems[user].index:
        itemDF = trustItems.drop(user, axis=1).loc[item].dropna().values
        predsByUser = trustItems[user][item]
        difference = [predsByUser - itemDF if i else '' for i in itemDF]
        trustvalues = [1 if (diff >= -0.5) & (diff <= 0.5) else -1 for diff in difference]
        trust = np.sum(trustvalues) / len(trustvalues)
    else:
        trust = 0
    return trust


"""
Method to create a DF with all the possible item predictions by a user.
For items that have been rated by other users, we will use the method above to calculate the trust
Then, for the items rated by user but not rated by anyone else, we will give them a trust value = average trust per user
"""
def trustDFPerUser(user, df = feedback):
    answeredUser = answeredQ(df, user)
    trustDF = pd.DataFrame(index=[answeredUser], columns=['trust'])
    for i in trustDF.index:
        trustDF.loc[i] = trustPerItem(user, i)
    trustDF = trustDF.apply(lambda x: trustDF.mean() if x.all() == 0 else x, axis=1)
    return trustDF


"""
Function to get the trust value for an item
"""
def findTrust(user, item):
    df = trustDFPerUser(user)
    if item in df.index:
        trust = df.loc[item]['trust']
    else:
        trust = 0
    return trust

"""
Function to calculate the weight of the prediction of an item between user 1 (receiving the prediction) and user 2 (giving the prediction)
The weight will be equal to 2 * trust(p,i) * similarity / similarity + trust(p,i)
"""
def weight(user1C, user2P, item, userCorrelations = userCorrs):
    trustP = findTrust(user2P, item)
    similarity = userCorrelations[user1C][user2P]
    w = 2*trustP*similarity/(similarity+trustP)
    return w


def unAnsweredPerUser(df, col):
    nonAnswered = df[col] == 0
    return df.loc[nonAnswered, col].index


"""
Method to calculate the average rating given by a user
"""
def averageUser(user, df = feedback):
    meanRating = df.loc[df[user] != 0, user].mean()
    return meanRating


"""
Method to build the trust-based predictions for a user.
The prediction will follow the trust-based weighting formula.
This function will only apply to the questions that were not rated by the user.
we will follow the below methodology:
1- get the list of questions to predict (unanswered by user)
2- for each question in the dataframe we will:
    1- For each user who rated this question, get the needed parameters to apply the trust-based weight formula.

The dataframe returned by this function will contain an empty entry for each question that was not rated by any user.
"""
def buildPredictions(user, preds = predicitionsDF, df = feedback):
    unAnswered = [unAnsweredQ(df, user)]
    predsdf = preds.apply(lambda x: x.index, axis=0)
    questslist = list(nonCommon(predsdf, unAnswered, user))
    toPredictDf = pd.DataFrame(index=questslist, columns=['predictions'])
    feedbackNoUser = feedback.drop(user, axis=1)
    for q in toPredictDf.index:
        predictedNom = np.empty(0)
        predDenom = np.ones(0)
        for u in feedbackNoUser.columns:
            if feedbackNoUser.loc[q][u] != 0:
                denom = weight(user, u, q)
                nom = (feedbackNoUser.loc[q][u] - averageUser(u)) * denom
                predictedNom = np.append(predictedNom, nom)
                predDenom = np.append(predDenom, denom)
        toPredictDf.loc[q] = averageUser(user) + np.sum(predictedNom) / np.sum(np.abs(predDenom))
    return toPredictDf


"""
Method to combine the CB IDF and trust-based filtering
For all questions that were not predicted by the trust-based method, we will get the predicted value by the CB IDF method.
Get the top 3 from the trust based predictions, and the top 2 from the CB IDF predictions, and combine them in one dataframe.
"""
def combinePreds(df1, df2, user):
    if not df1['predictions'].isnull().all():
        indices = df1.loc[df1['predictions'].isnull()].index
        df2loc = pd.DataFrame(df2[user].loc[df2[user].index.isin(indices)])
        df2loc = df2loc.rename(columns={df2loc.columns[0]: "predictions"})
        top3df1 = df1.apply(lambda x: x.sort_values(ascending=False).index, axis=0)[:3]
        top2df2 = df2loc.apply(lambda x: x.sort_values(ascending=False).index, axis=0)[:2]
        predictionsDF = pd.concat([top3df1,top2df2], axis = 0).reset_index(drop=True)
    else:
        predictionsDF = top5PerUser(df1, df2)[[user]]
    return predictionsDF


u1 = buildPredictions('User 1')
u2 = buildPredictions('User 2')
u3 = buildPredictions('User 3')
u4 = buildPredictions('User 4')

predictionsUser1 = combinePreds(u1, predictionsIDF, 'User 1')
predictionsUser2 = combinePreds(u2, predictionsIDF, 'User 2')
predictionsUser3 = combinePreds(u3, predictionsIDF, 'User 3')
predictionsUser4 = combinePreds(u4, predictionsIDF, 'User 4')

"""
Predictions for User 4
"""
for i in predicitionsDF.columns:
    for q in predicitionsDF.index:
        predicitionsDF.loc[q][i] = findTrust(i, q)


pred4 = pd.DataFrame(genericTop(predicitionsDF))