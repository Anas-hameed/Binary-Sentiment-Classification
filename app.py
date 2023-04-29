# import dependencies
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# check length of the testing file
def checkLength(data):
    length=0
    for index, row in data.iterrows():
        if len(row['text'].split())<460:
            length+=1
    print("Total Testing Length: ", length)
    return length


# Read the testing file
def read_testingFile(path='testing/results'):
    nameList= [312,  626,  939,1252, 1565, 1878, 2191,2504, 2817, 3130, 3443,3756, 4069, 4382, 4695,5000]
    for name in nameList:
        fileName= "{}/output_{}.csv".format(path, name)
        # create a single csv file
        df = pd.read_csv(fileName)
        df.to_csv('testing/output.csv', mode='a', header=False, index=False)
        print("File: ", fileName)



def PrintMatrix():
    file_path= "testReview.csv"
    df = pd.read_csv(file_path, header=None)
    # delete the zeroth row with null values
    df.columns = ['index', 'text', 'sentiments']
    df = df.drop([0])
    length= checkLength(df)

    
    # read a testing file
    file_path= "testing/output.csv"
    df = pd.read_csv(file_path)
    df.columns=['review', 'predictedReview', 'ActualSentiment', 'predictedSentiment']

    # unique sentiments
    print("Unique sentiments: ", df['predictedSentiment'].unique().tolist())

    # replace column with positive to 1 and negative to 0
    df['predictedSentiment'] = df['predictedSentiment'].replace([' positive ', ' positive _END', ' positive _', ' positive'], 1)
    df['predictedSentiment'] = df['predictedSentiment'].replace([' negative ', ' negative _', ' negative'], 0)
    # replace nan with -1
    df['predictedSentiment'] = df['predictedSentiment'].fillna(-1)


    # unique sentiments
    print("Unique sentiments: ", df['predictedSentiment'].unique().tolist())



    # print the accuracy by comparing the predicted sentiment with the actual sentiment
    print("Accuracy: ", ((df['ActualSentiment'] == df['predictedSentiment']).sum()/length) * 100)

    print("----------------------------------------------------------------------------------------------------")
    # Accuracy without considering the null values as predicted sentiment
    print("Accuracy without null values: ", ((df['ActualSentiment'] == df['predictedSentiment']).sum()/len(df)) * 100)


    # import confusion matrix 
    # replace -1 with 1, as there is only 1 null value predicted by the model where expected was 0 or negative
    df['predictedSentiment'] = df['predictedSentiment'].replace(-1, 1)

    # print the x-y labels
    print("####################################################################################################")
    print("Confusion matrix")
    print("----------------------------------------------------------------------------------------------------")
    print(confusion_matrix(df['ActualSentiment'], df['predictedSentiment']))
    # print(confusion_matrix(df['ActualSentiment'], df['predictedSentiment']))


    # print the precision, recall and f1-score
    print("####################################################################################################")
    print("Classification report")
    print(classification_report(df['ActualSentiment'], df['predictedSentiment']))






if __name__ == '__main__':
    PrintMatrix()


  


