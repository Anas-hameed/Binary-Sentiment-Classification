# There are two top-level directories [train/, test/] corresponding to
# the training and test sets. Each contains [pos/, neg/] directories for
# the reviews with binary labels positive and negative. Within these
# directories, reviews are stored in text files named following the
# convention [[id]_[rating].txt] where [id] is a unique id and [rating] is
# the star rating for that review on a 1-10 scale. For example, the file
# [test/pos/200_8.txt] is the text for a positive-labeled test set
# example with unique id 200 and star rating 8/10 from IMDb. The
# [train/unsup/] directory has 0 for all ratings because the ratings are
# omitted for this portion of the dataset.

# import libraries
import os
import pandas as pd
import numpy as np


# create a csv single csv file with three columns text, rating, sentiments
def create_csv():
    path= 'aclImdb'
    train_path = os.path.join(path, 'train')
    test_path = os.path.join(path, 'test')
    train_pos_path = os.path.join(train_path, 'pos')
    train_neg_path = os.path.join(train_path, 'neg')
    test_pos_path = os.path.join(test_path, 'pos')
    test_neg_path = os.path.join(test_path, 'neg')
    
    train_pos_files = os.listdir(train_pos_path)
    train_neg_files = os.listdir(train_neg_path)
    test_pos_files = os.listdir(test_pos_path)
    test_neg_files = os.listdir(test_neg_path)

    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []

    print("Looping through the files...")
    x=0
    # get the rating from the file name and text from the file
    for file in train_pos_files:
        with open(os.path.join(train_pos_path, file), 'r', encoding='utf-8') as f:
            train_pos.append([f.read(), file.split('_')[1].split('.')[0], 1])
        
    print("Done with the loop on train positive...")

    for file in train_neg_files:
        with open(os.path.join(train_neg_path, file), 'r', encoding='utf-8') as f:
            train_neg.append([f.read(), file.split('_')[1].split('.')[0], 0])
    
    print("Done with the loop on train negative...")

    for file in test_pos_files:
        with open(os.path.join(test_pos_path, file), 'r', encoding='utf-8') as f:
            test_pos.append([f.read(), file.split('_')[1].split('.')[0], 1])
    
    print("Done with the loop on test positive...")

    for file in test_neg_files:
        with open(os.path.join(test_neg_path, file), 'r', encoding='utf-8') as f:
            test_neg.append([f.read(), file.split('_')[1].split('.')[0], 0])

    print("Done with the loop on test negative...")

    # append all the lists together
    train_pos.extend(train_neg)
    train_pos.extend(test_pos)
    train_pos.extend(test_neg)
    print("Done with the loop on all the files...")
    print(len(train_pos))
    

    # shuffle the list to get a random data
    np.random.shuffle(train_pos)
    
    # create a dataframe
    df = pd.DataFrame(train_pos, columns=['text', 'rating', 'sentiments'])
    df.to_csv('imdb.csv', index=False)


if __name__ == '__main__':
    # create_csv()

    file_path = "/kaggle/input/imdb-movies-review/imdb.csv"
    df = pd.read_csv(file_path, header=None)
    df.columns = ['text', 'rating', 'sentiments']
    df = df.drop('rating', axis=1)
    df = df.sample(50000, random_state=1)

    # 1001 row print only

    # print the unique value in sentiments columns with count
    print(df['sentiments'].value_counts())

    # print type of sentiments column
    print(df['sentiments'].dtype)
    # drop the rows with count less than 10
    df = df.groupby('sentiments').filter(lambda x: len(x) > 10)

    sess = gpt2.start_tf_sess()
    # kill the session
    gpt2.reset_session(sess)

    # dataframe 1000th row print only
    print(df.iloc[1000])