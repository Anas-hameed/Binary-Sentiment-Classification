# Binary-Sentiment-Classification
In this, we will fine tune a pretrained for binary sentiment classification using a stanford movie review dataset 

dataset_url: http://ai.stanford.edu/~amaas/data/sentiment/

@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}

# Explanation:

We took the data set from Stanford regarding the binary classification of movie reviews from IMDB.
The dataset has 50,000 rows, with text, rating and sentiment values, and the model that was fine-tuned was the GPT-2.
We split the dataset to be 90% training and 10% testing, but after data cleaning, we we’re left with “40,689” training samples and “5,000” testing samples

<a href="https://imgur.com/cqcAPlO"><img src="https://i.imgur.com/cqcAPlO.png" title="source: imgur.com" /></a>

After completing training on the samples, we used the model to check a predict a few (50) samples:

<a href="https://imgur.com/SXOQxu5"><img src="https://i.imgur.com/SXOQxu5.png" title="source: imgur.com" /></a>

After these, the model was run on the test set, and all the data was put into an output.csv file, and the results are shown as below:

<a href="https://imgur.com/N3Slg1d"><img src="https://i.imgur.com/N3Slg1d.png" title="source: imgur.com" /></a>

We see all the different sentiments inside the dataset, which were changed to “0 for negative”, “1 for positive” and “-1 for Null”.
Calculating the accuracy without Null values resulted in an accuracy of 94.5%, with the confusion matrix and classification report being shown below:

<a href="https://imgur.com/JxiZlvX"><img src="https://i.imgur.com/JxiZlvX.png" title="source: imgur.com" /></a>
