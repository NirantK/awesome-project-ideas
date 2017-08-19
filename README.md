# Deep Learning Project Ideas
A list of practical Deep Learning projects

These projects are divided in multiple categories, same problem may appear in more than one categories. 

Problems are motivated by the ones shared at:

* [CMU Machine Learning](http://www.cs.cmu.edu/~./10701/projects.html)
* [Stanford CS229 Machine Learning Projects](http://cs229.stanford.edu/)

## Table of Contents

- [Text](https://github.com/NirantK/awesome-project-ideas#text) - include NLP tasks here for now
- [Forecasting](https://github.com/NirantK/awesome-project-ideas#forecasting) - mostly Time Series and similar forecasting challenges
- [Recommender Systems](https://github.com/NirantK/awesome-project-ideas#recommender-systems)
- [Vision](https://github.com/NirantK/awesome-project-ideas#vision) - includes image and video processing

Text
---------
**Autonomous Tagging of Stack Overflow Questions** 
- Make a multi-label classification system that automatically assigns tags for questions posted on a forum such as Stackoverflow or Quora. 
- Find [StackLite](https://www.kaggle.com/stackoverflow/stacklite) or [10% sample](https://www.kaggle.com/stackoverflow/stacksample) 

**Keyword/Concept Identification** 
- Identify keywords from millions of text questions
- Find the [data on Kaggle](https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/data)

### Natural Language Understanding
**Automated Essay Grading** 
- The purpose of this project is to implement and train machine learning algorithms to automatically assess and grade essay responses. 
- [Dataset](https://www.kaggle.com/c/asap-aes/data) sponsored by Hewlett Foundation on Kaggle

**Sentence to Sentence Semantic Similarity** 
- Can you identify question pairs that have the same intent? Or sentences that have the same meaning? 
- [Quora Question Pairs Dataset](https://www.kaggle.com/c/quora-question-pairs/data)

**Open Domain Question Answering** 
- Can you build a bot which answers questions according to the student's age or her curriculum? 
- [Facebook's FAIR](https://github.com/facebookresearch/DrQA) built similar for Wikipedia 
- Dataset: [NCERT books](https://www.github.com/NirantK/ncert) for K-12/school  students in India

**Copy Writing Style**
- Generate plausible new text which looks like some other text
- Obama Speeches? For instance, you can create a bot which writes [new speeches in Obama's style](https://medium.com/@samim/obama-rnn-machine-generated-political-speeches-c8abd18a2ea0)
- Trump Bot? Or a Twitter bot which mimics [@realDonaldTrump](http://www.twitter.com/@realdonaldtrump)
- Want to level up? Make a **Narendra Modi bot** by scrapping off his *Hindi* speeches from his [personal website](http://www.narendramodi.in).

Check [mlm/blog](http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/) for hints

**Text Classification**
- Can you classify the text of an e-mail message to decide who sent it?
- A data set is available at [Enron Data](https://www.cs.cmu.edu/~./enron/)

Forecasting
---------
**Rainfall prediction** 
- How much will it rain this year where you live? 
- 45 years of daily precipitation data from the Northwest of the US mentioned [here](http://research.jisao.washington.edu/data_sets/widmann/) is good for for getting started. 

**Pollution Level Forecasting** 
- Multi-variate Time Series forecasting on the [Air Quality dataset](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)

**Home Electricity Forecasting** 
- Find a short term forecast on electricity consumption of a single home. Find the dataset [here](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption). 
- Use the model above to forecast your home's electricity consumption - does the model still work? Why or why not? 

Recommender Systems
---------
**Movie Recommender** 
- Can you predict the rating a user will give on a movie? 
- Do this using the movies that user has rated in the past, as well as the ratings similar users have given similar movies. 
- The data is available at [Netflix Prize](http://www.netflixprize.com/) and [MovieLens Datasets](https://grouplens.org/datasets/movielens/)

**Search + Recommendation System** 
- Predict which Xbox game a visitor will be most interested in based on their search query
- [BestBuy dataset via Kaggle](https://www.kaggle.com/c/acm-sf-chapter-hackathon-small/data)

**Can you predict Influencers in the Social Network?** 
- Trying finding influencers via the [PeerIndex dataset](https://www.kaggle.com/c/predict-who-is-more-influential-in-a-social-network/data)

Vision
---------
**Image Classification**
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet](http://www.image-net.org/)
- [MS Coco](http://mscoco.org/) is the modern replacement to the ImageNet challenge
- [MNIST Handwritten Digit Classification Challenge](http://yann.lecun.com/exdb/mnist/)  is the classic entry point
- [Character recognition (digits)](http://ai.stanford.edu/~btaskar/ocr/) is the good old Optical Character Recognition problem
- Bird Species Identification from an Image using the [Caltech-UCSD Birds dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset
- Diagnosing and Segmenting Brain Tumors and Phenotypes using MRI Scans
    - MICCAI Machine Learning Challenge aka [MLC 2014 dataset](https://www.nmr.mgh.harvard.edu/lab/laboratory-computational-imaging-biomarkers/miccai-2014-machine-learning-challenge)


## FAQ
1. **Can I use the ideas here for my thesis?** Yeah, totally. I'd love to know how it went. 

2. **Do you want to share my solution/code to a problem here**? Yeah, sure - why not? Go to [Github issues](https://github.com/NirantK/awesome-project-ideas/issues) in the repository and let me know there. 

3. **How can I add my ideas here?** Just send a pull request and we'll discuss? 

4. **Hey @NirantK, something is wrong here!** Yikes, I am sorry. Please tell me by raising a [Github issue](https://github.com/NirantK/awesome-project-ideas/issues). I'll try to fix it as soon as possible. 

### Credits
This repo was compiled by [Nirant Kasliwal](http://twitter.com/NirantK)
