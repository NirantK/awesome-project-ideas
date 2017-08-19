# Mega Project List of 
# Machine Learning/Artifical Intelligence Project Ideas
A list of practical Machine Learning or AI projects for learners

These projects are divided in multiple categories, same problem may appear in more than one categories. 

Tags:

:whale: ***Deep learning** suitable ideas*

:computer: *Data will probably fit on your laptop* 

:cloud: *Might need cloud service*. Recommended for those with programming experience

Problems are motivated by the ones shared at:

* [CMU Machine Learning](http://www.cs.cmu.edu/~./10701/projects.html)
* [Stanford CS229 Machine Learning Projects](http://cs229.stanford.edu/)

## Table of Contents

- [Text](https://github.com/NirantK/awesome-project-ideas#text) - include NLP tasks here for now
- [Vision](https://github.com/NirantK/awesome-project-ideas#vision) - includes image and video processing
- [Forecasting](https://github.com/NirantK/awesome-project-ideas#forecasting) - mostly Time Series and similar forecasting challenges
- [Recommender Systems](https://github.com/NirantK/awesome-project-ideas#recommender-systems)

Text
---------
### Text Classification

:computer: **Can you classify the text of an e-mail message to decide who sent it?** The Enron E-mail data set contains about 500,000 e-mails from about 150 users. The data set is available here: [Enron Data](https://www.cs.cmu.edu/~./enron/)

:cloud: :whale: **Autonomous Tagging of Stack Overflow Questions** Make a multi-label classification system that automatically assigns tags for questions posted on a forum such as Stackoverflow or Quora. 
Find [StackLite](https://www.kaggle.com/stackoverflow/stacklite) for your :computer: or [10% sample](https://www.kaggle.com/stackoverflow/stacksample) for :cloud: 

:cloud: :whale: :computer: **Sentiment Analysis** on Tweets or Long text such as news items, Quora answers

:cloud: :whale: **Identify keywords from millions of text questions** such as those on Stackoverflow for a Facebook Data Science Recruitment Challenge? Find the [data on Kaggle](https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/data)

### Natural Language Understanding
:computer: :whale: **Automated Essay Grading** The purpose of this project is to implement and train machine learning algorithms to automatically assess and grade essay responses. These grades from the automatic grading system should match the human grades consistently

:computer: :whale: **Quora Question Pairs** Can you identify question pairs that have the same intent? [Data](https://www.kaggle.com/c/quora-question-pairs/data) via Kaggle

:whale: :whale2: **Machine Translation**

:whale: :whale2: **Text Generation** Generate plausible text sequences for a given problem. You can use the [Alice in Wonderland](https://www.gutenberg.org/ebooks/11) text from Project Gutenberg. 
Check [mlm/blog](http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/) for further details

Vision
---------
#### Image Classification
**Classify/tag images from a dataset** such as CIFAR-10, [ImageNet](http://www.image-net.org/) or [MS Coco](http://mscoco.org/) into classes such as dog, cat, horse.
You can try to create an object recognition system which can identify which object category is the best match for a given test image.
Apply clustering to learn object categories without supervision

:computer: :whale: **MNIST Handwritten Digit Classification Challenge**  is the classic entry point. The [MNIST data](http://yann.lecun.com/exdb/mnist/) is beginner-friendly and is small enough to fit on one computer.

:computer: :whale: **Character recognition** (digits) is the good old Optical Character Recognition problem. [Stanford dataset](http://ai.stanford.edu/~btaskar/ocr/) is a good starting point. 

:whale: **Bird Species Identification from an Image** using the [Caltech-UCSD Birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset

:whale: :cloud: **Diagnosing and Segmenting Brain Tumors and Phenotypes using MRI Scans** using the MICCAI Machine Learning Challenge aka [MLC 2014 dataset](https://www.nmr.mgh.harvard.edu/lab/laboratory-computational-imaging-biomarkers/miccai-2014-machine-learning-challenge)

Forecasting
---------
**Rainfall prediction** Learn a probabilistic model to predict rain levels. 45 years of daily precipitation data from the Northwest of the US mentioned [here](http://research.jisao.washington.edu/data_sets/widmann/) is good for for getting started. 

:computer: **Pollution Level Forecasting** using Multi-variate Time Series forecasting on the [Air Quality dataset](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)

:computer: **Home Electricity Forecasting** Find a short term forecast on electricity consumption of a single home. Find the dataset [here](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

Recommender Systems
---------
:cloud: **Movie Recommender** **Can you predict the rating a user will give on a movie?** Do this using the movies that user has rated in the past, as well as the ratings similar users have given similar movies. The data is available here: [Netflix Prize](http://www.netflixprize.com/)

:computer: **Best Buy Search + Recommendation System** Predict which Xbox game a visitor will be most interested in based on their search query using the [BestBuy dataset on Kaggle](https://www.kaggle.com/c/acm-sf-chapter-hackathon-small/data)

Interesting
---------
:computer: **Can you predict Influencers in the Social Network?** In the Facebook/Twitter era, it's extremely useful to find influencers for targeting advertising. Trying finding them via the [PeerIndex dataset](https://www.kaggle.com/c/predict-who-is-more-influential-in-a-social-network/data) 

## FAQ
1. **Can I use the ideas here for my thesis?** Yeah, totally. I'd love to know how it went. 

2. **Do you want to share my solution/code to a problem here**? Yeah, sure - why not? Go to [Github issues](https://github.com/NirantK/awesome-project-ideas/issues) in the repository and let me know there. 

3. **How can I add my ideas here?** Just send a pull request and we'll discuss? 

4. **Hey @NirantK, something is wrong here!** Yikes, I am sorry. Please tell me by raising a [Github issue](https://github.com/NirantK/awesome-project-ideas/issues). I'll try to fix it as soon as possible. 

### Credits
This repo was compiled by [Nirant Kasliwal](http://twitter.com/NirantK)
