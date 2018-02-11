# Awesome Deep Learning Project Ideas 
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) 

**Curated List of Practical Deep Learning, Machine learning project Ideas**
 - 25+ ideas 
 - Relevant to academia and industry both
 - Range from beginner friendly to research projects

Problems are motivated by the ones shared at:

* [CMU Machine Learning](http://www.cs.cmu.edu/~./10701/projects.html)
* [Stanford CS229 Machine Learning Projects](http://cs229.stanford.edu/)


Please do send a PR or open an issue with your suggestions and requests!
---

## Contents

- [Text](#text) - including natural language under this section 
- [Forecasting](#forecasting) - mostly Time Series and similar forecasting challenges
- [Recommender Systems](#recommender-systems)
- [Vision](#vision) - includes image and video processing

Text
---------
**Autonomous Tagging of Stack Overflow Questions** 
- Make a multi-label classification system that automatically assigns tags for questions posted on a forum such as Stackoverflow or Quora 
- Dataset: [StackLite](https://www.kaggle.com/stackoverflow/stacklite) or [10% sample](https://www.kaggle.com/stackoverflow/stacksample) 

**Keyword/Concept Identification** 
- Identify keywords from millions of text questions
- Dataset: [StackOverflow Questions Sample by Facebook](https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/data)

**Topics Identification**
- Multi-label classification of printed media articles to topics
- Dataset: [Greek Media Monitoring Multilabel Classification](https://www.kaggle.com/c/wise-2014/data)

### Natural Language Understanding
**Automated Essay Grading** 
- The purpose of this project is to implement and train machine learning algorithms to automatically assess and grade essay responses. 
- Dataset: [Essays with human graded scores](https://www.kaggle.com/c/asap-aes/data)

**Sentence to Sentence Semantic Similarity** 
- Can you identify question pairs that have the same intent? Or sentences that have the same meaning? 
- Dataset: [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs/data) with similar questions marked

**Fight online abuse**
- Can you confidently and accurately tell via a particular is abusive? 
- Dataset: [Toxic Comments on Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

**Open Domain Question Answering** 
- Can you build a bot which answers questions according to the student's age or her curriculum? 
- [Facebook's FAIR](https://github.com/facebookresearch/DrQA) built similar for Wikipedia 
- Dataset: [NCERT books](https://www.github.com/NirantK/ncert) for K-12/school students in India, [NarrativeQA by Google DeepMind](https://github.com/deepmind/narrativeqa) and [SQuAD by Stanford](https://rajpurkar.github.io/SQuAD-explorer/)

**Automatic Text Summarization**
- Can you create a summary with the major points of the original document?
- Abstractive (write your own summary) and Extractive (select pieces of text from original) are two popular approaches
- Dataset: [CNN and DailyMail News Pieces](http://cs.nyu.edu/~kcho/DMQA/) by Google Deepmind

**Copy-cat Bot**
- Generate plausible new text which looks like some other text
- Obama Speeches? For instance, you can create a bot which writes [new speeches in Obama's style](https://medium.com/@samim/obama-rnn-machine-generated-political-speeches-c8abd18a2ea0)
- Trump Bot? Or a Twitter bot which mimics [@realDonaldTrump](http://www.twitter.com/@realdonaldtrump)
- Narendra Modi bot saying *doston*? Start by scrapping off his *Hindi* speeches from his [personal website](http://www.narendramodi.in)
  - Example Dataset: [English Transcript of Modi speeches](https://github.com/mgupta1410/pm_modi_speeches_repo)

Check [mlm/blog](http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/) for hints

**Sentiment Analysis**
- Do Twitter Sentiment Analysis on tweets sorted by geography and timestamp
- Dataset: [Tweets sentiment tagged by humans](https://inclass.kaggle.com/c/si650winter11/data)

**De-anonymization**
- Can you classify the text of an e-mail message to decide who sent it?
- Dataset: [150,000 Enron emails](https://www.cs.cmu.edu/~./enron/)

Forecasting
---------
**Univariate Time Series Forecasting** 
- How much will it rain this year?
- Dataset: [45 years of rainfall data](http://research.jisao.washington.edu/data_sets/widmann/)

**Multi-variate Time Series Forecasting** 
- How polluted will your town air be? Pollution Level Forecasting
- Dataset: [Air Quality dataset](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)

**Demand/load forecasting** 
- Find a short term forecast on electricity consumption of a single home
- Dataset: [Electricity Consumption of a household](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

**Predict Blood Donation**
- We're interested in predicting if a blood donor will donate within a given time window
- More on the problem statement at [Driven Data](https://www.drivendata.org/competitions/2/warm-up-predict-blood-donations/page/7/)
- Dataset: [UCI ML Datasets Repo](https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center)

Recommender Systems
---------
**Movie Recommender** 
- Can you predict the rating a user will give on a movie? 
- Do this using the movies that user has rated in the past, as well as the ratings similar users have given similar movies. 
- Dataset: [Netflix Prize](http://www.netflixprize.com/) and [MovieLens Datasets](https://grouplens.org/datasets/movielens/)

**Search + Recommendation System** 
- Predict which Xbox game a visitor will be most interested in based on their search query
- Dataset: [BestBuy](https://www.kaggle.com/c/acm-sf-chapter-hackathon-small/data)

**Can you predict Influencers in the Social Network?** 
- How can you predict social influencers? 
- Dataset: [PeerIndex](https://www.kaggle.com/c/predict-who-is-more-influential-in-a-social-network/data) 

Vision
---------
**Image Classification**
- Object recognition or image classification task is how Deep Learning shot up to it's present-day resurgence
- Datasets: 
  - [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
  - [ImageNet](http://www.image-net.org/)
  - [MS COCO](http://mscoco.org/) is the modern replacement to the ImageNet challenge
  - [MNIST Handwritten Digit Classification Challenge](http://yann.lecun.com/exdb/mnist/)  is the classic entry point
  - [Character recognition (digits)](http://ai.stanford.edu/~btaskar/ocr/) is the good old Optical Character Recognition problem
  - Bird Species Identification from an Image using the [Caltech-UCSD Birds dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset
  - Diagnosing and Segmenting Brain Tumors and Phenotypes using MRI Scans
      - Dataset: MICCAI Machine Learning Challenge aka [MLC 2014](https://www.nmr.mgh.harvard.edu/lab/laboratory-computational-imaging-biomarkers/miccai-2014-machine-learning-challenge)
  - Identify endangered right whales in aerial photographs
      - Dataset: [MOAA Right Whale](https://www.kaggle.com/c/noaa-right-whale-recognition)
  - Can computer vision spot distracted drivers?
      - Dataset: [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data) on Kaggle
      
**Image Captioning**
- Can you caption/explain the photo a way human would? 
- Dataset: [MS COCO](http://mscoco.org/dataset/#captions-challenge2015)

**Image Segmentation/Object Detection**
- Can you extract an object of interest from an image? 
- Dataset: [MS COCO](http://mscoco.org/dataset/#detections-challenge2017), [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge/data) on Kaggle

**Large-Scale Video Understanding**
- Can you produce the best video tag predictions?
- Dataset: [Youtube 8M](https://research.google.com/youtube8m/index.html)

**Video Summarization**
- Can you select the semantically relevant/important parts from the video? 
- Example: [Fast-Forward Video Based on Semantic Extraction](https://arxiv.org/abs/1708.04160)
- Dataset: Unaware of any standard dataset or agreed upon metrics, [Youtube 8M](https://research.google.com/youtube8m/index.html) might be good starting point

**Style Transfer**
- Can you recompose images in the style of other images? 
- Dataset: [fzliu on Github](https://github.com/fzliu/style-transfer/tree/master/images) shared target and source images with results

**Face Recognition**
- Can you identify whose photo is this? Similar to Facebook's photo tagging or Apple's FaceId
- Dataset: [face-rec.org](http://www.face-rec.org/databases/), or [facedetection.com](https://facedetection.com/datasets/)

**Clinical Diagnostics: Image Idenitification, classification & segmentation**
- Can you help build an open source software for lung cancer detection to help radiologists? 
- Link: [Concept to Clinic](https://concepttoclinic.drivendata.org/) challenge on DrivenData

**Satellite Imagery Processing for Socioeconomic Analysis**
- Can you estimate the standard of living or energy consumption of a place from night time satellite imagery? 
- Reference for Project details: [Stanford Poverty Estimation Project](http://sustain.stanford.edu/predicting-poverty/)

**Satellite Imagery Processing for Automated Tagging**
- Can you automatically tag satellite images with human features such as buildings, roads, waterways and so on? 
- Help free the manual effort in tagging satellite imagery: [Kaggle Dataset by DSTL, UK](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection)

## FAQ
**Can I use the ideas here for my thesis?** 
Yeah, totally. I'd love to know how it went. 

**Do you want to share my solution/code to a problem here**? Yeah, sure - why not? 
Go to [Github issues](https://github.com/NirantK/awesome-project-ideas/issues) in the repository and let me know there. 

**How can I add my ideas here?** Just send a pull request and we'll discuss? 

**Hey @NirantK, something is wrong here!** Yikes, I am sorry. Please tell me by raising a [Github issue](https://github.com/NirantK/awesome-project-ideas/issues). I'll try to fix it as soon as possible. 

### Queries? 
[Nirant Kasliwal](http://www.linkedin.com/in/nirant) compiled the ideas in this repository 
Find him on [Twitter](http:/www.twitter.com/NirantK) or [Linkedin](http://www.linkedin.com/in/nirant)

[![HitCount](http://hits.dwyl.io/NirantK/awesome-project-ideas.svg)](http://hits.dwyl.io/NirantK/awesome-project-ideas)
