<!-- markdownlint-disable MD033 -->

# Awesome Deep Learning Project Ideas

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of practical deep learning and machine learning project ideas

- 30+ ideas
- Relevant to both the academia and industry
- Ranges from beginner friendly to research projects

---

## Contents

- [Text](#text) - With some topics about Natural language processing

- [Forecasting](#forecasting) - Most of the topics in this section is about Time Series and similar forecasting challenges

- [Recommendation Systems](#recommendation-systems)

- [Vision](#vision) - With topics about image and video processing

- [Covid19](#covid19) - Multi or Single Domain ideas from the Covid19 theme

- [Music and Audio](#music) - These topics are about combining ideas from language and audio to understand music

- [Conclusion](#conclusion)

---


## Covid19

- **Bing Coronavirus** 

  - Classify Bing Queries as either specific (e.g. about a specific location) or generic. You might have to figure out a more exact definition of specific or generic though
  - Dataset: [BingCoronavirusQuerySet](https://github.com/microsoft/BingCoronavirusQuerySet)
  
- **Covid Clinical Data**
  - Rank and sort high risk patients using clinical data. Pick an interpretable approach if you can.
  - Dataset: [CovidClinicalData](https://covidclinicaldata.org/)
  
If you haven't already, checkout [Kaggle's Covid19 Section](https://www.kaggle.com/covid19) as well. It has datasets and ideas both.
  
## Text

- **Autonomous Tagging of StackOverflow Questions**
  
  - Make a multi-label classification system that automatically assigns tags for questions posted on a forum such as StackOverflow or Quora.
  - Dataset: [StackLite](https://www.kaggle.com/stackoverflow/stacklite) or [10% sample](https://www.kaggle.com/stackoverflow/stacksample)

- **Keyword/Concept identification**
  
  - Identify keywords from millions of questions
  - Dataset: [StackOverflow question samples by Facebook](https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/data)

- **Topic identification**
  - Multi-label classification of printed media articles to topics
  - Dataset: [Greek Media monitoring multi-label classification](https://www.kaggle.com/c/wise-2014/data)

### Natural Language Understanding

- **Automated essay grading**
  - The purpose of this project is to implement and train machine learning algorithms to automatically assess and grade essay responses.
  - Dataset: [Essays with human graded scores](https://www.kaggle.com/c/asap-aes/data)

- **Sentence to Sentence semantic similarity**
  - Can you identify question pairs that have the same intent or meaning?
  - Dataset: [Quora question pairs](https://www.kaggle.com/c/quora-question-pairs/data) with similar questions marked

- **Fight online abuse**
  - Can you confidently and accurately tell whether a particular comment is abusive?
  - Dataset: [Toxic comments on Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

- **Open Domain question answering**
  - Can you build a bot which answers questions according to the student's age or her curriculum?
  - [Facebook's FAIR](https://github.com/facebookresearch/DrQA) is built in a similar way for Wikipedia.
  - Dataset: [NCERT books](http://ncert.nic.in/ebooks.html) for K-12/school students in India, [NarrativeQA by Google DeepMind](https://github.com/deepmind/narrativeqa) and [SQuAD by Stanford](https://rajpurkar.github.io/SQuAD-explorer/)

- **Social Chat/Conversational Bots**
  - Can you build a bot which talks to you just like people talk on social networking sites?
  - Reference: [Chat-bot architechture](https://github.com/aryanc55/TS3000_TheChatBOT) 
  - Dataset: [Reddit Dataset](http://files.pushshift.io/reddit/comments/) 

- **Automatic text summarization**
  - Can you create a summary with the major points of the original document?
  - Abstractive (write your own summary) and Extractive (select pieces of text from original) are two popular approaches
  - Dataset: [CNN and DailyMail News Pieces](http://cs.nyu.edu/~kcho/DMQA/) by Google DeepMind

- **Copy-cat Bot**
  - Generate plausible new text which looks like some other text
  - Obama Speeches? For instance, you can create a bot which writes some [new speeches in Obama's style](https://medium.com/@samim/obama-rnn-machine-generated-political-speeches-c8abd18a2ea0)
  - Trump Bot? Or a Twitter bot which mimics [@realDonaldTrump](http://www.twitter.com/@realdonaldtrump)
  - Narendra Modi bot saying "*doston*"? Start by scrapping off his *Hindi* speeches from his [personal website](http://www.narendramodi.in)
  - Example Dataset: [English Transcript of Modi speeches](https://github.com/mgupta1410/pm_modi_speeches_repo)

Check [mlm/blog](http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/) for some hints.

- **Sentiment Analysis**
  - Do Twitter Sentiment Analysis on tweets sorted by geography and timestamp.
  - Dataset: [Tweets sentiment tagged by humans](https://inclass.kaggle.com/c/si650winter11/data)

- **De-anonymization**
  - Can you classify the text of an e-mail message to decide who sent it?
  - Dataset: [150,000 Enron emails](https://www.cs.cmu.edu/~./enron/)

## Forecasting

- **Univariate Time Series Forecasting**
  - How much will it rain this year?
  - Dataset: [45 years of rainfall data](http://research.jisao.washington.edu/data_sets/widmann/)

- **Multi-variate Time Series Forecasting**
  - How polluted will your town's air be? Pollution Level Forecasting
  - Dataset: [Air Quality dataset](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)

- **Demand/load forecasting**
  - Find a short term forecast on electricity consumption of a single home
  - Dataset: [Electricity consumption of a household](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

- **Predict Blood Donation**
  - We're interested in predicting if a blood donor will donate within a given time window.
  - More on the problem statement at [Driven Data](https://www.drivendata.org/competitions/2/warm-up-predict-blood-donations/page/7/).
  - Dataset: [UCI ML Datasets Repo](https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center)

## Recommendation systems

- **Movie Recommender**
  - Can you predict the rating a user will give on a movie?
  - Do this using the movies that user has rated in the past, as well as the ratings similar users have given similar movies.
  - Dataset: [Netflix Prize](http://www.netflixprize.com/) and [MovieLens Datasets](https://grouplens.org/datasets/movielens/)

- **Search + Recommendation System**
  - Predict which Xbox game a visitor will be most interested in based on their search query
  - Dataset: [BestBuy](https://www.kaggle.com/c/acm-sf-chapter-hackathon-small/data)

- **Can you predict Influencers in the Social Network?**
  - How can you predict social influencers?
  - Dataset: [PeerIndex](https://www.kaggle.com/c/predict-who-is-more-influential-in-a-social-network/data)

## Vision

- **Image classification**
  - Object recognition or image classification task is how Deep Learning shot up to it's present-day resurgence
  - Datasets:
    - [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
    - [ImageNet](http://www.image-net.org/)
    - [MS COCO](http://mscoco.org/) is the modern replacement to the ImageNet challenge
    - [MNIST Handwritten Digit Classification Challenge](http://yann.lecun.com/exdb/mnist/)  is the classic entry point
    - [Character recognition (digits)](http://ai.stanford.edu/~btaskar/ocr/) is the good old Optical Character Recognition problem
    - Bird Species Identification from an Image using the [Caltech-UCSD Birds dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset
  - Diagnosing and Segmenting Brain Tumours and Phenotypes using MRI Scans
    - Dataset: MICCAI Machine Learning Challenge aka [MLC 2014](https://www.nmr.mgh.harvard.edu/lab/laboratory-computational-imaging-biomarkers/miccai-2014-machine-learning-challenge)
  - Identify endangered right whales in aerial photographs
    - Dataset: [MOAA Right Whale](https://www.kaggle.com/c/noaa-right-whale-recognition)
  - Can computer vision spot distracted drivers?
    - Dataset: [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data) on Kaggle

- **Bone X-Ray dompetition**
  - Can you identify if a hand is broken from a X-ray radiographs automatically with better than human performance?
  - Stanford's Bone XRay Deep Learning Competition with [MURA Dataset](https://stanfordmlgroup.github.io/competitions/mura/)

- **Image Captioning**
  - Can you caption/explain the photo a way human would?
  - Dataset: [MS COCO](http://mscoco.org/dataset/#captions-challenge2015)

- **Image Segmentation/Object Detection**
  - Can you extract an object of interest from an image?
  - Dataset: [MS COCO](http://mscoco.org/dataset/#detections-challenge2017), [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge/data) on Kaggle

- **Large-Scale Video Understanding**
  - Can you produce the best video tag predictions?
  - Dataset: [YouTube 8M](https://research.google.com/youtube8m/index.html)

- **Video Summarization**
  - Can you select the semantically relevant/important parts from the video?
  - Example: [Fast-Forward Video Based on Semantic Extraction](https://arxiv.org/abs/1708.04160)
  - Dataset: Unaware of any standard dataset or agreed upon metrics? I think [YouTube 8M](https://research.google.com/youtube8m/index.html) might be good starting point.

- **Style Transfer**
  - Can you recompose images in the style of other images?
  - Dataset: [fzliu on GitHub](https://github.com/fzliu/style-transfer/tree/master/images) shared target and source images with results

- **Chest XRay**
  - Can you detect if someone is sick from their chest XRay? Or guess their radiology report?
  - Dataset: [MIMIC-CXR at Physionet](https://physionet.org/content/mimic-cxr/2.0.0/)

- **Face Recognition**
  - Can you identify whose photo is this? Similar to Facebook's photo tagging or Apple's FaceId
  - Dataset: [face-rec.org](http://www.face-rec.org/databases/), or [facedetection.com](https://facedetection.com/datasets/)

- **Clinical Diagnostics: Image Identification, classification & segmentation**
  - Can you help build an open source software for lung cancer detection to help radiologists?
  - Link: [Concept to clinic](https://concepttoclinic.drivendata.org/) challenge on DrivenData

- **Satellite Imagery Processing for Socioeconomic Analysis**
  - Can you estimate the standard of living or energy consumption of a place from night time satellite imagery?
  - Reference for Project details: [Stanford Poverty Estimation Project](http://sustain.stanford.edu/predicting-poverty/)

- **Satellite Imagery Processing for Automated Tagging**
  - Can you automatically tag satellite images with human features such as buildings, roads, waterways and so on?
  - Help free the manual effort in tagging satellite imagery: [Kaggle Dataset by DSTL, UK](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection)

## Music

- **Music/Audio Recommendation Systems**
  - Can you tell if two songs are similar using their sound or lyrics?
  - Dataset: [Million Songs Dataset](https://labrosa.ee.columbia.edu/millionsong/) and it's 1% sample.
  - Example: [Anusha et al](https://cs224d.stanford.edu/reports/BalakrishnanDixit.pdf)

- **Music Genre recognition using neural networks**
  - Can you identify the musical genre using their spectrograms or other sound information?
  - Datasets: [FMA](https://github.com/mdeff/fma) or [GTZAN on Keras](https://github.com/Hguimaraes/gtzan.keras)
  - Get started with [Librosa](https://librosa.github.io/librosa/index.html) for feature extraction

---

### FAQ

- **Can I use the ideas here for my thesis?**
  Yes, totally! I'd love to know how it went.

- **Do you have any advice before I start my project?**
  [Advice for Short Term Machine Learning Projects](https://rockt.github.io/2018/08/29/msc-advice) by Tim R. is a pretty good starting point!

- **Would you like to share my solution/code to a problem here?**
  Sure - why not?
  
  Go to the [GitHub issues](https://github.com/NirantK/awesome-project-ideas/issues) tab in this repository and let me know there.

- **How can I add my ideas here?**
  Just send a pull request and we'll discuss?

- **Hey, something is wrong here!**
  Yikes, I am sorry. Please tell me by raising a [GitHub issue](https://github.com/NirantK/awesome-project-ideas/issues).

  I'll fix it as soon as possible.

### Acknowledgements

Problems are motivated by the ones shared at:

- [CMU Machine Learning](http://www.cs.cmu.edu/~./10701/projects.html)
- [Stanford CS229 Machine Learning Projects](http://cs229.stanford.edu/)

### Credit

Built with lots of keyboard smashing and copy-pasta love by NirantK. Find me on [Twitter](http://www.twitter.com/@nirantk)!

### [Receive New & Exclusive Ideas right in your Inbox](http://eepurl.com/dIKLhz)

These ideas have been seen by [![hit count](http://hits.dwyl.io/NirantK/awesome-project-ideas.svg)](http://hits.dwyl.io/NirantK/awesome-project-ideas) people in last few months!

If you are interested in seeing **exclusive** machine learning and deep learning project ideas, share your [e-mail address here](http://eepurl.com/dIKLhz)!

### License

This repository is licensed under the MIT License. Please see the [LICENSE file](./LICENSE) for more details.
