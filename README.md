# Assignment 2: Zebra Finch Call-Type Classification

## Background

Zebra finches (Taeniopygia guttata) produce a variety of distinct vocalisations: distance calls, nest calls, aggressive calls, songs, etc.

Each call type is acoustically distinct and serves a different social function. Automated classification of these calls from audio recordings is an active research problem in bioacoustics.

## Dataset

The provided folder contains ~2,900 short WAV recordings from multiple zebra finches. Each filename encodes the zebra ID, recording date, call type, and a sequence number:

```
BlaBla0506_110302-AggC-04.wav
└────────┘ └─────┘└──┘ └┘
Zebra ID   Date   VT   Seq
```

Your task is to build a classifier that predicts the vocalization (call) type from the audio.

The first 2 letters of the `VT` tag encode the type of vocalization, i.e. the ten classes:

- Ag -> Aggressive call 
- Be -> Begging calls (not present in this data set as we include adult birds only)
- DC -> Distance call 
- Di -> Distress call 
- LT -> Long tonal call 
- Ne -> Nest call 
- So -> Song 
- Te -> Tet call 
- Th -> Thuk call 
- Tu -> Tuck call 
- Wh -> Whine call

You can find more details in the given file `Library_notes.pdf`.

## Objectives

**1. Data processing**

* Parse filenames to extract the call type labels
* Perform a stratified train / val / test split (70% / 15% / 15%)

**2. Data exploration**

* Plot the class distribution and discuss the class imbalance
* Show the duration (length) distribution per class and justify your chosen fixed clip length
* Discuss which is most suitable transformation (e.g. log-mel spectrogram, PCEN, etc.) for zebra finch calls

**3. Modelling**

* Implement both a CNN and RNN based model to predict the class, and compare their performance
* Finetune your model hyperparameters (number of layers, kernels, etc.) to increase performance further
* Use a validation set to monitor overfitting; implement early stopping

**4. Evaluation**

* Pick suitable metrics to report your final model's performance on the held-out test set
* Select a suitable approach to interpretate your model performance and behavior

Submit a single Jupyter Notebook (.ipynb) that runs top-to-bottom without errors. All plots and outputs must be pre-rendered (i.e. run all cells before submitting).

## Tips

* The class distribution is uneven. Addressing this imbalance (e.g. through loss weighting or oversampling) is likely to improve your model's performance, especially on rarer call types
* Recordings vary considerably in length. You will need to pad shorter clips and/or trim longer ones to a fixed duration
* For the audio preprocessing, you may use any of the approaches introduced in the lab session, or look for methods specifically designed for bird or animal vocalisations