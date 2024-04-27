# Genre classification
_Why do I care about genre classification?_

My main task is neural style transfer between a musical piece (the content) and an auditory ambience (the style). Genre classification is a means to model the timbral and acoustic qualities of a musical piece (as both are relevant factors in determining genre), and are key aspects I want to modify with neural style transfer.

---

For genre classification, I have trained 2 types of models:

- CNN using mel-frequency cepstral coefficients (MFCCs) as input
- CNN using melspectrograms as input

The reasons for each are given in the respecting training sections in `GenreClassification.ipynb`.

---

**DATASET**:

GTZAN Dataset for Music Classification: <br> https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
