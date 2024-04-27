# Ambience-to-Music Neural Style Transfer (AM-NST)

**Google Colab notebook**: [Neural Style Transfer (Basic)](https://drive.google.com/file/d/1NltGH6EDjP2ZkHsMax9aB90_pkpDRFkR/view?usp=sharing)

**Technical & conceptual notes**: [NOTES.md](https://github.com/pranigopu/ambience-to-music-neuralStyleTransfer/blob/27e2f006712c806734d8f87000979e5184fc2852/NOTES.md)

## Project overview
The AI system - Ambience-to-Music Neural Style Transfer (AM-NST) - produces audio reconstructed from a Mel-spectrogram representing some transfer of an ambient audio (style) to a musical audio (content). AM-NST uses the identity and/or convolutional layers of a self-designed, self-trained convolutional neural network (CNN) to compare the Mel-spectrogram of the target audio to the Mel-spectrograms of the content and style audio at different levels of abstraction (each convolutional layer extracts features from the Mel-spectrogram; the deeper the layer, the higher the level of measurement/detail omission, i.e. of abstraction).

Comparisons at higher levels of abstraction indicate differences in broader features (e.g. timbre, aural texture, etc.), while comparisons at lower levels indicate differences in narrower, i.e. more fine-grained features (e.g. content, acoustics, etc.). The target Mel-spectrogram is modified using gradient descent. How? (1) Style loss is calculated as the mean squared error of the Gram matrices of the target and style Mel-spectrograms. (2) Content loss is calculated as the mean squared error of the content and target Mel-spectrograms. (3) Total loss is calculated as the weighted sum of the content loss and style loss weighted by content weight and style weight respectively. (4) The gradient is calculated for total loss and applied to the target Mel-spectrogram. Gradient descent is done in this way for a set number of iterations, after which the target Mel-spectrogram is converted to an audio signal that can be listened to.

An art piece's style represents the way the piece's features interact with each other, i.e. it represents a method/approach to creativity. Thus, learning to emulate style is a step toward expanding our grasp of creativity. Hence, I was interested in exploring style transfer in general, and I was keen to explore NST for music in particular as it is not yet as researched as NST for visual art.

**NOTE**: _The whole process is end-to-end and includes audio pre-processing and post-processing._

## Neural model used for NST
Neural style transfer uses layers of a genre classification CNN (details below). Genre classification is expected to extract and use a piece's style; hence it is used.

### Genre classification CNN
#### Source dataset details
- **Name**: GTZAN Dataset
- **Audio used**: 100 30-second tracks for each genre out of 10 total genres
- **Annotations used**: CSV files with genre labels for each 30-second track
- **Link**: [GTZAN Dataset – Music Genre Classification (Kaggle)](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

#### Inputs
Inputs were Mel-spectrogram segments with 384 Mel-bands and 431 frames each. The Mel-spectrograms segments were obtained by (1) using the sampling rate 22050 Hz, FFT window size 1024 and hop length 256 and (2) segmenting track-wise Mel-spectrograms into segments 431 frames (corresponding to ~5 seconds of audio). _The sampling rate was chosen because that was the sampling rate of the training data. The other parameters were chosen based on the most memory-efficient parameters observed (experimentation was done in the audio preprocessing demo notebook) that would retain audio quality upon reconversion of the Mel-spectrogram to a raw audio signal._

#### Architecture
| Layer | Input shape | Parameters |
| --- | --- | --- |
Input | (None, 384, 431, 1) | 0 |
Identity \* | (None, 384, 431, 1) | 0 |
Batch normalisation 1 | (None, 384, 431, 1) | 4 |
Convolutional layer 1 | (None, 383, 430, 32) | 160 |
Max pooling layer 1 | (None, 192, 215, 32) | 0 |
Convolutional layer 2 | (None, 191, 214, 32) | 4128 |
Max pooling layer 2 | (None, 96, 107, 32) | 0 |
Convolutional layer 3 | (None, 95, 106, 32) | 4128 |
Max pooling layer 3 | (None, 48, 53, 32) | 0 |
Convolutional layer 4 | (None, 47, 52, 32) | 4128 |
Max pooling layer 4 | (None, 24, 26, 32) | 0 |
Convolutional layer 5 | (None, 23, 25, 32) | 4128 |
Max pooling layer 5 | (None, 12, 13, 32) | 0 |
Convolutional layer 6 | (None, 11, 12, 32) | 4128 |
Max pooling layer 6 | (None, 6, 6, 32) | 0 |
Convolutional layer 7 | (None, 5, 5, 32) | 4128 |
Max pooling layer 7 | (None, 5, 5, 32) | 0 |
Convolutional layer 8 | (None, 4, 4, 32) | 4128 |
Max pooling layer 8 | (None, 4, 4, 32) | 0 |
Batch normalisation 2 | (None, 4, 4, 32) | 128 |
Dropout (rate = 0.3) | (None, 4, 4, 32) | 0 |
Flatten | (None, 512) | 0 |
Dense layer 1 | (None, 64) | 32832 |
Dense layer 2 | (None, 10) | 650 |

\* _For facilitating value-wise comparison_

- **Total params**: 62670
- **Trainable params**: 62604
- **Non-trainable params**: 66

NOTE: Each convolutional layer has 32 filters, and each max pooling layer has pool size (2, 2) and stride (2, 2).

#### Training
- Training-to-validation split = 9 : 1 (5395 segments : 599 segments)
- Learning rate (LR) = 0.001 (initially)
- If validation accuracy fails to rise by at least 0.05 in 2 epochs, LR is halved
- Training was done for 30 epochs

Results after 30 epochs:

- Training accuracy = 0.9182576537132263 ~ 91.826%
- Training loss = 0.25373783707618713
- Validation accuracy = 0.8230383992195129 ~ 93.304%
- Validation loss = 0.5615319013595581

### Neural style transfer (NST)
#### Inputs and outputs
The same kind of input as for genre classifier is used (so that genre classifier layers can be used). Output is a set of Mel-spectrogram segments of the same shape that can be stitched together to get the whole piece. The intended output is a piece that combines the timbre, acoustics and texture of the style while retaining the melodic and harmonic structure of the content.

#### Data source
- Ambient sounds for style (free downloads):
    - https://mixkit.co/free-sound-effects/
    - https://pixabay.com/sound-effects/
- Music pieces for content (royalty free):
    - https://www.chosic.com/free-music/

## Process
### Overall process diagram for AM-NST
![resources/AM-NST Process Diagram.png](https://github.com/pranigopu/ambience-to-music-neuralStyleTransfer/blob/c194443c53b59232aaf7abd8731ee6ef7f24f348/resources/AM-NST%20Process%20Diagram.png)

### Processing of inputs
1. Extracting CNN's convolutional layers (i.e. feature filters) for style and content <br> **NOTE**: _Hence, we get the "content layers" and "style layers"_
2. Obtaining Mel-spectrograms for each audio
3. The above are segmented and reshaped to match the CNN's input shape

### Application of neural model
The neural model (i.e.the CNN) is applied as follows:

- Content and target are passed to each content layer <br> $\rightarrow$ Each time, layer-wise outputs are passed to content loss function
- Style and target are passed to each style layer <br> $\rightarrow$ Each time, layer-wise outputs are passed to style loss function

**NOTE**: _The content/style loss obtained for each layer is added to the total loss_

### Output assessment and display
Based on my goals, a good style transfer is one wherein (1) the melodic and harmonic structure of the content are present in the output, (2) the timbre and acoustics of the style are present in the output such that it is not merely a superposition of two sets of audio data, and (3) the level of noise is minimal. Apart from human-based evaluation, I did not address output assessment (other than total loss), since I did not understand how to quantify my criteria. Output is displayed by reshaping the NST output to a Mel-spectrogram, reconstructing the raw audio signal from the Mel-spectrogram, and displaying an audio player for the raw audio signal.

### User interaction
Users can input parameters and audio names in input boxes using Google Colab's GUI system (input instructions are given). Interaction requires either mounting Google Drive with required data or uploading required data to the session storage. The process is end-to-end, so the user need not worry about pre-processing or post-processing.

## Evaluation of project outcomes
### Output quality
Using Ritchie's criteria, the evaluation of the outputs is as follows:

**Novelty**:

Since the outputs are a combination of the content and style genres, the outputs can stand apart from typical examples of either genre. For example, combining a piano piece by Bach with an electric hum leads to an output dissimilar from Bach's usual sound but unlike examples of genres with similar textures, such as techno or electronic music. However, the system has no process to alter the artistic quality of either the content or the style, so novelty depends entirely on the novelty of the content with respect to the style.

**Quality**:

Quality depends heavily on the input quality (artistic and acoustic) and NST parameters used. The system cannot isolate and alter the melodic and harmonic aspects of the content alone, which means style transfer leads to varying levels of background noise and sometimes a mere superposition of content and audio. Furthermore, the system has no means to show what features constitute timbre or texture and how to transfer those features from the style to the audio, thereby often resulting in a dubious, unconvincing mixture.

**Typicality**:

Considering a hypothetical typical mixture of content and style (e.g. Bach played in a horror-ambience style, or Mozart played in a cinematic style), the outputs of the system are atypical, i.e. they do not conform to any well-defined idea of a content-style mixture; this is in part due to poor quality, in part due to an ill-defined conceptual basis for audio style transfer.

### Higher-level issues
AM-NST is a weak computational creativity project, i.e. it focuses more on generating valuable artefacts than emulating creative behaviour. NST only leverages the performance of CNNs in estimating abstractions in an opaque manner (i.e. as a black-box) without emulating – at least not in a well-defined way – the essence creative behaviour.

## Value addition from the project
For NST in audio, the available pretrained networks for images were overkill, hence I trained my own, simpler genre-classification model that inputs Mel-spectrograms (Mel-spectrograms were used as they are a means store audio visually that is efficient (in terms of memory) and a good model for human auditory perception). I made the CNN architecture (inspired from Meng, 2021), after testing for a variety of parameters for Mel-spectrogram dimensions and CNN architectures. The training process is customised (using learning rate decay on plateau) to ensure steadier convergence. The NST process itself is relatively standard, although I added a means to decay learning rate and rollback target updates upon exploding gradients. The NST loop is made to be highly customisable to enable testing a range of potential parameters.

My project explores the potential NST in a specific task, namely the transferring the timbre, texture and acoustics of one piece to the melody and harmony of another. Hence, it aims to address the problem of inferring and transferring musical features by capturing a wider variety of abstract audio features at various levels of abstractions using more layers in the CNN with smaller kernels, hence the 8 convolutional layers. Using a wider selection of layers representing potentially valuable audio features, my project gives a means to experiment with features (of the style audio especially) at varying levels of abstractions and explore the effects of considering any set of abstract features. I used the framework of my project to do the same and have maintained my findings in this repository (see `FINDINGS.md`).

Using Google Colab, I have provided a sufficiently comprehensive interface that allows for a high level of customisability and thus experimentation in NST parameters (mainly the content and style layers and weights) and NST loop parameters (e.g. learning rate, decay rate, etc.). It also allows for experimentation using any audio that can be uploaded in the session storage or Google Drive, with many comprehensive and customisable path parameters. The process is also end-to-end, removing the need for any pre-processing or post-processing by the user. The only dependency is the file containing pretrained model weights, which can be easily uploaded and used.

As a final note, I have trained genre-classification CNNs that use MFCCs (Mel-frequency Cepstrum Coefficients), which – I have reason to believe – better capture qualities such as timbre. Using my project’s framework, I tried to combine inferences from Mel-spectrograms (for content) and MFCCs (for style). Although the results were dismal, I think the possibility was worth exploring and can be further explored using my project’s framework. Results can be found in this repository in the file named `NeuralStyleTransferWithUnevenModelling.ipynb`.

> CITATIONS:
> (In-text: Meng, 2021) <br>
> Men, Y. (2021). 'Music Genre Classification: A Comparative Analysis of CNN and XGBoost Approaches with Mel-frequency cepstral coefficients and Mel Spectrograms'. Department of Statistics, University of California.
> <br> **LINK**: https://arxiv.org/abs/2401.04737
