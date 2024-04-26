# Ambience-to-Music Neural Style Transfer

# Previous work
Previous similar project: https://www.youtube.com/watch?v=jmB4IhfGhuY

# Google Colab notebook
https://drive.google.com/file/d/1NltGH6EDjP2ZkHsMax9aB90_pkpDRFkR/view?usp=sharing

# Model parameters chosen
- **Model type**: Convolutional neural network
- **Model purpose**: Genre classification
- **Model input**: Melspectrograms with the following parameters:
    - Sampling rate: 22050
    - FFT window size: 1024
    - Hop length: 256
    - Number of mel bands: 384
- **Remarks**: Weights with best validation accuracy during training were chosen

The melspectrogram parameters (apart from sampling rate) were chosen based on the most memory-efficient parameters observed to retain audio quality upon melspectrogram's reconversion, given the sampling rate 22050 Hz. The sampling rate 222050 Hz was chosen because that was the sampling rate of the training audio.

# Key concepts in audio processing
**_I shall also discuss their relevance to my project where necessary_**.

## Audio processing parameters
- Sampling rate is the number of samples each second of audio is divided into
- Window length is the number of samples over which fast Fourier transform (FFT) is applied to the waveform
- A frame is a fixed-length sequence of consecutive samples for which FFT is sampled (hence, each frame is the length of one window)
- Hop length is the distance (in samples) between the starting point of two consecutive frames
- Hence, hop lengths determine how much consecutive frames overlap

## Short-time Fourier transfer (STFT)
The short-time Fourier transform (STFT) is a Fourier-related transform used to determine the sinusoidal frequency and phase content of local sections of a signal (ex. an audio file) as it changes over time. In practice, the procedure for computing STFTs is to divide a longer time signal (ex. a longer audio file) into shorter segments (windows) of equal length and then compute the Fourier transform separately on each shorter segment. Conceptually, we are taking the discrete Fourier transform (DFT) of successive windowed regions of the original signal (note that these regions overlap if the hop length is smaller than the window length).


> REFERENCES:
> - https://en.wikipedia.org/wiki/Short-time_Fourier_transform
> - https://sigproc.mit.edu/_static/fall19/lectures/lec09a.pdf

## Melspectrogram
The way we hear frequencies in sound is known as 'pitch'. It is a subjective impression of the frequency. So a high-pitched sound has a higher frequency than a low-pitched sound. Humans do not perceive frequencies linearly. We are more sensitive to differences between lower frequencies than higher frequencies. For example, the pair at 100Hz and 200Hz will sound further apart than the pair at 1000Hz and 1100Hz, even if the actual frequency difference between each pair is the same (100 Hz); you will hardly be able to distinguish between the pair at 10000Hz and 10100Hz. _
However, this may seem less surprising if we realize that the 200Hz frequency is actually double the 100Hz, whereas the 10100Hz frequency is only 1% higher than the 10000Hz frequency.

This is how humans perceive frequencies; we hear them on a logarithmic scale rather than a linear scale. The mel scale was developed to take this into account by conducting experiments with a large number of listeners. It is a scale of pitches such that each would be unit is judged by human listeners to be equal in pitch distance from the next.

> REFERENCE: https://towardsdatascience.com/audio-deep-learning-made-simple-part-2-why-mel-spectrograms-perform-better-aad889a93505

_Why use melspectrograms?_

Modelling audio based on how humans perceive audio is a practical starting point to try to model audio recognition according to human needs (e.g. genre, key signature, etc.). Instead of frequencies, we consider mel bands that convey the most significant frequencies to a human ear and their relationships between each other with respect to human perception; this allows us to reduce data while preserving mmost or all the relevant information needed for audio recognition tasks.

**NOTE**: We could also scale signal power with decibel (a logarithmic scale) instead of amplitude (a linear scale), the former modelling human perception more closely. However, I shall not do this because I intend eventually to reconstruct the audio signals from the melspectrograms passed and modified through my models, and I found that converting the signal power to decibel scale led to too much information loss and led to poor signal reconstruction.

## Mel-frequency cepstrum coefficient (MFCC)
The mel frequency cepstral coefficients (MFCCs) of an audio signal are a small set of features (usually about 10–20) which describe the overall shape of the spectral envelope (i.e. the overall distribution of a signal's power over its frequency). More precisely, we have the following chain of concepts:

- Power spectral density describes how the power of a signal or time series is distributed over frequency
- Mel-frequency cepstrum (MFC) is a representation of the short-term power spectrum of a sound
- Mel-frequency cepstral coefficients (MFCCs) are coefficients that collectively make up an MFC

> REFERENCES:
> - https://medium.com/@derutycsl/intuitive-understanding-of-mfccs-836d36a1f779
> - https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
> - https://en.wikipedia.org/wiki/Spectral_density#Power_spectral_density

**Relevant questions**...

_Why bother MFCC?_

Generally, the essential music dimension used in content-based approaches is timbre. Timbre can be defined as "the character or quality of a musical sound or voice as distinct from its pitch and intensity". It depends on the perception of the quality of sounds, which is related to the used musical instruments, with possible audio effects, and to the playing techniques. In the field of speech recognition, the mel-frequency cepstral coefficients (MFCCs) have been widely used to model important characteristics in speech. Since modelling speech characteristics and timbre are similar, the use of MFCCs has been extended with success in the field of music similarity (e.g. genre classification, which depends to an extent on timbre).

> REFERENCE: de Leon, F., Martinez, K. (2012). 'Enhancing Timbre Model Using MFCC And Its Time Derivatives For Music Similarity Estimation'. 20th European Signal Processing Conference (EUSIPCO 2012)

**NOTE**: Whether MFCCs models timbre is a somewhat open question, but it is reasonable to assume that they do to some extent.


_Why do I care about timbre?_

I aim to transfer the timbral and acoustic qualities of an ambience-type audio to a musical piece. Hence, I care about timbre and about a model that can - to some extent - model timbral and acoustic qualities of sound and music especially (we shall use a genre classifier to achieve such modelling).

# Technical notes
## Using `GradientTape` in Tensorflow
### Using Tensorflow variables to maintain record of operations under `GradientTape`
Tensorflow's `GradientTape` record operations on a given set of variables (given within its context) for automatic differentiation. Trainable variables (created by `tensorflow.Variable` or `tensorflow.compat.v1.get_variable`, where `trainable=True` is default in both cases) are automatically watched. Tensors can be manually watched by invoking the watch method on this context manager. Now, note that since `GradientTape` only watches trainable Tensorflow variables, converting the watched variables at any point within `GradientTape`'s context to other data types (e.g. NumPy arrays) erases previous operations, thus erasing resetting its gradient to zero. Needless to say, operations on variables of other data types are not recorded by `GradientTape`, and thus, do not contribute to the gradient's calculation.

### Use of `tensorflow.GradientTape.watch`
Tensorflow's `GradientTape` watches all trainable variables under its context by default. However, if I have given the optional argument `watch_accessed_variables` as false (to allow for more fine-grained control over the variables observed, if needed), I have to specify which variables I want `GradientTape` to record operations for, a `tensorflow.GradientTape.watch` function call would be needed. By default, however, the function call is redundant.

> REFERENCE: https://www.tensorflow.org/api_docs/python/tf/GradientTape

### Use of optional argument `persistent=True` in `GradientTape`
A persistent tape is a tape that can be used multiple times to compute multiple gradients. By default, a tape is not persistent and can only be used once. If I intend to compute the gradient for two variables, so I need a persistent tape.

> REFERENCE: https://www.geeksforgeeks.org/tf-gradienttape-in-tensorflow/

# Ambience style to musical content

To discuss:

- Why choose fewer convolutional layers for genre classifier
    - I want to study the effects of the more fine-grained features (if I have many convolutional layers, the prediction will be based on more high-level, abstract features, preventing me from studying the lower-level features)
    - Why do I want lower-level features? Because I do not care about accurate genre classification but rather about accurate sound quality and timbre modelling (which is what I want to alter through NST). Hence, if a 3-second segments of a classical piece sounds like jazz due to its sound quality and timbre, I want to treat it as jazz in these respects
  
- Why train on 3-second segments instead of longer (e.g. 30-second) segments?
    - 1: Ambience is more stable for shorter segments, so these is lesser chance of altering content melody during NST
    - 2: I am focusing on low-level features, so broader abstract qualities of the piece are not of interest
    - 3: Training data is augmented, thus training quality is improved
