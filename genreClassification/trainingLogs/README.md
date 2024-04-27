# Training logs
_All training logs are for 5-second Mel-spectrogram inputs only._

Logs of training for various models (not just the ones defined in `ImportsForModelHandling.ipynb`. These logs were taken when experimenting with different model architectures. The goal was to reach a sufficiently accurate (at least 90% training accuracy) and least overfitting architecture. While I have not maintained all the exact architectures, the following comparisons were made.

- With batch normalisation vs. without batch normalisation
- Using 1 or more convolutional layers with $3 \times 3$ kernels and $3 \times 3$ max-pooling size

Note the following:

- `2x2` in the log file name means all convolutional layers used had $2 \times 2$ kernels and max-pooling size
- `3x3_1_conv` means 1 convolutional layer (from the top) had $3 \times 3$ kernel and $3 \times 3$ max-pooling size
- `3x3_2_conv` means 2 convolutional layers (from the top) had $3 \times 3$ kernel and $3 \times 3$ max-pooling size
- `_withLearningRate` tag indicates that it is a copy of the verbose training outputs with learning rates across epochs
- The various tags such as `n_fft-1024` indicate the audio parameters used for the input data

The learning rates are relevant because they were set to decay upon reaching a plateau. The plateau condition was checked every 2 epochs.
