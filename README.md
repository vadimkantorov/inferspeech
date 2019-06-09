# speech2text.py
This is a PyTorch inference script for the NVidia openseq2seq's [wav2letter model](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/wave2letter.html) to PyTorch. 

The [pretrained model weights for English](https://github.com/vadimkantorov/inferspeech/releases/download/pretrained/w2l_plus_large_mp.h5) were exported from a TensorFlow [checkpoint](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/wave2letter.html#training) to HDF5 using a little [tfcheckpoint2pytorch](https://github.com/vadimkantorov/tfcheckpoint2pytorch) script.

The credit for the original [wav2letter++ model](https://arxiv.org/abs/1812.07625) is to awesome Facebook AI Research scientists.

**Dependencies:** PyTorch (cpu version is OK), pytorch_speech_features, NumPy, scipy, h5py

# Example

```
# download the pretrained model weights
wget https://github.com/vadimkantorov/inferspeech/releases/download/pretrained/w2l_plus_large_mp.h5

# transcribe a wav file
python3 inferspeech.py --weights w2l_plus_large_mp.h5 -i test.wav
```
