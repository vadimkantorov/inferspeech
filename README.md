# speech2text.py
This is a PyTorch inference script for the NVidia openseq2seq's [wav2letter model](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/wave2letter.html) to PyTorch. 

The [pretrained model weights for English](https://github.com/vadimkantorov/inferspeech/releases/download/pretrained/w2l_plus_large_mp.h5) were exported from a TensorFlow [checkpoint](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/wave2letter.html#training) to HDF5 using a little [tfcheckpoint2pytorch](https://github.com/vadimkantorov/tfcheckpoint2pytorch) script that I wrote.

**Limitations:** not ready for production, usesfloat32 weights; does not use gpu; uses greedy decoder; does not chunk the input

**Dependencies:** `pytorch` (cpu version is OK), `pytorch_speech_features`, `numpy`, `scipy`, `h5py`; optional dependencies for saving the model weights to tfjs format: `tensorflow` v1.13.1 (install as `pip3 install tensorflow==1.13.1`), tensorflowjs (install as `pip3 install tensorflowjs --no-deps`, otherwise it would upgrade your TensorFlow from v1 to v2 and break everything)

The credit for the original [wav2letter++ model](https://arxiv.org/abs/1812.07625) goes to awesome Facebook AI Research scientists.

# Example
```shell
# download the pretrained model weights
wget https://github.com/vadimkantorov/inferspeech/releases/download/pretrained/w2l_plus_large_mp.h5

# transcribe a wav file
python3 speech2text.py --weights w2l_plus_large_mp.h5 -i test.wav

# save the model to ONNX format
python3 speech2text.py --weights w2l_plus_large_mp.h5 --onnx w2l_plus_large_mp.onnx

# save the model to TensorFlow.js format
python3 speech2text.py --weights w2l_plus_large_mp.h5 --tfjs w2l_plus_large_mp.tfjs
```
