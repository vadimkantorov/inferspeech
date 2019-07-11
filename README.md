# speech2text.py
This is a PyTorch inference script for the NVidia openseq2seq's [wav2letter model](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/wave2letter.html) to PyTorch. 

The [pretrained model weights for English](https://github.com/vadimkantorov/inferspeech/releases/download/pretrained/w2l_plus_large_mp.h5) were exported from a TensorFlow [checkpoint](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/wave2letter.html#training) to HDF5 using a little [tfcheckpoint2pytorch](https://github.com/vadimkantorov/tfcheckpoint2pytorch) script that I wrote.

**Limitations:** not ready for production, uses float32 weights; does not use gpu; uses greedy decoder; does not chunk the input

**Dependencies:** `pytorch` (cpu version is OK), `numpy`, `scipy`, `h5py`; optional dependencies for saving the model weights to tfjs format: `tensorflow` v1.13.1 (install as `pip3 install tensorflow==1.13.1`), tensorflowjs (install as `pip3 install tensorflowjs --no-deps`, otherwise it would upgrade your TensorFlow from v1 to v2 and break everything) 

The credit for the original [wav2letter++ model](https://arxiv.org/abs/1812.07625) goes to awesome Facebook AI Research scientists.

# Example
```shell
# download the pretrained model weights for English and Russian
wget https://github.com/vadimkantorov/inferspeech/releases/download/pretrained/w2l_plus_large_mp.h5 # English
wget https://github.com/vadimkantorov/inferspeech/releases/download/pretrained/checkpoint_0010_epoch_01_iter_62500.model.h5 # Russian

# download and transcribe a wav file (16 kHz)
# should print: my heart doth plead that thou in him doth lie a closet never pierced with crystal eyes but the defendant doth that plea deny and says in him thy fair appearance lies
wget https://github.com/vadimkantorov/inferspeech/releases/download/pretrained/121-123852-0004.wav
python3 speech2text.py --weights w2l_plus_large_mp.h5 -i 121-123852-0004.wav

# transcribe some Russian wav file
python3 speech2text.py --weights checkpoint_0010_epoch_01_iter_62500.model.h5 --model ru -i some_test.wav

# save the model to ONNX format for inspection with https://lutzroeder.github.io/netron/
python3 speech2text.py --weights w2l_plus_large_mp.h5 --onnx w2l_plus_large_mp.onnx

# save the model to TensorFlow.js format
python3 speech2text.py --weights w2l_plus_large_mp.h5 --tfjs w2l_plus_large_mp.tfjs
```

# Browser demo with TensorFlow.js (work in progress)
```shell
# download and extract the exported tfjs model
https://github.com/vadimkantorov/inferspeech/releases/download/pretrained/w2l_plus_large_mp.tfjs.tar.gz
tar -xf w2l_plus_large_mp.tfjs.tar.gz

# serve the tfjs model and demo.html file
python3 -m http.server

# open the demo at http://localhost:8000/demo.html and transcribe the test file 121-123852-0004.wav
```
