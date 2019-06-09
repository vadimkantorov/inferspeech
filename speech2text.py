import argparse
import numpy as np
import h5py
import scipy.io.wavfile
import python_speech_features
import torch
import torch.nn as nn

def load_model(model_weights, batch_norm_eps = 0.001, dtype = torch.float32):
	def conv_block(kernel_size, num_channels, stride = 1, dilation = 1, repeat = 1, padding = 0):
		modules = []
		for i in range(repeat):
			conv = nn.Conv1d(num_channels[0] if i == 0 else num_channels[1], num_channels[1], kernel_size = kernel_size, stride = stride, dilation = dilation, padding = padding)
			modules.append(conv)
			modules.append(nn.Hardtanh(0, 20, inplace = True))
		return nn.Sequential(*modules)

	model = nn.Sequential(
		conv_block(kernel_size = 11, num_channels = (64, 256), stride = 2, padding = 5),
		conv_block(kernel_size = 11, num_channels = (256, 256), repeat = 3, padding = 5),
		conv_block(kernel_size = 13, num_channels = (256, 384), repeat = 3, padding = 6),
		conv_block(kernel_size = 17, num_channels = (384, 512), repeat = 3, padding = 8),
		conv_block(kernel_size = 21, num_channels = (512, 640), repeat = 3, padding = 10),
		conv_block(kernel_size = 25, num_channels = (640, 768), repeat = 3, padding = 12),
		conv_block(kernel_size = 29, num_channels = (768, 896), repeat = 1, padding = 28, dilation = 2),
		conv_block(kernel_size = 1, num_channels = (896, 1024), repeat = 1),
		nn.Conv1d(1024, 29, 1)
	)

	state_dict = {}
	with h5py.File(model_weights) as h:
		to_tensor = lambda path: torch.from_numpy(np.asarray(h[path])).to(dtype)
		for param_name in model.state_dict():
			ij = [int(c) for c in param_name if c.isdigit()]
			if len(ij) > 1:
				kernel, moving_mean, moving_variance, beta, gamma = [to_tensor(f'ForwardPass/w2l_encoder/conv{1 + ij[0]}{1 + ij[1] // 2}/{suffix}') for suffix in ['kernel', '/bn/moving_mean', '/bn/moving_variance', '/bn/beta', '/bn/gamma']]
				factor = gamma * (moving_variance + batch_norm_eps).rsqrt()
				kernel *= factor
				bias = beta - moving_mean * factor
			else:
				kernel, bias = [to_tensor(f'ForwardPass/fully_connected_ctc_decoder/fully_connected/{suffix}') for suffix in ['kernel', 'bias']]	
				kernel.unsqueeze_(0)
			state_dict[param_name] = kernel.permute(2, 1, 0) if 'weight' in param_name else bias
	model.load_state_dict(state_dict)
	return model

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_path', default = 'test.wav')
	parser.add_argument('--weights', default = 'w2l_plus_large_mp.h5')
	args = parser.parse_args()

	dtype = torch.float32

	sample_rate, signal = scipy.io.wavfile.read(args.input_path)
	features = torch.from_numpy(python_speech_features.logfbank(signal=signal,
                            samplerate=sample_rate,
                            winlen=20e-3,
                            winstep=10e-3,
                            nfilt=64,
                            nfft=512,
                            lowfreq=0, highfreq=sample_rate/2,
                            preemph=0.97)).to(dtype)

	batch = features.t().unsqueeze(0)
	model = load_model(args.weights, dtype = dtype)
	scores = model(batch).squeeze(0)
	
	decoded_greedy = scores.argmax(dim = 0).tolist()
	decoded_text = ''.join({0 : ' ', 27 : "'", 28 : '|'}.get(c, chr(c - 1 + ord('a'))) for c in decoded_greedy)
	postproc_text = ''.join(c for i, c in enumerate(decoded_text) if i == 0 or c != decoded_text[i - 1]).replace('|', '')

	print(postproc_text)
