import sys
import argparse
import math
import numpy as np
import h5py
import scipy.io.wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_model_ru(model_weights, num_classes = 37, batch_norm_eps = 1e-05, fuse = False):
	def conv_block(kernel_size, num_channels, stride = 1, padding = 0):
		return nn.Sequential(
			nn.Conv1d(num_channels[0], num_channels[1], kernel_size=kernel_size, stride=stride, padding=padding),
			nn.ReLU(inplace = True)
		)

	model = nn.Sequential(
		conv_block(kernel_size = 13, num_channels = (161, 768), stride = 2, padding = 6),
		conv_block(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
		conv_block(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
		conv_block(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
		conv_block(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
		conv_block(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
		conv_block(kernel_size = 13, num_channels = (768, 768), stride = 1, padding = 6),
		conv_block(kernel_size = 31, num_channels = (768, 2048), stride = 1, padding = 15),
		conv_block(kernel_size = 1,  num_channels = (2048, 2048), stride = 1, padding = 0),
		nn.Conv1d(2048, num_classes, kernel_size=1, stride=1)
	)

	h = h5py.File(model_weights)
	to_tensor = lambda path: torch.from_numpy(np.asarray(h[path])).to(torch.float32)
	state_dict = {}
	for param_name, param in model.state_dict().items():
		ij = [int(c) for c in param_name if c.isdigit()]
		if len(ij) > 1:
			weight = to_tensor(f'rnns.{ij[0] * 3}.weight')
			moving_mean, moving_variance, gamma, beta = [to_tensor(f'rnns.{ij[0] * 3 + 1}.{suffix}') for suffix in ['running_mean', 'running_var', 'weight', 'bias']]
			factor = gamma * (moving_variance + batch_norm_eps).rsqrt()
			weight *= factor.view(-1, *([1] * (weight.dim() - 1)))
			bias = beta - moving_mean * factor
		else:
			weight, bias = [to_tensor(f'fc.0.{suffix}') for suffix in ['weight', 'bias']]
		state_dict[param_name] = (weight if 'weight' in param_name else bias).to(param.dtype)
	model.load_state_dict(state_dict)

	def frontend(signal, sample_rate, window_size = 0.020, window_stride = 0.010, window = 'hann'):
		signal = signal / signal.abs().max()
		if sample_rate == 8000:
			signal, sample_rate = F.interpolate(signal.view(1, 1, -1), scale_factor = 2).squeeze(), 16000
		win_length = int(sample_rate * (window_size + 1e-8))
		hop_length = int(sample_rate * (window_stride + 1e-8))
		nfft = win_length
		return torch.stft(signal, nfft, win_length = win_length, hop_length = hop_length, window = torch.hann_window(nfft), pad_mode = 'reflect', center = True).pow(2).sum(dim = -1).sqrt()

	def decode(scores):
		decoded_greedy = scores.argmax(dim = 0).tolist()
		decoded_text = ''.join('|АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ2* '[c] for c in decoded_greedy)
		return ''.join(c for i, c in enumerate(decoded_text) if (i == 0 or c != decoded_text[i - 1]) and c != '|')

	return frontend, model, decode

def load_model_en(model_weights, batch_norm_eps = 0.001, num_classes = 29):
	def conv_block(kernel_size, num_channels, stride = 1, dilation = 1, repeat = 1, padding = 0):
		modules = []
		for i in range(repeat):
			modules.append(nn.Conv1d(num_channels[0] if i == 0 else num_channels[1], num_channels[1], kernel_size = kernel_size, stride = stride, dilation = dilation, padding = padding))
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
		nn.Conv1d(1024, num_classes, 1)
	)

	h = h5py.File(model_weights)
	to_tensor = lambda path: torch.from_numpy(np.asarray(h[path])).to(torch.float32)
	state_dict = {}
	for param_name, param in model.state_dict().items():
		ij = [int(c) for c in param_name if c.isdigit()]
		if len(ij) > 1:
			kernel, moving_mean, moving_variance, beta, gamma = [to_tensor(f'ForwardPass/w2l_encoder/conv{1 + ij[0]}{1 + ij[1] // 2}/{suffix}') for suffix in ['kernel', '/bn/moving_mean', '/bn/moving_variance', '/bn/beta', '/bn/gamma']]
			factor = gamma * (moving_variance + batch_norm_eps).rsqrt()
			kernel *= factor
			bias = beta - moving_mean * factor
		else:
			kernel, bias = [to_tensor(f'ForwardPass/fully_connected_ctc_decoder/fully_connected/{suffix}') for suffix in ['kernel', 'bias']]	
			kernel.unsqueeze_(0)
		state_dict[param_name] = (kernel.permute(2, 1, 0) if 'weight' in param_name else bias).to(param.dtype)
	model.load_state_dict(state_dict)

	def frontend(signal, sample_rate, nfft = 512, nfilt = 64, preemph = 0.97, window_size = 0.020, window_stride = 0.010):
		def get_melscale_filterbanks(nfilt, nfft, samplerate):
			hz2mel = lambda hz: 2595 * math.log10(1+hz/700.)
			mel2hz = lambda mel: torch.mul(700, torch.sub(torch.pow(10, torch.div(mel, 2595)), 1))

			lowfreq = 0
			highfreq = samplerate // 2
			lowmel = hz2mel(lowfreq)
			highmel = hz2mel(highfreq)
			melpoints = torch.linspace(lowmel,highmel,nfilt+2);
			bin = torch.floor(torch.mul(nfft+1, torch.div(mel2hz(melpoints), samplerate))).tolist()

			fbank = torch.zeros([nfilt, nfft // 2 + 1]).tolist()
			for j in range(nfilt):
				for i in range(int(bin[j]), int(bin[j+1])):
					fbank[j][i] = (i - bin[j]) / (bin[j+1]-bin[j])
				for i in range(int(bin[j+1]), int(bin[j+2])):
					fbank[j][i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
			return torch.tensor(fbank)

		preemphasis = lambda signal, coeff: torch.cat([signal[:1], torch.sub(signal[1:], torch.mul(signal[:-1], coeff))])
		win_length = int(sample_rate * (window_size + 1e-8))
		hop_length = int(sample_rate * (window_stride + 1e-8))
		pspec = torch.stft(preemphasis(signal, preemph), nfft, win_length = win_length, hop_length = hop_length, window = torch.hann_window(win_length), pad_mode = 'constant', center = False).pow(2).sum(dim = -1) / nfft 
		mel_basis = get_melscale_filterbanks(nfilt, nfft, sample_rate)
		features = torch.log(torch.add(torch.matmul(mel_basis, pspec), 1e-20)) 
		return (features - features.mean()) / features.std()

	def decode(scores):
		decoded_greedy = scores.argmax(dim = 0).tolist()
		decoded_text = ''.join({0 : ' ', 27 : "'", 28 : '|'}.get(c, chr(c - 1 + ord('a'))) for c in decoded_greedy)
		return ''.join(c for i, c in enumerate(decoded_text) if (i == 0 or c != decoded_text[i - 1]) and c != '|')

	return frontend, model, decode

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', default = 'w2l_plus_large_mp.h5')
	parser.add_argument('--model', default = 'en', choices = ['en', 'ru'])
	parser.add_argument('-i', '--input_path', default = '/mnt/c/Work/sample_ok/sample_ok/1560749203.651862.wav_1_402.79_408.523.wav')
	parser.add_argument('--onnx')
	parser.add_argument('--tfjs')
	parser.add_argument('--tfjs_quantization_dtype', default = None, choices = ['uint8', 'uint16', None])
	args = parser.parse_args()

	torch.set_grad_enabled(False)
	frontend, model, decode = dict(en = load_model_en, ru = load_model_ru)[args.model](args.weights)

	if args.input_path:
		sample_rate, signal = scipy.io.wavfile.read(args.input_path)
		assert sample_rate in [8000, 16000]
		features = frontend(torch.from_numpy(signal).to(torch.float32), sample_rate)
		scores = model(features.unsqueeze(0)).squeeze(0)
		print(decode(scores))

	if args.tfjs:
		# monkey-patching a module to have tfjs converter load with tf v1
		convert_tf_saved_model = None
		sys.modules['tensorflowjs.converters.tf_saved_model_conversion_v2'] = sys.modules[__name__]
		import tensorflowjs 
		import tensorflow.keras as K
		pytorch2keras = lambda module: K.layers.Conv1D(module.out_channels, module.kernel_size, input_shape = (None, module.in_channels), data_format = 'channels_last', strides = module.stride, dilation_rate = module.dilation, padding = 'same', weights = [module.weight.detach().permute(2, 1, 0).numpy(), module.bias.detach().flatten().numpy()]) if isinstance(module, nn.Conv1d) else K.layers.ReLU(threshold = module.min_val, max_value = module.max_val) if isinstance(module, nn.Hardtanh) else K.layers.ReLU() if isinstance(module, nn.ReLU) else K.models.Sequential(list(map(pytorch2keras, module)))
		model, in_channels = pytorch2keras(model), model[0][0].in_channels
		model.build((None, None, in_channels))
		tensorflowjs.converters.save_keras_model(model, args.tfjs, quantization_dtype = getattr(np, args.tfjs_quantization_dtype or '', None))

	if args.onnx:
		# https://github.com/onnx/onnx/issues/740
		batch = torch.zeros(1, 1000, model[0][0].in_channels, dtype = torch.float32)
		torch.onnx.export(model, batch, args.onnx, input_names = ['input'], output_names = ['output'])
