import sys
import argparse
import math
import numpy as np
import h5py
import scipy.io.wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_model_en_jasper(model_weights, batch_norm_eps = 0.001, num_classes = 29, ABC = " ABCDEFGHIJKLMNOPQRSTUVWXYZ'|"):
	class conv_block(nn.Module):
		def __init__(self, kernel_size, num_channels, stride = 1, dilation = 1, padding = 0, repeat = 1, num_channels_residual = []):
			super(conv_block, self).__init__()
			self.conv = nn.ModuleList([nn.Conv1d(num_channels[0] if i == 0 else num_channels[1], num_channels[1], kernel_size = kernel_size, stride = stride, dilation = dilation, padding = padding) for i in range(repeat)])
			self.conv_residual = nn.ModuleList([nn.Conv1d(in_channels, num_channels[1], kernel_size = 1) for in_channels in num_channels_residual])

	class JasperNetDenseResidual(nn.Module):
		def __init__(self, num_classes):
			super(JasperNetDenseResidual, self).__init__()
			self.blocks = nn.ModuleList([
				conv_block(kernel_size = 11, num_channels = (64, 256), padding = 5, stride = 2),

				conv_block(kernel_size = 11, num_channels = (256, 256), padding = 5, repeat = 5, num_channels_residual = [256]),
				conv_block(kernel_size = 11, num_channels = (256, 256), padding = 5, repeat = 5, num_channels_residual = [256, 256]),

				conv_block(kernel_size = 13, num_channels = (256, 384), padding = 6, repeat = 5, num_channels_residual = [256, 256, 256]),
				conv_block(kernel_size = 13, num_channels = (384, 384), padding = 6, repeat = 5, num_channels_residual = [256, 256, 256, 384]),

				conv_block(kernel_size = 17, num_channels = (384, 512), padding = 8, repeat = 5, num_channels_residual = [256, 256, 256, 384, 384]),
				conv_block(kernel_size = 17, num_channels = (512, 512), padding = 8, repeat = 5, num_channels_residual = [256, 256, 256, 384, 384, 512]),

				conv_block(kernel_size = 21, num_channels = (512, 640), padding = 10, repeat = 5, num_channels_residual = [256, 256, 256, 384, 384, 512, 512]),
				conv_block(kernel_size = 21, num_channels = (640, 640), padding = 10, repeat = 5, num_channels_residual = [256, 256, 256, 384, 384, 512, 512, 640]),

				conv_block(kernel_size = 25, num_channels = (640, 768), padding = 12, repeat = 5, num_channels_residual = [256, 256, 256, 384, 384, 512, 512, 640, 640]),
				conv_block(kernel_size = 25, num_channels = (768, 768), padding = 12, repeat = 5, num_channels_residual = [256, 256, 256, 384, 384, 512, 512, 640, 640, 768]),

				conv_block(kernel_size = 29, num_channels = (768, 896), padding = 28, dilation = 2),
				conv_block(kernel_size = 1, num_channels = (896, 1024)),

				nn.Conv1d(1024, num_classes, 1)
			])

		def forward(self, x):
			residual = []
			for i, block in enumerate(self.blocks[:-1]):
				for j in range(len(block.conv) - 1):
					x = F.relu(block.conv[j](x), inplace = True)
				x = block.conv[-1](x)
				for r, conv in zip(residual if i < len(self.blocks) - 3 else [], block.conv_residual):
					x = x + conv(r)
				x = F.relu(x, inplace = True)
				residual.append(x)
			return self.blocks[-1](x)

	model = JasperNetDenseResidual(num_classes = len(ABC))
	h = h5py.File(model_weights)
	to_tensor = lambda path: torch.from_numpy(np.asarray(h[path])).to(torch.float32)
	state_dict = {}
	for param_name, param in model.state_dict().items():
		ij = [int(c) for c in param_name.split('.') if c.isdigit()]
		weight, bias = None, None
		if len(ij) > 1:
			weight, moving_mean, moving_variance, gamma, beta = [to_tensor(f'ForwardPass/w2l_encoder/conv{1 + ij[0]}{1 + ij[1]}/{suffix}') for suffix in ['kernel', 'bn/moving_mean', 'bn/moving_variance', 'bn/gamma', 'bn/beta']] if 'residual' not in param_name else [to_tensor(f'ForwardPass/w2l_encoder/conv{1 + ij[0]}5/{suffix}') for suffix in [f'res_{ij[1]}/kernel', f'res_bn_{ij[1]}/moving_mean', f'res_bn_{ij[1]}/moving_variance', f'res_bn_{ij[1]}/gamma', f'res_bn_{ij[1]}/beta']]
			weight, bias = fuse_conv_bn(weight.permute(2, 1, 0), moving_mean, moving_variance, gamma, beta, batch_norm_eps = batch_norm_eps)
		else:
			weight, bias = [to_tensor(f'ForwardPass/fully_connected_ctc_decoder/fully_connected/{suffix}') for suffix in ['kernel', 'bias']]
			weight = weight.t().unsqueeze(-1)

		state_dict[param_name] = (weight if 'weight' in param_name else bias).to(param.dtype)
	model.load_state_dict(state_dict)

	def frontend(signal, sample_freq, window_size=20e-3, window_stride=10e-3, dither = 1e-5, window_fn = np.hanning, num_features = 64):
		def get_melscale_filterbanks(sr, n_fft, n_mels, fmin, fmax, dtype = np.float32):
			def hz_to_mel(frequencies):
				frequencies = np.asanyarray(frequencies)
				f_min = 0.0
				f_sp = 200.0 / 3

				mels = (frequencies - f_min) / f_sp

				min_log_hz = 1000.0                         # beginning of log region (Hz)
				min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
				logstep = np.log(6.4) / 27.0                # step size for log region

				if frequencies.ndim:
					log_t = (frequencies >= min_log_hz)
					mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep
				elif frequencies >= min_log_hz:
					mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

				return mels

			def mel_to_hz(mels):
				mels = np.asanyarray(mels)

				f_min = 0.0
				f_sp = 200.0 / 3
				freqs = f_min + f_sp * mels

				min_log_hz = 1000.0                         # beginning of log region (Hz)
				min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
				logstep = np.log(6.4) / 27.0                # step size for log region

				if mels.ndim:
					log_t = (mels >= min_log_mel)
					freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
				elif mels >= min_log_mel:
					freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

				return freqs

			n_mels = int(n_mels)
			weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

			fftfreqs = np.linspace(0, float(sr) / 2, int(1 + n_fft//2),endpoint=True)
			mel_f = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))

			fdiff = np.diff(mel_f)
			ramps = np.subtract.outer(mel_f, fftfreqs)

			for i in range(n_mels):
				lower = -ramps[i] / fdiff[i]
				upper = ramps[i+2] / fdiff[i+1]
				weights[i] = np.maximum(0, np.minimum(lower, upper))

			enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
			weights *= enorm[:, np.newaxis]
			return torch.from_numpy(weights)

		signal = signal / (signal.abs().max() + 1e-5)
		audio_duration = len(signal) * 1.0 / sample_freq
		n_window_size = int(sample_freq * window_size)
		n_window_stride = int(sample_freq * window_stride)
		num_fft = 2**math.ceil(math.log2(window_size*sample_freq))

		signal += dither * torch.randn_like(signal)
		S = torch.stft(signal, num_fft, hop_length=int(window_stride * sample_freq), win_length=int(window_size * sample_freq), window = torch.hann_window(int(window_size * sample_freq)).type_as(signal), pad_mode = 'reflect', center = True).pow(2).sum(dim = -1)
		mel_basis = get_melscale_filterbanks(sample_freq, num_fft, num_features, fmin=0, fmax=int(sample_freq/2)).type_as(S)

		features = torch.log(torch.matmul(mel_basis, S) + 1e-20)
		mean = features.mean(dim = 1, keepdim = True)
		std_dev = features.std(dim = 1, keepdim = True)
		return (features - mean) / std_dev

	return frontend, model, (lambda c: ABC[c]), ABC.index

def load_model_ru_w2l(model_weights, batch_norm_eps = 1e-05, ABC = '|АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ2* '):
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
		nn.Conv1d(2048, len(ABC), kernel_size=1, stride=1)
	)

	h = h5py.File(model_weights)
	to_tensor = lambda path: torch.from_numpy(np.asarray(h[path])).to(torch.float32)
	state_dict = {}
	for param_name, param in model.state_dict().items():
		ij = [int(c) for c in param_name if c.isdigit()]
		if len(ij) > 1:
			weight, moving_mean, moving_variance, gamma, beta = [to_tensor(f'rnns.{ij[0] * 3}.weight')] + [to_tensor(f'rnns.{ij[0] * 3 + 1}.{suffix}') for suffix in ['running_mean', 'running_var', 'weight', 'bias']]
			weight, bias = fuse_conv_bn(weight, moving_mean, moving_variance, gamma, beta, batch_norm_eps = batch_norm_eps)
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
		return torch.stft(signal, nfft, win_length = win_length, hop_length = hop_length, window = torch.hann_window(nfft).type_as(signal), pad_mode = 'reflect', center = True).pow(2).sum(dim = -1).sqrt()

	return frontend, model, (lambda c: ABC[c]), ABC.index

def load_model_en_w2l(model_weights, batch_norm_eps = 0.001, ABC = " ABCDEFGHIJKLMNOPQRSTUVWXYZ'|"):
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
		nn.Conv1d(1024, len(ABC), 1)
	)

	h = h5py.File(model_weights)
	to_tensor = lambda path: torch.from_numpy(np.asarray(h[path])).to(torch.float32)
	state_dict = {}
	for param_name, param in model.state_dict().items():
		ij = [int(c) for c in param_name if c.isdigit()]
		if len(ij) > 1:
			weight, moving_mean, moving_variance, gamma, beta = [to_tensor(f'ForwardPass/w2l_encoder/conv{1 + ij[0]}{1 + ij[1] // 2}/{suffix}') for suffix in ['kernel', 'bn/moving_mean', 'bn/moving_variance', 'bn/gamma', 'bn/beta']]
			weight, bias = fuse_conv_bn(weight.permute(2, 1, 0), moving_mean, moving_variance, gamma, beta, batch_norm_eps = batch_norm_eps)
		else:
			weight, bias = [to_tensor(f'ForwardPass/fully_connected_ctc_decoder/fully_connected/{suffix}') for suffix in ['kernel', 'bias']]	
			weight = weight.t().unsqueeze(-1)
		state_dict[param_name] = (weigth if 'weight' in param_name else bias).to(param.dtype)
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
		mel_basis = get_melscale_filterbanks(nfilt, nfft, sample_rate).type_as(pspec)
		features = torch.log(torch.add(torch.matmul(mel_basis, pspec), 1e-20)) 
		return (features - features.mean()) / features.std()

	return frontend, model, (lambda c: ABC[c]), ABC.index

def fuse_conv_bn(weight, moving_mean, moving_variance, gamma, beta, batch_norm_eps):
	factor = gamma * (moving_variance + batch_norm_eps).rsqrt()
	weight *= factor.view(-1, *([1] * (weight.dim() - 1)))
	bias = beta - moving_mean * factor
	return weight, bias

def decode_greedy(scores, idx2chr):
	decoded_greedy = scores.argmax(dim = 0).tolist()
	decoded_text = ''.join(map(idx2chr, decoded_greedy))
	return ''.join(c for i, c in enumerate(decoded_text) if (i == 0 or c != decoded_text[i - 1]) and c != '|')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', default = 'w2l_plus_large_mp.h5')
	parser.add_argument('--model', default = 'en_w2l', choices = ['en_w2l', 'ru_w2l', 'en_jasper'])
	parser.add_argument('-i', '--input_path')
	parser.add_argument('--onnx')
	parser.add_argument('--tfjs')
	parser.add_argument('--tfjs_quantization_dtype', default = None, choices = ['uint8', 'uint16', None])
	parser.add_argument('--device', default = 'cpu')
	args = parser.parse_args()

	torch.set_grad_enabled(False)
	frontend, model, idx2chr, chr2idx = dict(en_w2l = load_model_en_w2l, en_jasper = load_model_en_jasper, ru_w2l = load_model_ru_w2l)[args.model](args.weights)

	if args.input_path:
		sample_rate, signal = scipy.io.wavfile.read(args.input_path)
		assert sample_rate in [8000, 16000]
		features = frontend(torch.from_numpy(signal).to(torch.float32), sample_rate)
		scores = model.to(args.device)(features.unsqueeze(0).to(args.device)).squeeze(0)
		print(decode_greedy(scores, idx2chr))

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
