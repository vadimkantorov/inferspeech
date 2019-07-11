import math
import argparse
import scipy.io.wavfile
import librosa; import librosa.display; import numpy as np
import torch
import torch.nn.functional as F
import speech2text

def get_filterbanks(nfilt, nfft, samplerate):
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

def frontend(sample_rate, signal, nfft = 512, nfilt = 64, preemph = 0.97):
	preemphasis = lambda signal, coeff: torch.cat([signal[:1], torch.sub(signal[1:], torch.mul(signal[:-1], coeff))])
	window_length = 20 * sample_rate // 1000
	hop_length = 10 * sample_rate // 1000
	pspec = torch.stft(preemphasis(signal, preemph), nfft, win_length = window_length, hop_length = hop_length, window = torch.hann_window(window_length), pad_mode = 'constant', center = False).pow(2).sum(dim = -1).t() / nfft
	mel_basis = get_filterbanks(nfilt, nfft, sample_rate).t()
	features = torch.log(torch.add(torch.matmul(pspec, mel_basis), 1e-20))
	#return features.t()
	return (features.t() - features.mean()) / features.std()

def vis(mel, scores, saliency):
	import matplotlib; matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	postproc = lambda decoded: ''.join('.' if c == '|' else c if i == 0 or c == ' ' or c != idx2chr(decoded[i - 1]) else '_' for i, c in enumerate(''.join(map(idx2chr, decoded))))
	postproc_greedy = postproc(scores.argmax(dim = 0).tolist())

	def merge_epsilon(scores):
		decoded = list(map(idx2chr, scores.argmax(dim = 0).tolist()))
		scores = scores.clone()
		noneps = decoded[0]
		for i, c in enumerate(decoded):
			if i > 0:
				scores[chr2idx(noneps), i] += scores[chr2idx('|'), i]
				if c != '|':
					noneps = c
		return scores[:-1]

	entropy = lambda x, dim, eps = 1e-15: -(x * (x + eps).log()).sum(dim = dim)
	
	plt.subplot(511)
	#log_spec = 10 * math.log10(math.e) * (mel - mel.max())
	#D = log_spec.clamp(min = log_spec.max() - 80.0)
	#librosa.display.specshow(D.numpy(), x_axis='time', y_axis='mel', fmax = 8000)
	plt.imshow(mel, origin = 'lower', aspect = 'auto')
	plt.title('Mel log-spectrogram')

	plt.subplot(512)
	plt.imshow((saliency - saliency.min(dim = 0).values) / (saliency.max(dim = 0).values - saliency.min(dim = 0).values), origin = 'lower', aspect = 'auto')
	plt.title('Gradient')

	plt.subplot(513)
	plt.imshow(scores, origin = 'lower', aspect = 'auto')
	plt.title('Scores')

	scores = merge_epsilon(scores)

	plt.subplot(514)
	plt.plot(entropy(F.softmax(scores, dim = 0), dim = 0).numpy(), linewidth = 0.3)
	plt.ylim([0, 1])
	plt.xticks(torch.arange(scores.shape[-1]), postproc_greedy)
	plt.gca().tick_params(axis='both', which='both', labelsize=0.5, length = 0)
	plt.title('Entropy')

	plt.subplot(515)
	best, next = scores.topk(2, dim = 0).values
	plt.plot(best.numpy(), linewidth = 0.3, color = 'r')
	plt.plot(next.numpy(), linewidth = 0.3, color = 'b')
	plt.title('Margin')

	plt.savefig('vis.png', bbox_inches = 'tight', dpi = 800)

idx2chr = lambda c: {0 : ' ', 27 : "'", 28 : '|'}.get(c, chr(c - 1 + ord('A')))
chr2idx = lambda c: {' ' : 0, "'" : 27, '|' : 28}.get(c, 1 + ord(c) - ord('A'))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', default = 'w2l_plus_large_mp.h5')
	parser.add_argument('-i', '--input_path', default = '121-123852-0004.wav')
	parser.add_argument('-o', '--output_path', default = 'dream.wav')
	parser.add_argument('-t', '--transcript', default = '''MY HEART DOTH PLEAD THAT THOU IN HIM DOST LIE A CLOSET NEVER PIERC'D WITH CRYSTAL EYES BUT THE DEFENDANT DOTH THAT PLEA DENY AND SAYS IN HIM THY FAIR APPEARANCE LIES''')
	parser.add_argument('--device', default = 'cpu')
	parser.add_argument('--num_iter', default = 1, type = int)
	parser.add_argument('--lr', default = 1e6, type = float)
	args = parser.parse_args()

	model = speech2text.load_model(args.weights).to(args.device)
	sample_rate, signal = scipy.io.wavfile.read(args.input_path)
	signal = torch.from_numpy(signal).to(torch.float32).to(args.device).requires_grad_()
	labels = torch.IntTensor(list(map(chr2idx, args.transcript))).to(args.device)

	print('!', args.transcript)
	for i in range(args.num_iter):
		batch = frontend(sample_rate, signal).unsqueeze(0).requires_grad_(); batch.retain_grad()
		scores = model(batch)

		decoded_greedy = scores.squeeze(0).argmax(dim = 0).tolist()
		decoded_text = ''.join(map(idx2chr, decoded_greedy))
		print(i, ''.join(c for i, c in enumerate(decoded_text) if i == 0 or c != decoded_text[i - 1]).replace('|', ''))
		loss = F.ctc_loss(F.log_softmax(scores, dim = 1).permute(2, 0, 1), labels.unsqueeze(0), torch.IntTensor([scores.shape[-1]]).to(args.device), torch.IntTensor([len(labels)]).to(args.device), blank = 28)

		model.zero_grad()
		loss.backward()
		
		gradients, activations = batch.grad, batch
		weights = gradients.clamp(min = 0).mean(dim = -1, keepdim = True)
		saliency = -gradients #weights * activations

		vis(batch[0].detach().cpu(), scores[0].detach().cpu(), saliency[0].detach().cpu())
		

		#signal.data.sub_(signal.grad.data.mul_(args.lr))
		#signal.grad.data.zero_()
		print(i, float(loss))

	scipy.io.wavfile.write(args.output_path, sample_rate, signal.detach().cpu().to(torch.int16).numpy())
