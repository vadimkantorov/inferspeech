import math
import argparse
import scipy.io.wavfile
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
	signal = preemphasis(signal, preemph)
	pspec = torch.stft(signal, nfft, win_length = window_length, hop_length = hop_length, window = torch.hann_window(window_length), pad_mode = 'constant', center = False).pow(2).sum(dim = -1).t() / nfft
	mel_basis = get_filterbanks(nfilt, nfft, sample_rate).t()
	features = torch.log(torch.add(torch.matmul(pspec, mel_basis), 1e-20))
	return (features.t() - features.mean()) / features.std()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', default = 'w2l_plus_large_mp.h5')
	parser.add_argument('-i', '--input_path', default = '121-123852-0004.wav')
	parser.add_argument('-o', '--output_path', default = 'dream.wav')
	parser.add_argument('-t', '--transcript', default = '''MY HEART DOTH PLEAD THAT THOU IN HIM DOST LIE A CLOSET NEVER PIERC'D WITH CRYSTAL EYES BUT THE DEFENDANT DOTH THAT PLEA DENY AND SAYS IN HIM THY FAIR APPEARANCE LIES''')
	parser.add_argument('--device', default = 'cpu')
	parser.add_argument('--num_iter', default = 5, type = int)
	parser.add_argument('--lr', default = 1e6, type = float)
	args = parser.parse_args()

	model = speech2text.load_model(args.weights).to(args.device)
	sample_rate, signal = scipy.io.wavfile.read(args.input_path)
	signal = torch.from_numpy(signal).to(torch.float32).to(args.device).requires_grad_()
	labels = torch.IntTensor([{' ' : 0, "'" : 27}.get(c, 1 + ord(c) - ord('A')) for c in args.transcript]).to(args.device)

	print('R', args.transcript)
	for i in range(args.num_iter):
		batch = frontend(sample_rate, signal).unsqueeze(0)
		scores = model(batch).requires_grad_()

		decoded_greedy = scores.squeeze(0).argmax(dim = 0).tolist()
		decoded_text = ''.join({0 : ' ', 27 : "'", 28 : '|'}.get(c, chr(c - 1 + ord('A'))) for c in decoded_greedy)
		print(i, ''.join(c for i, c in enumerate(decoded_text) if i == 0 or c != decoded_text[i - 1]).replace('|', ''))
		loss = F.ctc_loss(F.log_softmax(scores, dim = 1).permute(2, 0, 1), labels.unsqueeze(0), torch.IntTensor([scores.shape[-1]]).to(args.device), torch.IntTensor([len(labels)]).to(args.device))
		model.zero_grad()
		loss.backward()
		signal.data.sub_(signal.grad.data.mul_(args.lr))
		signal.grad.data.zero_()
		print(i, float(loss))

	scipy.io.wavfile.write(args.output_path, sample_rate, signal.detach().cpu().to(torch.int16).numpy())
