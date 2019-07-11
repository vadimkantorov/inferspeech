import math
import argparse
import scipy.io.wavfile
import numpy as np

import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import speech2text

import librosa; import librosa.display
 
def vis(mel, scores, saliency):

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

	sample_rate, signal = scipy.io.wavfile.read(args.input_path)

	frontend, model, idx2chr, chr2idx = speech2text.load_model_en(args.weights).to(args.device)
	signal = torch.from_numpy(signal).to(torch.float32).to(args.device).requires_grad_()
	labels = torch.IntTensor(list(map(chr2idx, args.transcript))).to(args.device)

	print('!', args.transcript)
	for i in range(args.num_iter):
		batch = frontend(sample_rate, signal).unsqueeze(0).requires_grad_(); batch.retain_grad()
		scores = model(batch)
		print(decode(scores.squeeze(0), idx2chr))

		loss = F.ctc_loss(F.log_softmax(scores, dim = 1).permute(2, 0, 1), labels.unsqueeze(0), torch.IntTensor([scores.shape[-1]]).to(args.device), torch.IntTensor([len(labels)]).to(args.device), blank = chr2idx('|'))

		model.zero_grad()
		loss.backward()
		
		vis(batch[0].detach().cpu(), scores[0].detach().cpu(), (-batch.grad)[0].detach().cpu())

		#signal.data.sub_(signal.grad.data.mul_(args.lr))
		#signal.grad.data.zero_()
		print(i, float(loss))

	scipy.io.wavfile.write(args.output_path, sample_rate, signal.detach().cpu().to(torch.int16).numpy())
