import os
import math
import json
import base64
import argparse
import scipy.io.wavfile
import numpy as np

import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import speech2text

def vis(mel, scores, saliency, mel_dream):
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
	
	plt.figure(figsize=(6, 3))
	plt.subplot(511)
	plt.imshow(mel, origin = 'lower', aspect = 'auto')
	plt.title('Mel log-spectrogram')

	plt.figure(figsize=(6, 3))
	plt.subplot(512)
	plt.imshow(mel, origin = 'lower', aspect = 'auto')
	plt.title('Mel log-spectrogram')

	#plt.subplot(512)
	#plt.imshow((saliency - saliency.min(dim = 0).values) / (saliency.max(dim = 0).values - saliency.min(dim = 0).values), origin = 'lower', aspect = 'auto')
	#plt.title('Gradient')

	plt.subplot(513)
	plt.imshow(scores, origin = 'lower', aspect = 'auto')
	plt.title('Scores')

	scores = merge_epsilon(scores)

	plt.subplot(514)
	plt.plot(entropy(F.softmax(scores, dim = 0), dim = 0).numpy(), linewidth = 0.3)
	plt.ylim([0, 3])
	plt.xticks(torch.arange(scores.shape[-1]), postproc_greedy)
	plt.gca().tick_params(axis='both', which='both', labelsize=0.5, length = 0)
	plt.title('Entropy')

	plt.subplot(515)
	best, next = scores.topk(2, dim = 0).values
	plt.plot(best.numpy(), linewidth = 0.3, color = 'r')
	plt.plot(next.numpy(), linewidth = 0.3, color = 'b')
	plt.title('Margin')

	plt.savefig('vis.jpg', bbox_inches = 'tight', dpi = 150)
	plt.close()
	with open('vis.jpg', 'rb') as f:
		return f.read()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', default = 'checkpoint_0010_epoch_01_iter_62500.model.h5')
	parser.add_argument('-o', '--output_path', default = 'dream.html')
	parser.add_argument('-i', '--input_path', default = 'transcripts.json')

	parser.add_argument('--device', default = 'cuda')
	parser.add_argument('--num_iter', default = 10, type = int)
	parser.add_argument('--lr', default = 1e6, type = float)
	args = parser.parse_args()
	
	ref_tra = list(sorted(json.load(open(args.transcripts)), key = lambda j: j['cer']))

	frontend, model, idx2chr, chr2idx = speech2text.load_model_ru(args.weights)
	model = model.to(args.device)

	vis = open(args.vis or args.transcripts + '.html' , 'w')
	vis.write('<html><head><meta charset="utf-8"></head><body>')

	for i, (reference, transcript, filename, cer) in enumerate(list(map(j.get, ['reference', 'transcript', 'filename', 'cer'])) for j in ref_tra):
		sample_rate, signal = scipy.io.wavfile.read(filename)

		signal = torch.from_numpy(signal).to(torch.float32).to(args.device).requires_grad_()
		labels = torch.IntTensor(list(map(chr2idx, transcript))).to(args.device)

		batch0, scores0 = None, None
		for k in range(args.num_iter):
			batch = frontend(signal, sample_rate).unsqueeze(0).requires_grad_(); batch.retain_grad()
			scores = model(batch)
			hyp = speech2text.decode(scores.squeeze(0), idx2chr)
			loss = F.ctc_loss(F.log_softmax(scores, dim = 1).permute(2, 0, 1), labels.unsqueeze(0), torch.IntTensor([scores.shape[-1]]).to(args.device), torch.IntTensor([len(labels)]).to(args.device), blank = chr2idx('|'))
			model.zero_grad()
			loss.backward()
			signal.data.sub_(signal.grad.data.mul_(args.lr))
			signal.grad.data.zero_()
			if k == 0:
				batch0 = batch.clone()
				scores0 = scores.clone()
			print(k, hyp)
		
		encoded = base64.b64encode(open(filename, 'rb').read()).decode('utf-8').replace('\n', '')
		vis.write(f'<div><h4>{filename} | {cer}</h4>')
		vis.write(f'<h6>original</h6><div><audio controls src="data:audio/wav;base64,{encoded}"></audio></div>')
		vis.write(f'<h6>REF: {reference}</h6>')
		vis.write(f'<h6>HYP: {transcript}</h6>')

		jpeg_bytes = vis(batch0[0].detach().cpu(), scores0[0].detach().cpu(), (-batch0.grad)[0].detach().cpu())
		encoded = base64.b64encode(jpeg_bytes).decode('utf-8').replace('\n', '')
		vis.write(f'<img src="data:image/jpeg;base64,{encoded}"></img>')
		
		#scipy.io.wavfile.write(args.output_path, sample_rate, signal.detach().cpu().to(torch.int16).numpy())
		vis.write('</div>')

	vis.write('</body></html>')
