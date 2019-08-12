import os
import io
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

def perturb(batch_first, batch_last, K = 80):
	diff = batch_last - batch_first
	positive = F.relu(diff)
	small = diff.clone()
	small[:, K:] = 0
	large = diff * (diff.abs() < 0.25 * diff.max()).float()
	return batch_first + positive, batch_first + small, batch_first + large

def vis_(batch_first_grad, batch_first, batch_last, scores_first, scores_last, K = 80):
	postproc = lambda decoded: ''.join('.' if c == '|' else c if i == 0 or c == ' ' or c != idx2chr(decoded[i - 1]) else '_' for i, c in enumerate(''.join(map(idx2chr, decoded))))
	normalize_min_max = lambda scores, dim = 0: (scores - scores.min(dim = dim).values) / (scores.max(dim = dim).values - scores.min(dim = dim).values + 1e-16)
	entropy = lambda x, dim, eps = 1e-15: -(x / x.sum(dim = dim, keepdim = True).add(eps) * (x / x.sum(dim = dim, keepdim = True).add(eps) + eps).log()).sum(dim = dim)
	
	plt.figure(figsize=(6, 3))
	def colorbar(): cb = plt.colorbar(); cb.outline.set_visible(False); cb.ax.tick_params(labelsize = 4, length = 0.3)
	title = lambda s: plt.title(s, fontsize = 5)
	ticks = lambda labelsize = 4, length = 1: plt.gca().tick_params(axis='both', which='both', labelsize=labelsize, length=length) or [ax.set_linewidth(0) for ax in plt.gca().spines.values()]
	plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.8, wspace=0.4)

	num_subplots = 9
	plt.subplot(num_subplots, 1, 1)
	plt.imshow(batch_first[:K].log1p(), origin = 'lower', aspect = 'auto'); ticks(); colorbar()
	title('LogSpectrogram, original')

	plt.subplot(num_subplots, 1, 2)
	plt.imshow(batch_last[:K].log1p(), origin = 'lower', aspect = 'auto'); ticks(); colorbar()
	title('LogSpectrogram, dream')

	plt.subplot(num_subplots, 1, 3)
	diff = batch_last - batch_first
	plt.imshow((diff * (diff.abs() < 0.25 * diff.max()).float())[:K].log1p(), origin = 'lower', aspect = 'auto'); ticks(); colorbar()
	title('LogDiff')

	plt.subplot(num_subplots, 1, 4)
	plt.imshow(batch_first_grad[:K].log1p(), origin = 'lower', aspect = 'auto'); ticks(); colorbar()
	title('LogGrad')

	scores_first_01 = normalize_min_max(scores_first)
	scores_last_01 = normalize_min_max(scores_last)
	scores_first_softmax = F.softmax(scores_first, dim = 0)
	scores_last_softmax = F.softmax(scores_last, dim = 0)

	plt.subplot(num_subplots, 1, 5)
	plt.imshow(scores_first_01, origin = 'lower', aspect = 'auto'); ticks(); colorbar()
	title('Scores, original')

	plt.subplot(num_subplots, 1, 6)
	plt.imshow(scores_last_01, origin = 'lower', aspect = 'auto'); ticks(); colorbar()
	title('Scores, dream')

	plt.subplot(num_subplots, 1, 7)
	plt.imshow(scores_last_01 - scores_first_01, origin = 'lower', aspect = 'auto'); ticks(); colorbar()
	title('Diff')

	plt.subplot(num_subplots, 1, 8)
	plt.plot(entropy(scores_first_softmax, dim = 0).numpy(), linewidth = 0.3, color='b')
	plt.plot(entropy(scores_last_softmax, dim = 0).numpy(), linewidth = 0.3, color='r')
	plt.hlines(1.0, 0, scores_first_01.shape[-1]); colorbar()
	plt.xlim([0, scores_first_01.shape[-1]])
	plt.ylim([0, 2.5])
	plt.xticks(torch.arange(scores_first_01.shape[-1]), postproc(scores_last.argmax(dim = 0).tolist()))
	ticks(labelsize=2, length = 0)
	ax = plt.gca().twiny()
	ax.tick_params(axis='x')
	plt.xticks(torch.arange(scores_first_01.shape[-1]), postproc(scores_first.argmax(dim = 0).tolist()))
	ticks(labelsize=2, length = 0)
	title('Entropy')

	plt.subplot(num_subplots, 1, 9)
	plt.plot(F.kl_div(scores_first_softmax, scores_last_softmax, reduction = 'none').sum(dim = 0), linewidth = 0.3, color = 'b')
	plt.plot((scores_last_softmax - scores_first_softmax).abs().sum(dim = 0), linewidth = 0.3, color = 'g'); ticks(); colorbar()
	plt.xlim([0, scores_first_01.shape[-1]])
	plt.ylim([-2, 2])
	title('KL')	

	buf = io.BytesIO()
	plt.savefig(buf, format = 'jpg', bbox_inches = 'tight', dpi = 300)
	plt.close()
	return buf.getvalue()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', default = 'model_checkpoint_0027_epoch_02.model.h5') #'checkpoint_0010_epoch_01_iter_62500.model.h5')
	parser.add_argument('-o', '--output-path', default = 'dream.html')
	parser.add_argument('-i', '--input-path', default = 'transcripts.json')
	parser.add_argument('--device', default = 'cuda')
	parser.add_argument('--num-iter', default = 100, type = int)
	parser.add_argument('--lr', default = 1e6, type = float)
	parser.add_argument('--max-norm', default = 100, type = float)
	args = parser.parse_args()
	
	ref_tra = list(sorted(json.load(open(args.input_path)), key = lambda j: j['cer'], reverse = True))

	frontend, model, idx2chr, chr2idx = speech2text.load_model_ru_w2l(args.weights)
	model = model.to(args.device)

	vis = open(args.output_path , 'w')
	vis.write('<html><head><meta charset="utf-8"></head><body>')

	for i, (reference, transcript, filename, cer) in enumerate(list(map(j.get, ['reference', 'transcript', 'filename', 'cer'])) for j in ref_tra):
		sample_rate, signal = scipy.io.wavfile.read(filename)
		if i > 5: continue

		signal = torch.from_numpy(signal).to(torch.float32).to(args.device).requires_grad_()
		labels = torch.IntTensor(list(map(chr2idx, reference))).to(args.device)

		batch_first, batch_first_grad, batch_last, scores_first, hyp_first, hyp_last, scores_last = None, None, None, None, None, None, None
		for k in range(args.num_iter):
			batch = frontend(signal, sample_rate).unsqueeze(0).requires_grad_(); batch.retain_grad()
			scores = model(batch)

			hyp = speech2text.decode_greedy(scores.squeeze(0), idx2chr)
			loss = F.ctc_loss(F.log_softmax(scores, dim = 1).permute(2, 0, 1), labels.unsqueeze(0), torch.IntTensor([scores.shape[-1]]).to(args.device), torch.IntTensor([len(labels)]).to(args.device), blank = chr2idx('|'))
			print(i, 'Loss:', float(loss))
			if not hyp or (torch.isnan(loss) | torch.isinf(loss)).any():
				continue
			model.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
			signal.data.sub_(signal.grad.data.mul_(args.lr))
			signal.grad.data.zero_()
			if k == 0:
				batch_first = batch.clone()
				batch_first_grad = batch.grad.clone().neg_()
				scores_first = scores.clone()
				hyp_first = hyp
			scores_last = scores.clone()
			batch_last = batch.clone()
			hyp_last = hyp

			print(i, '| #', k, 'REF: ', reference)
			print(i, '| #', k, 'HYP: ', hyp)
			print()


		hyp_positive, hyp_small, hyp_large = [speech2text.decode_greedy(model(x).squeeze(0), idx2chr) for x in perturb(batch_first, batch_last)]
		encoded = base64.b64encode(open(filename, 'rb').read()).decode('utf-8').replace('\n', '')
		vis.write(f'<div><hr /><h4>{filename} | {cer}</h4>')
		vis.write(f'<h6>original</h6><div><audio controls src="data:audio/wav;base64,{encoded}"></audio></div>')
		buf = io.BytesIO()
		scipy.io.wavfile.write(buf, sample_rate, signal.detach().cpu().to(torch.int16).numpy())
		encoded = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
		vis.write(f'<h6>dream</h6><div><audio controls src="data:audio/wav;base64,{encoded}"></audio></div>')
		vis.write(f'<h6>REF: {reference}</h6>')
		vis.write(f'<h6>HYP: {hyp_first}</h6>')
		vis.write(f'<h6>DREAM: {hyp_last}</h6>')
		vis.write(f'<h6>POSITIVE: {hyp_positive}</h6>')
		vis.write(f'<h6>SMALL: {hyp_small}</h6>')
		vis.write(f'<h6>LARGE: {hyp_large}</h6>')

		jpeg_bytes = vis_(*[x[0].detach().cpu() for x in [batch_first_grad, batch_first, batch_last, scores_first, scores_last]])
		encoded = base64.b64encode(jpeg_bytes).decode('utf-8').replace('\n', '')
		vis.write(f'<img src="data:image/jpeg;base64,{encoded}"></img>')
		
		vis.write('</div>')

	vis.write('</body></html>')
