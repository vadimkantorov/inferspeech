import matplotlib.pyplot as plt
import torch
import speech2text

frontend, model, idx2chr, chr2idx = speech2text.load_model_en_jasper('jasper10x5_LibriSpeech_nvgrad_masks.h5')

#model = torch.load('w2l_plus_large_mp.pt')

convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv1d) and m.kernel_size[0] > 1]
print('\n'.join(str(c) for c in convs))
plt.figure(figsize = (6, 15))

for i, conv in enumerate(convs, start = 1):
	weight = conv.weight
	plt.subplot(len(convs), 1, i)
	plt.imshow(weight.abs().mean(dim = 1).detach().numpy(), origin = 'lower', aspect = 'auto')
	plt.gca().tick_params(axis='both', which='both', labelsize=5, length = 0)
#	plt.title(f'out_channels = {weight.shape[0]} | kernel_size = {weight.shape[-1]}')

plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.8, wspace=0.4)
plt.savefig('vis.png', dpi = 150)
