import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F
import bindsnet.encoding.encodings as encodings

# Class for monitoring network activity
class StatsManager:
	def __init__(self, network, classes, grid_assignments):
		self.network = network
		self.classes = classes
		self.n_classes = len(self.classes)

		# Neuron assignments for computing accuracy
		self.assignments = torch.zeros((1, *self.network.neuron_shape))
		for k in range(self.network.neuron_shape[0]):
			for l in range(self.network.neuron_shape[1]):
				self.assignments[:, k, l] = grid_assignments[:, self.network.coord_y_disc[k, l], self.network.coord_x_disc[k, l]]
		self.assignments = self.assignments.view(-1)

		# Variables for tracking accuracy
		self.hits = 0
		self.count = 0
		self.trn_accuracy = []
		self.eval_accuracy = []
		self.best_acc = 0.
		self.best_step = 0

	def update(self, batch):
		labels = batch["label"]
		# Get network predictions
		pred = torch.tensor(self.classes)[torch.argmax(self.get_class_scores(), dim=1).cpu()]
		batch_hits = torch.sum(labels.long() == pred).item()
		batch_count = len(labels)
		# Update hits and count stats
		self.hits += batch_hits
		self.count += batch_count

	def get_class_scores(self):
		# This is the collection of spikes. After the permutation, dimension 0 is the batch dimension, dimension 1 is the time dimension, dimension 2 is the neuron dimension
		spikes = self.network.spike_monitor.get("s")
		spikes = spikes.view(spikes.size(0), spikes.size(1), -1).permute((1, 0, 2))
		n_samples = spikes.size(0)
		spikes = spikes.sum(1)
		scores = torch.zeros(n_samples, self.n_classes, device=spikes.device)
		for i in range(self.n_classes):
			n_assigns = torch.sum(self.assignments == self.classes[i]).float().to(spikes.device)
			if n_assigns > 0:
				indices = torch.nonzero(self.assignments == self.classes[i]).view(-1)
				scores[:, i] = torch.sum(spikes[:, indices], 1) / n_assigns
		return scores
	
	def get_st_resp(self, bin_size=10, tau=20.):
		s = self.network.spike_monitor.get("s")
		time = s.size(0)
		n_samples = s.size(1)
		height = s.size(2)
		width = s.size(3)
		s = s.view(time, n_samples, -1).permute((1, 2, 0)).view(-1, 1, time).float()
		exp_knl = torch.cat([torch.exp(-torch.arange(time, device=s.device) / tau).flip(0), torch.zeros(time - 1, device=s.device)],dim=0).view(1, 1, -1)
		s = torch.conv1d(s, exp_knl, padding=time - 1)
		s = F.avg_pool1d(s, bin_size, ceil_mode=True).squeeze(1).t().view(-1, n_samples, height, width).permute((1, 0, 2, 3))
		num_bins = s.size(1)
		r = torch.zeros(n_samples, num_bins, self.network.inpt_shape[1], self.network.inpt_shape[2], device=s.device)
		c = torch.zeros(n_samples, num_bins, self.network.inpt_shape[1], self.network.inpt_shape[2], device=s.device)
		for k in range(height):
			for l in range(width):
				r[:, :, self.network.coord_y_disc[k, l], self.network.coord_x_disc[k, l]] += s[:, :, k, l]
				c[:, :, self.network.coord_y_disc[k, l], self.network.coord_x_disc[k, l]] += 1
		c[c == 0.] = 1. # Prevent division by zero
		return r / c
	
	def record_accuracy(self, train):
		acc = self.hits / self.count
		if train: self.trn_accuracy.append(acc)
		else: self.eval_accuracy.append(acc)
		self.hits = 0
		self.count = 0
	
	def check_improvement(self):
		if self.eval_accuracy[-1] > self.best_acc:
			self.best_acc = self.eval_accuracy[-1]
			self.best_step = len(self.eval_accuracy)
			return True
		return False

class CustomEncoder():
	def __init__(self, time: int, delay: int, duration: int = 10, dt: float = 1.0, sparsity : float = None, noise_intensity: float = 0., eps=1e-5):
		if time <= delay: time = delay + 1
		self.time = time
		self.delay = delay
		self.duration = duration
		self.dt = dt
		self.sparsity = sparsity
		self.noise_intensity = noise_intensity
		self.p_enc = encodings.poisson
		self.eps=eps

	def __call__(self, img:torch.Tensor):
		res = self.p_enc(torch.ones_like(img) * self.noise_intensity, self.delay, self.dt)
		res = torch.cat((res, torch.zeros(self.time - self.delay, *img.size(), device=res.device).byte()), dim=0)
		if self.sparsity is not None:
			repeats = [self.duration]
			for i in range(len(img.size())): repeats.append(1)
			img_repeat = img.unsqueeze(0).repeat(repeats)
			img_repeat += self.eps*img.std() * torch.randn_like(img_repeat)
			quantile, _ = torch.kthvalue(img_repeat.view(-1), int((1 - self.sparsity) * (img_repeat.view(-1).size(0) - 1)) + 1)
			res[self.delay : self.delay + self.duration] = (img_repeat > quantile).byte()
		else:
			res[self.delay : self.delay + self.duration] =self.p_enc(img, self.duration, self.dt)
		return res

# Convert tensor shape to total tensor size
def shape2size(shape):
	size = 1
	for s in shape: size *= s
	return size

# Return formatted string with time information
def format_time(seconds):
	seconds = int(seconds)
	minutes, seconds = divmod(seconds, 60)
	hours, minutes = divmod(minutes, 60)
	return str(hours) + "h " + str(minutes) + "m " + str(seconds) + "s"

# Print information on the training progress
def print_train_progress(current_epoch, total_epochs, elapsed_time):
	print("\nEPOCH " + str(current_epoch) + "/" + str(total_epochs))

	elapsed_epochs = current_epoch - 1
	if elapsed_epochs == 0:
		elapsed_time_str = "-"
		avg_epoch_duration_str = "-"
		exp_remaining_time_str = "-"
	else:
		avg_epoch_duration = elapsed_time / elapsed_epochs
		remaining_epochs = total_epochs - elapsed_epochs
		elapsed_time_str = format_time(elapsed_time)
		avg_epoch_duration_str = format_time(avg_epoch_duration)
		exp_remaining_time_str = format_time(remaining_epochs * avg_epoch_duration)
	print("Elapsed time: " + elapsed_time_str)
	print("Average epoch duration: " + avg_epoch_duration_str)
	print("Expected remaining time: " + exp_remaining_time_str)

def plot_performance(train_acc_data, val_acc_data, path):
	plt.ioff()
	graph = plt.axes(xlabel='Step', ylabel='Accuracy')
	graph.plot(range(1, len(train_acc_data)+1), train_acc_data, label='Train Acc.')
	graph.plot(range(1, len(val_acc_data)+1), val_acc_data, label='Val. Acc.')
	graph.grid(True)
	graph.legend()
	os.makedirs(os.path.dirname(path), exist_ok=True)
	graph.get_figure().savefig(path, bbox_inches='tight')
	graph.get_figure().clear()
	plt.close(graph.get_figure())

def plot_grid(tensor, path, num_rows=6, num_cols=10):
	#tensor = torch.sigmoid((tensor-tensor.mean())/tensor.std()).permute(0, 2, 3, 1).cpu().detach().numpy()
	tensor = ((tensor - tensor.min())/(tensor.max() - tensor.min())).cpu().detach().numpy()
	plt.ioff()
	fig = plt.figure()
	for i in range(min(tensor.shape[0], num_rows*num_cols)):
		ax1 = fig.add_subplot(num_rows,num_cols,i+1)
		ax1.imshow(tensor[i])
		ax1.axis('off')
		ax1.set_xticklabels([])
		ax1.set_yticklabels([])
	plt.subplots_adjust(wspace=0.1, hspace=0.1)
	os.makedirs(os.path.dirname(path), exist_ok=True)
	fig.savefig(path, bbox_inches='tight')
	plt.close(fig)

def plot_out_resp(series_mean, series_std, series_count, series_names, out_classes, path, ci_lvl=.95):
	plt.ioff()
	graph = plt.axes(xlabel='Output', ylabel='Spike count')
	for i in range(len(series_mean)):
		mean = series_mean[i].cpu().detach().numpy()
		std = series_std[i].cpu().detach().numpy()
		count = series_count[i]
		name = series_names[i]
		series_err = st.t.interval(ci_lvl, count - 1, loc=0., scale=std / count ** 0.5)[1]
		graph.errorbar(out_classes, mean, yerr=series_err, fmt='o', capsize=3, label=name)
	graph.set_xticks(out_classes)
	graph.set_ylim(0., 10.)
	graph.grid(True)
	graph.legend()
	os.makedirs(os.path.dirname(path), exist_ok=True)
	graph.get_figure().savefig(path, bbox_inches='tight')
	graph.get_figure().clear()
	plt.close(graph.get_figure())

def plot_out_dist(series_mean, series_std, out_classes, path, xlim=10., n_bins=100):
	series_mean = series_mean.cpu().detach().numpy()
	series_std = series_std.cpu().detach().numpy()
	plt.ioff()
	graph = plt.axes(xlabel='Spike count', ylabel='Density')
	interv = (torch.arange(n_bins)*xlim/n_bins).numpy()
	dist = [st.norm.pdf(interv, loc=series_mean[i], scale=max(series_std[i], xlim/n_bins)) for i in range(len(series_mean))]
	cmap = plt.get_cmap("Accent")
	for i in range(len(series_mean)):
		graph.plot(interv, dist[i], color=cmap(i), label=str(out_classes[i]) + " (mean: {:.2f}, std: {:.2f})".format(series_mean[i], series_std[i]))
	if len(series_mean) == 2: # Also plot overlap
		overlap_dist = np.amin(dist, axis=0)
		overlap = overlap_dist.sum() * xlim/n_bins
		graph.plot(interv, overlap_dist, "--", dashes=(2, 4), color=cmap(2), label="Overlap ({:.2f}%)".format(100*overlap))
		graph.fill_between(interv, np.zeros_like(interv), overlap_dist, color=cmap(2), alpha=.1)
	graph.set_ylim(0., 2.)
	graph.grid(True)
	graph.legend()
	os.makedirs(os.path.dirname(path), exist_ok=True)
	graph.get_figure().savefig(path, bbox_inches='tight')
	graph.get_figure().clear()
	plt.close(graph.get_figure())

def plot_st_resp(data, series_names, step, path):
	plt.ioff()
	nrows = data[0].size(1)
	ncols = data[0].size(2)
	fig = plt.figure()
	for k in range(nrows):
		for l in range(ncols):
			ax1 = fig.add_subplot(nrows, ncols, k*ncols + l + 1)
			for i in range(len(data)):
				series = data[i][:, k, l].cpu().detach().numpy()
				ax1.plot(range(0, len(series)*step, step), series, label=series_names[i])
			if k == nrows - 1 and l == 0:
				ax1.set_xticks([0, 100])
				ax1.set_xticklabels([0, 100])
				ax1.set_xlabel("Time")
				ax1.set_yticks([0., 1.])
				ax1.set_yticklabels([0., 1.])
				ax1.set_ylabel("Spikes")
				#ax1.legend()
			else:
				ax1.set_xticklabels([])
				ax1.set_yticklabels([])
			ax1.set_ylim(0., 2.)
	plt.subplots_adjust(wspace=0.1, hspace=0.1)
	os.makedirs(os.path.dirname(path), exist_ok=True)
	fig.savefig(path, bbox_inches='tight')
	plt.close(fig)

def plot_series(series_mean, series_std, series_names, step, path, ci_lvl=.95):
	plt.ioff()
	graph = plt.axes(xlabel='Time', ylabel='Spikes')
	for i in range(len(series_mean)):
		mean = series_mean[i].cpu().detach().numpy()
		std = series_std[i].cpu().detach().numpy()
		name = series_names[i]
		series_err = st.norm.interval(ci_lvl)[1] * std
		graph.errorbar(range(0, len(mean)*step, step), mean, yerr=series_err, capsize=3, label=name)
		#graph.fill_between(range(0, len(mean)*step, step), mean-series_err, mean+series_err, alpha=.1)
	graph.set_ylim(0., 2.)
	graph.grid(True)
	graph.legend()
	os.makedirs(os.path.dirname(path), exist_ok=True)
	graph.get_figure().savefig(path, bbox_inches='tight')
	graph.get_figure().clear()
	plt.close(graph.get_figure())

