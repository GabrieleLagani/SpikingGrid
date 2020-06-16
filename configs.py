import torch
import params as P

class Configuration:
	def __init__(self, config_name, inpt_shape=(1, 6, 6), grid_shape=(1, 6, 10), classes=None,
	             n_train=50000, n_eval=10000, batch_size=1, eval_batch_size=64, eval_interval=50,
	             n_epochs=1, inpt_norm=None, intensity=200., label_intensity=200., time=100, dt=1.,
	             refr=5, v_rest=.5, v_reset=.5, v_lb=0., v_th=1., v_decay=50., tr_decay=20., lr=(1e-4, 1e-2),
	             neuron_shape=(12, 20), sigma=1., conn_str=1., sigma_exc=1., exc_str=1., sigma_inh=1., inh_str=1., theta_w=1e-3):
		self.CONFIG_NAME = config_name
		self.RESULT_FOLDER = P.RESULT_FOLDER + "/" + self.CONFIG_NAME
		
		# Data-related params
		self.INPT_SHAPE = inpt_shape
		self.GRID_SHAPE = grid_shape
		self.LABEL_SHAPE = (self.GRID_SHAPE[0], self.GRID_SHAPE[1], self.GRID_SHAPE[2] - self.INPT_SHAPE[2])
		self.CLASSES = classes
		self.N_TRAIN = n_train
		self.N_EVAL = n_eval
		self.BATCH_SIZE = batch_size
		self.EVAL_BATCH_SIZE = eval_batch_size
		self.EVAL_INTERVAL = eval_interval * self.BATCH_SIZE
		self.N_EPOCHS = n_epochs
		self.INPT_NORM = inpt_norm
		self.INTENSITY = intensity
		self.LABEL_INTENSITY = label_intensity
		
		# Simulation-related params
		self.TIME = time
		self.DT = dt
		
		# Neuron-related params
		self.REFR = refr
		self.V_REST = v_rest
		self.V_RESET = v_reset
		self.V_LB = v_lb
		self.V_TH = v_th
		self.V_DECAY = v_decay
		self.TR_DECAY = tr_decay
		self.LR = lr
		
		# Topology-related params
		self.NEURON_SHAPE = neuron_shape
		self.SIGMA = sigma
		self.CONN_STR = conn_str
		self.SIGMA_EXC = sigma_exc
		self.EXC_STR = exc_str
		self.SIGMA_INH = sigma_inh
		self.INH_STR = inh_str
		self.THETA_W = theta_w
		
		self.ASSIGNMENTS = self.get_assignments(self.GRID_SHAPE, self.CLASSES)
	
	def get_assignments(self, shape, classes):
		if classes is None: classes = range(P.N_CLASSES)
		assignments = -torch.ones(shape)
		n_classes = len(classes)
		if n_classes < 2:
			pass
		elif n_classes == 2:
			assignments[:, :shape[1] // 2, -1] = classes[0]
			assignments[:, shape[1] // 2 + (0 if (shape[1] / 2 - shape[1] // 2 == 0) else 1):, -1] = classes[1]
		elif n_classes == 3 and shape[1] % 3 != 2:
			assignments[:, :shape[1] // 3, -1] = classes[0]
			assignments[:, shape[1] // 3: 2 * (shape[1] // 3), -1] = classes[1]
			assignments[:, 2 * (shape[1] // 3): 3 * (shape[1] // 3), -1] = classes[2]
		elif n_classes == 3 and shape[1] % 3 == 2:
			assignments[:, :shape[1] // 3, -1] = classes[0]
			assignments[:, (shape[1] // 3) + 1: 2 * (shape[1] // 3) + 1, -1] = classes[1]
			assignments[:, 2 * (shape[1] // 3) + 2: 3 * (shape[1] // 3) + 2, -1] = classes[2]
		elif n_classes == 4 and shape[1] == 6:
			assignments[:, 0:2, -2] = classes[0]
			assignments[:, 0, -1] = classes[0]
			assignments[:, 2, -2] = classes[1]
			assignments[:, 1:3, -1] = classes[1]
			assignments[:, 3:5, -2] = classes[2]
			assignments[:, 3, -1] = classes[2]
			assignments[:, 5, -2] = classes[3]
			assignments[:, 4:, -1] = classes[3]
		else: raise NotImplementedError("Error in config. " + self.CONFIG_NAME + ": assignments undefined for grid shape " + str(self.GRID_SHAPE))
		return assignments

# Task params
# V_REST = 0., V_LB = None
#                                           NEURON_SHAPE = (12, 20)       NEURON_SHAPE = (78, 130)
# EXC_STRENGTH = 3., INH_STRENGTH = 300.            95%                             85%
# EXC_STRENGTH = 8., INH_STRENGTH = 800.            50%                             94%

# Bio params
# V_REST = 0.5, V_LB = 0.
#                                           NEURON_SHAPE = (12, 20)       NEURON_SHAPE = (78, 130)
# EXC_STRENGTH = 1., INH_STRENGTH = 10.            90%                             70%

CONFIG_LIST = [
	
	Configuration(
		config_name="mnist01_240n_bio", classes=[0, 1], inpt_norm=None, intensity=200., label_intensity=200., n_train=10000,
		refr=5, v_rest=0.5, v_reset=0.5, v_lb=0., v_th=1., v_decay=50., tr_decay=20., lr=(1e-4, 1e-2),
		neuron_shape=(12, 20), sigma=0.1, conn_str=1., sigma_exc=0.4, exc_str=1., sigma_inh=0.05, inh_str=10.,
		theta_w=1e-3,
	),
	
	Configuration(
		config_name="mnist01_240n_tsk", classes=[0, 1], inpt_norm=None, intensity=200., label_intensity=200., n_train=10000,
		refr=5, v_rest=0., v_reset=0.5, v_lb=None, v_th=1., v_decay=50., tr_decay=20., lr=(1e-4, 1e-2),
		neuron_shape=(12, 20), sigma=0.1, conn_str=1., sigma_exc=0.4, exc_str=3., sigma_inh=0.05, inh_str=300.,
		theta_w=1e-3,
	),
	
	Configuration(
		config_name="mnist01_10000n_bio", classes=[0, 1], inpt_norm=None, intensity=200., label_intensity=200., n_train=10000,
		refr=5, v_rest=0.5, v_reset=0.5, v_lb=0., v_th=1., v_decay=50., tr_decay=20., lr=(1e-4, 1e-2),
		neuron_shape=(78, 130), sigma=0.1, conn_str=1., sigma_exc=0.4, exc_str=1., sigma_inh=0.05, inh_str=10.,
		theta_w=1e-3,
	),
	
	Configuration(
		config_name="mnist01_10000n_tsk", classes=[0, 1], inpt_norm=None, intensity=200., label_intensity=200., n_train=10000,
		refr=5, v_rest=0., v_reset=0.5, v_lb=None, v_th=1., v_decay=50., tr_decay=20., lr=(1e-4, 1e-2),
		neuron_shape=(78, 130), sigma=0.1, conn_str=1., sigma_exc=0.4, exc_str=8., sigma_inh=0.05, inh_str=800.,
		theta_w=1e-3,
	),
	
	Configuration(
		config_name="mnist012_10000n_bio", classes=[0, 1, 2], inpt_norm=None, intensity=200., label_intensity=200., n_train=10000,
		refr=5, v_rest=0.5, v_reset=0.5, v_lb=0., v_th=1., v_decay=50., tr_decay=20., lr=(1e-4, 1e-2),
		neuron_shape=(78, 130), sigma=0.1, conn_str=1., sigma_exc=0.4, exc_str=1., sigma_inh=0.05, inh_str=10.,
		theta_w=1e-3,
	),
	
	Configuration(
		config_name="mnist0123_10000n_bio", classes=[0, 1, 2, 3], inpt_norm=None, intensity=200., label_intensity=200., n_train=10000,
		refr=5, v_rest=0.5, v_reset=0.5, v_lb=0., v_th=1., v_decay=50., tr_decay=20., lr=(1e-4, 1e-2),
		neuron_shape=(78, 130), sigma=0.1, conn_str=1., sigma_exc=0.4, exc_str=1., sigma_inh=0.05, inh_str=10.,
		theta_w=1e-3,
	),
]


CONFIGURATIONS = {c.CONFIG_NAME: c for c in CONFIG_LIST}
