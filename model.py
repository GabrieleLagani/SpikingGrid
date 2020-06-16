import torch
from bindsnet.network import Network
from bindsnet.network.nodes import Input, IFNodes, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import PostPre
from bindsnet.network.monitors import Monitor

import utils

class Net(Network):
    def __init__(self, inpt_shape=(1, 28, 28), neuron_shape=(10, 10),
                 vrest=0.5, vreset=0.5, vth=1., lbound=0.,
                 theta_w=1e-3, sigma=1., conn_strength=1.,
                 sigma_lateral_exc=1., exc_strength=1.,
                 sigma_lateral_inh=1., inh_strength=1.,
                 refrac=5, tc_decay=50., tc_trace=20., dt=1.0,
                 nu=(1e-4, 1e-2), reduction=None):
        super().__init__(dt=dt)

        self.inpt_shape = inpt_shape
        self.n_inpt = utils.shape2size(self.inpt_shape)
        self.neuron_shape = neuron_shape
        self.n_neurons = utils.shape2size(self.neuron_shape)
        self.dt = dt

        # Layers
        input = Input(n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=tc_trace)
        population = LIFNodes(shape=self.neuron_shape, traces=True, lbound=lbound, rest=vrest, reset=vreset, thresh=vth, refrac=refrac, tc_decay=tc_decay, tc_trace=tc_trace)
        inh = IFNodes(shape=self.neuron_shape, traces=True, lbound=0., rest=0., reset=0., thresh=0.99, refrac=0, tc_trace=tc_trace)

        # Coordinates
        self.coord_x = torch.rand(neuron_shape) * self.neuron_shape[1] / self.neuron_shape[0]
        self.coord_y = torch.rand(neuron_shape)
        self.coord_x_disc = (self.coord_x * self.inpt_shape[2]/(self.neuron_shape[1]/self.neuron_shape[0])).long()
        self.coord_y_disc = (self.coord_y * self.inpt_shape[1]).long()
        grid_x = (torch.arange(self.inpt_shape[2]).unsqueeze(0).float() + 0.5) * (self.neuron_shape[1] / self.neuron_shape[0]) / self.inpt_shape[2]
        grid_y = (torch.arange(self.inpt_shape[1]).unsqueeze(1).float() + 0.5) / self.inpt_shape[1]

        # Input-Neurons connections
        w = torch.abs(torch.randn(self.inpt_shape[1], self.inpt_shape[2], *self.neuron_shape))
        for k in range(neuron_shape[0]):
            for l in range(neuron_shape[1]):
                sq_dist = (grid_x - self.coord_x[k, l]) ** 2 + (grid_y - self.coord_y[k, l]) ** 2
                w[:, :, k, l] *= torch.exp(- sq_dist / (2 * sigma ** 2))
        w = w.view(self.n_inpt, self.n_neurons)
        input_mask = w < theta_w
        w[input_mask] = 0.  # Drop connections smaller than threshold
        input_conn = Connection(source=input, target=population, w=w, update_rule=PostPre, nu=nu, reduction=reduction, wmin=0, norm=conn_strength)
        input_conn.normalize()

        # Excitatory self-connections
        w =  torch.abs(torch.randn(*self.neuron_shape, *self.neuron_shape))
        for k in range(neuron_shape[0]):
            for l in range(neuron_shape[1]):
                sq_dist = (self.coord_x - self.coord_x[k, l]) ** 2 + (self.coord_y - self.coord_y[k, l]) ** 2
                w[:, :, k, l] *= torch.exp(- sq_dist / (2 * sigma_lateral_exc ** 2))
                w[k, l, k, l] = 0. # set connection from neuron to itself to zero
        w = w.view(self.n_neurons, self.n_neurons)
        exc_mask = w < theta_w
        w[exc_mask] = 0.  # Drop connections smaller than threshold
        self_conn_exc = Connection(source=population, target=population, w=w, update_rule=PostPre, nu=nu, reduction=reduction, wmin=0, norm=exc_strength)
        self_conn_exc.normalize()

        # Inhibitory self-connection
        w = torch.eye(self.n_neurons)
        exc_inh = Connection(source=population, target=inh, w=w)
        w = -torch.abs(torch.randn(*self.neuron_shape, *self.neuron_shape))
        for k in range(neuron_shape[0]):
            for l in range(neuron_shape[1]):
                sq_dist = (self.coord_x - self.coord_x[k, l]) ** 2 + (self.coord_y - self.coord_y[k, l]) ** 2
                w[:, :, k, l] *= torch.exp(- sq_dist / (2 * sigma_lateral_inh ** 2))
                w[k, l, k, l] = 0.  # set connection from neuron to itself to zero
        w = w.view(self.n_neurons, self.n_neurons)
        inh_mask = w > -theta_w
        w[inh_mask] = 0.  # Drop connections smaller than threshold
        self_conn_inh = Connection(source=inh, target=population, w=w, update_rule=PostPre, nu=tuple(-a for a in nu), reduction=reduction, wmax=0, norm=inh_strength)
        self_conn_inh.normalize()

        # Add layers to network
        self.add_layer(input, name="X")
        self.add_layer(population, name="Y")
        self.add_layer(inh, name="Z")

        # Add connections
        self.add_connection(input_conn, source="X", target="Y")
        self.add_connection(self_conn_exc, source="Y", target="Y")
        self.add_connection(exc_inh, source="Y", target="Z")
        self.add_connection(self_conn_inh, source="Z", target="Y")

        # Add weight masks to network
        self.masks = {}
        self.add_weight_mask(mask=input_mask, connection_id=("X", "Y"))
        self.add_weight_mask(mask=exc_mask, connection_id=("Y", "Y"))
        self.add_weight_mask(mask=inh_mask, connection_id=("Z", "Y"))

        # Add monitors to record neuron spikes
        self.spike_monitor = Monitor(self.layers["Y"], ["s"])
        self.add_monitor(self.spike_monitor, name="Spikes")

    def add_weight_mask(self, mask, connection_id):
        self.masks[connection_id] = mask

    def run(self, inputs, time, one_step=False, **kwargs):
        super().run(inputs=inputs, time=time, one_step=one_step, masks=self.masks, **kwargs)

    def to_gpu(self):
        self.masks = {k: v.cuda() for k, v in self.masks.items()}
        return self.to("cuda")

