import os
import torch
import torch.nn.functional as F
from bindsnet.encoding import PoissonEncoder
import utils
from utils import CustomEncoder
from model import Net


# Simulation parameters
RESULTS_PATH = "results/bioexp"
GPU = torch.cuda.is_available()
DEVICE = "cuda:0" if GPU else "cpu"
NUM_ITERS = 4
NUM_TRN_ITER = 40
TIME = 400
DT = 1.0

# Encoding-related parameters
GRID_SHAPE = (1, 6, 10)
ENC_POISSON = 'poisson'
ENC_CUSTOM = 'custom'
TST_ENCODER = ENC_CUSTOM
INTENSITY = 10
TETAN_INTENSITY = 250
NOISE_INTENSITY = 0
DELAY = 50
DURATION = 10
SPARSITY = 0.25
TAU = 20.

# Neuron-related params. NB: in simulation, 0. membrane potential is equivalent to -90mV,
# 1. membrane potential is equivalent to -50mV, 0.5 membrane potential is equivalent to -70mV
REFR = 5
V_REST = 0.5
V_TH = 1.
V_RESET = 0.5
V_LB = 0.
V_DECAY = 50.
TR_DECAY = 20.
LEARNING_RATE = (1e-4, 1e-2)

# Topology-related parameters
NEURON_SHAPE = (78, 130) #(12, 20) #
SIGMA = 0.1
CONN_STRENGTH = 1.
SIGMA_LATERAL_EXC = 0.4
EXC_STRENGTH = 1.
SIGMA_LATERAL_INH = 0.05
INH_STRENGTH = 10.
THETA_W = 1e-3

# Output-related params
BIN_SIZE = 10
NUM_BINS = TIME // BIN_SIZE + (1 if TIME % BIN_SIZE != 0 else 0)
OUT_BINS = 15
DELAY_BINS = DELAY // BIN_SIZE
CI_LVL = 0.95

def bio_test(net, patterns, encoder):
    results = torch.empty(len(patterns), NUM_BINS, GRID_SHAPE[1], GRID_SHAPE[2], device=DEVICE)
    for i in range(len(patterns)):
        # Prepare input
        input_enc = encoder(patterns[i] * INTENSITY)
        if GPU: input_enc = input_enc.cuda()

        # Run the network on the input
        net.train(False)
        net.run(inputs={'X': input_enc}, time=TIME)

        # Read out resulting output spike pattern
        # The monitor returns the collection of spikes. Dimension 0 is the time dimension, dimension 1 is the batch dimension,
        # dimension 2 is the height dimension and dimension 3 is the width dimension. The batch dimension can be
        # squeezed out in this case.
        s = net.spike_monitor.get("s").squeeze(1).float()
        # Reshape the tensor so that dim 0 is the height*width dimensions, which is considered as a new batch dimension,
        # dimension 1 is a new channel dimension and dimension 2 is the time dimension, so that the tensor is ready for
        # conv1d operations over time.
        s = s.view(s.size(0), -1).t().unsqueeze(1)
        # Estimate instantaneous firing rate at each instant as leaky integration of spikes
        exp_knl = torch.cat([torch.exp(-torch.arange(s.size(-1), device=DEVICE)/TAU).flip(0), torch.zeros(s.size(-1)-1, device=DEVICE)], dim=0).view(1, 1, -1)
        s = torch.conv1d(s, exp_knl, padding=s.size(-1)-1)
        # Aggregate firing rate along time dimension in NUM_BINS bins of BIN_SIZE each
        s = F.avg_pool1d(s, BIN_SIZE, ceil_mode=True).squeeze(1).t().view(-1, NEURON_SHAPE[0], NEURON_SHAPE[1])
        # Aggregate firing rate on each electrode at every time bin, i.e. aggregate firing rate in a tensor of
        # GRID_SHAPE height and width
        r = torch.zeros(NUM_BINS, GRID_SHAPE[1], GRID_SHAPE[2], device=DEVICE)
        c = torch.zeros(NUM_BINS, GRID_SHAPE[1], GRID_SHAPE[2], device=DEVICE)
        for k in range(s.size(1)):
            for l in range(s.size(2)):
                r[:, net.coord_y_disc[k, l], net.coord_x_disc[k, l]] += s[:, k, l]
                c[:, net.coord_y_disc[k, l], net.coord_x_disc[k, l]] += 1
        c[c == 0.] = 1. # Prevent division by zero
        results[i, :, :, :] = r / c

        # Reset network state
        net.reset_state_variables()

    return results

def bio_trn(net, pattern, encoder):
    for i in range(NUM_TRN_ITER):
        print("\rTrain iter: " + str(i+1) + "/" + str(NUM_TRN_ITER), end=('\n' if i+1 == NUM_TRN_ITER else ''))
        # Prepare input
        input_enc = encoder(pattern * TETAN_INTENSITY)
        if GPU: input_enc = input_enc.cuda()

        # Run the network on the input
        net.train(True)
        net.run(inputs={'X': input_enc}, time=TIME)
        net.train(False)

        # Reset network state
        net.reset_state_variables()

def run_bio_exp():
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Make inputs
    regL = torch.zeros(1, *GRID_SHAPE)
    regL[:, :, :, 0] = 1.
    regL[:, :, -1, :] = 1.
    upsL = torch.zeros(1, *GRID_SHAPE)
    upsL[:, :, 0, :] = 1.
    upsL[:, :, :, -1] = 1.
    patterns = [regL, upsL]
    ptn_names = ["regL", "upsL"]
    
    try: # Try to load simulation results from file
        results = torch.load(RESULTS_PATH + "/results.pt")
        print("Found existing result file, results loaded")
    except FileNotFoundError: # If file is not available, compute the results
        results = {"bef_tet": torch.empty(NUM_ITERS, len(patterns), NUM_BINS, GRID_SHAPE[1], GRID_SHAPE[2], device=DEVICE),
                   "aft_tet": torch.empty(NUM_ITERS, len(patterns), NUM_BINS, GRID_SHAPE[1], GRID_SHAPE[2], device=DEVICE)}
        for seed in range(NUM_ITERS):
            print("####    CURRENT ITERATION: " + str(seed) + "    ####")
            torch.manual_seed(seed)
            trn_encoder = PoissonEncoder(time=TIME, dt=DT)
            tst_encoder = PoissonEncoder(time=TIME, dt=DT) if TST_ENCODER == ENC_POISSON else CustomEncoder(time=TIME, delay=DELAY, duration=DURATION, dt=DT, sparsity=SPARSITY, noise_intensity=NOISE_INTENSITY)
    
            # Prepare network
            print("Preparing networks")
            net: Net = Net(inpt_shape=GRID_SHAPE, neuron_shape=NEURON_SHAPE,
                   lbound=V_LB, vrest=V_REST, vreset=V_RESET, vth=V_TH,
                   theta_w=THETA_W, sigma=SIGMA, conn_strength=CONN_STRENGTH,
                   sigma_lateral_exc=SIGMA_LATERAL_EXC, exc_strength=EXC_STRENGTH,
                   sigma_lateral_inh=SIGMA_LATERAL_INH, inh_strength=INH_STRENGTH,
                   dt=DT, refrac=REFR, tc_decay=V_DECAY, tc_trace=TR_DECAY,
                   nu=LEARNING_RATE)
            # Direct networks to GPU
            if GPU: net = net.to_gpu()
    
            # Test the network before tetanization
            print("Testing network before tetanization")
            results["bef_tet"][seed, :, :, :, :] = bio_test(net, patterns, tst_encoder)
    
            # Tetanize network on lower-L input
            print("Tetanizing network")
            bio_trn(net, regL, trn_encoder)
    
            # Test the network after tetanization
            print("Testing network after tetanization")
            results["aft_tet"][seed, :, :, :, :] = bio_test(net, patterns, tst_encoder)
    
            print("Done\n")
    
        print("Saving results")
        os.makedirs(RESULTS_PATH, exist_ok=True)
        torch.save(results, RESULTS_PATH + "/results.pt")
    
    print("Saving plots")
    bio_mean_series = {
        "bef_tet": torch.tensor([
            [0., 0., 0., 0., 0., 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05], # pattern regL
            [0., 0., 0., 0., 0., 0.4, 0.6, 0.5, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05]  # pattern upsL
        ]),
        "aft_tet": torch.tensor([
            [0., 0., 0., 0., 0., 0.7, 1.1, 0.8, 0.6, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2], # pattern regL
            [0., 0., 0., 0., 0., 0.4, 0.5, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # pattern upsL
        ]),
    }
    bio_std_series = {
        "bef_tet": torch.tensor([
            [0., 0., 0., 0., 0., 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], # pattern regL
            [0., 0., 0., 0., 0., 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]  # pattern upsL
        ]),
        "aft_tet": torch.tensor([
            [0., 0., 0., 0., 0., 0.15, 0.3, 0.2, 0.05, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1], # pattern regL
            [0., 0., 0., 0., 0., 0.1, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]  # pattern upsL
        ]),
    }
    decay_time = {"bef_tet": 20., "aft_tet": 50.}
    bio_mean = {"bef_tet": torch.tensor([5.6, 4.7]), "aft_tet": torch.tensor([6.4, 4.1])}
    bio_std = {"bef_tet": torch.tensor([0.8, 0.8]), "aft_tet": torch.tensor([0.7, 0.5])}
    bio_count = 4
    bio_st_resp = {
        "bef_tet": torch.tensor([
            [
                [0.5, 0.2, 0.2, 0.2],
                [0.5, 0., 0.5, 0.],
                [0.2, 1., 0.2, 0.2],
                [1., 0., 0., 0.],
                [0., 1., 0.5, 0.],
                [1., 0., 1., 0.2],
            ],  # pattern regL
            
            [
                [0.2, 1., 1., 1.],
                [0.5, 1., 0.2, 1.5],
                [0.5, 0.2, 0.2, 0.5],
                [0.5, 0.2, 0.5, 0.],
                [0.2, 0.2, 0., 0.],
                [0.2, 0., 0., 0.],
            ]  # pattern upsL
        ]),
        "aft_tet": torch.tensor([
            [
                [1.8, 1., 0., 0.5],
                [1.8, 1., 1.8, 0.],
                [1., 1.8, 0., 0.5],
                [1.8, 1., 0.1, 0.2],
                [1., 1.8, 1., 1.],
                [1.8, 0., 1.8, 0.5],
            ],  # pattern regL
            [
                [0., 1., 1., 0.5],
                [0.5, 0.5, 0., 1.],
                [0.5, 0.5, 0.2, 0.2],
                [0.2, 0.2, 0.5, 0.],
                [0., 0.2, 0., 0.],
                [0.2, 0., 0., 0.],
            ]  # pattern upsL
        ]),
    }
    for k in results.keys():
        out = None
        for i in range(len(patterns)):
            res = results[k][:, i, 0:OUT_BINS, :, :]
            bio = bio_st_resp[k][i].unsqueeze(0) * torch.cat(
                (torch.zeros(DELAY_BINS), torch.exp(- torch.arange(OUT_BINS - DELAY_BINS)/(TAU/BIN_SIZE))), dim=0).view(-1, 1, 1)
            utils.plot_st_resp([res[:, :, :, [0, 3, 6, 9]].mean(dim=0), bio], ["Simul.", "Biol."],
                               BIN_SIZE, RESULTS_PATH + "/" + k + "/st_resp_" + ptn_names[i] + ".png")
            res = res.mean(dim=3).mean(dim=2)
            utils.plot_series([res.mean(dim=0), bio_mean_series[k][i]], [res.std(dim=0), bio_std_series[k][i]],
                              ["Simul.", "Biol."], BIN_SIZE, RESULTS_PATH + "/" + k + "/time_resp_" + ptn_names[i] + ".png", CI_LVL)
            bin_count = int(decay_time[k]) // BIN_SIZE
            res = res[:, DELAY_BINS:DELAY_BINS+bin_count].mean(dim=1, keepdim=True) * BIN_SIZE
            out = res if out is None else torch.cat((out, res), dim=1)
        mean = out.mean(dim=0)
        std = out.std(dim=0)
        count = out.size(0)
        utils.plot_out_resp([mean, bio_mean[k]], [std, bio_std[k]], [count, bio_count], ["Simul.", "Biol."], ptn_names,
                            RESULTS_PATH + "/" + k + "/out_mean.png", CI_LVL)
        utils.plot_out_dist(mean, std, ptn_names, RESULTS_PATH + "/" + k + "/out_dist_simul.png")
        utils.plot_out_dist(bio_mean[k], bio_std[k], ptn_names, RESULTS_PATH + "/" + k + "/out_dist_biol.png")
    
    print("Finished")

if __name__=='__main__':
    # Launch experiment
    run_bio_exp()

