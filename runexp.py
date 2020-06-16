import argparse
from time import time as t
import torch
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import load

import params as P
import configs as C
import utils
from data import DataManager
from model import Net


class Experiment:
    def __init__(self, config, mode, seed):
        self.config: C.Configuration = config
        self.mode = mode
        self.seed = seed
        
        # For reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(self.seed)

        # Load MNIST dataset
        print("Preparing datasets")
        self.data_manager = DataManager(n_train=self.config.N_TRAIN, n_eval=self.config.N_EVAL, inpt_shape=self.config.INPT_SHAPE,
                                        grid_shape=self.config.GRID_SHAPE, label_shape=self.config.LABEL_SHAPE, assignments=self.config.ASSIGNMENTS,
                                        inpt_norm=self.config.INPT_NORM, intensity=self.config.INTENSITY, label_intensity=self.config.LABEL_INTENSITY)
        self.trn_set = self.data_manager.get_train(self.config.CLASSES, PoissonEncoder(time=self.config.TIME, dt=self.config.DT), self.config.BATCH_SIZE)
        self.trn_set4eval = self.data_manager.get_train4eval(self.config.CLASSES, PoissonEncoder(time=self.config.TIME, dt=self.config.DT), self.config.EVAL_BATCH_SIZE)
        self.val_set = self.data_manager.get_val(self.config.CLASSES, PoissonEncoder(time=self.config.TIME, dt=self.config.DT), self.config.EVAL_BATCH_SIZE)
        self.tst_set = self.data_manager.get_test(self.config.CLASSES, PoissonEncoder(time=self.config.TIME, dt=self.config.DT), self.config.EVAL_BATCH_SIZE)

        # Build network
        print("Preparing network")
        self.network: Net = Net(inpt_shape=self.config.GRID_SHAPE, neuron_shape=self.config.NEURON_SHAPE,
			   lbound=self.config.V_LB, vrest=self.config.V_REST, vreset=self.config.V_RESET, vth=self.config.V_TH,
			   theta_w=self.config.THETA_W, sigma=self.config.SIGMA, conn_strength=self.config.CONN_STR,
			   sigma_lateral_exc=self.config.SIGMA_EXC, exc_strength=self.config.EXC_STR,
			   sigma_lateral_inh=self.config.SIGMA_INH, inh_strength=self.config.INH_STR,
			   dt=self.config.DT, refrac=self.config.REFR, tc_decay=self.config.V_DECAY, tc_trace=self.config.TR_DECAY,
			   nu=self.config.LR)
        # Direct network to GPU
        if P.GPU: self.network.to_gpu()

        # Object for network monitoring
        print("Preparing stats manager")
        self.stats_manager = utils.StatsManager(self.network, self.config.CLASSES, self.config.ASSIGNMENTS)

    def train_pass(self):
        start_time = t()
        for step, batch in enumerate(self.trn_set):
            elapsed_time = t() - start_time
            time_per_step = elapsed_time/(step+1)
            exp_rem_time = (len(self.trn_set) - (step+1)) * time_per_step
            print("\r" + str(step+1) + "/" + str(len(self.trn_set)) + " processed batches (elapsed time: " + utils.format_time(elapsed_time) + " exp. rem. time: " + utils.format_time(exp_rem_time) + ")", end="")
            
            # Get next input
            input_enc = batch["encoded_image"]
            if P.GPU: input_enc = input_enc.cuda()

            # Run the network on the input
            self.network.train(True)
            self.network.run(inputs={"X": input_enc}, time=self.config.TIME)
            self.network.train(False)
            # Reset network state
            self.network.reset_state_variables()
            
            # Evaluate performance at fixed intervals
            if (((step + 1) * self.config.BATCH_SIZE) % self.config.EVAL_INTERVAL == 0) or (step == len(self.trn_set) - 1):
                print("\nEvaluating...")
                print("Computing train accuracy...")
                self.eval_pass(self.trn_set4eval, train=True)
                print("Computing validation accuracy...")
                self.eval_pass(self.val_set, train=False)
                # Print results
                print("Current evaluation step: " + str(len(self.stats_manager.eval_accuracy)))
                print("Current trn. accuracy: " + str(100 * self.stats_manager.trn_accuracy[-1]) + "%")
                print("Current val. accuracy: " + str(100 * self.stats_manager.eval_accuracy[-1]) + "%")
                print("Top accuracy so far: " + str(100 * self.stats_manager.best_acc) + "%" + " at evaluation step: " + str(self.stats_manager.best_step))
                # Plot results
                utils.plot_performance(self.stats_manager.trn_accuracy, self.stats_manager.eval_accuracy, self.config.RESULT_FOLDER + "/accuracy.png")
                # Check if accuracy improved
                if self.stats_manager.check_improvement(): # Save model
                    print("Top accuracy improved! Saving new best model...")
                    self.network.save(self.config.RESULT_FOLDER + "/model.pt")
                    print("Model saved!")
                print("Evaluation complete!")
                print("Continuing training...")

    def eval_pass(self, dataset, train):
        self.network.train(False)
        for step, batch in enumerate(dataset):
            print("\r" + str(step + 1) + "/" + str(len(dataset)) + " processed batches",end="\n" if step + 1 == len(dataset) else "")
            # Get next input sample.
            input_enc = batch["encoded_image"]
            if P.GPU: input_enc = input_enc.cuda()
            # Run the network on the input without labels
            self.network.run(inputs={"X": input_enc}, time=self.config.TIME)
            # Update network activity monitoring (update hits and count)
            self.stats_manager.update(batch)
            # Reset network state
            self.network.reset_state_variables()
        self.stats_manager.record_accuracy(train)

    def run_train(self):
        # Train the network.
        print("####    Begin training...    ####")
        start = t()
        for epoch in range(1, self.config.N_EPOCHS + 1):
            # Print overall progress information at each epoch
            utils.print_train_progress(epoch, self.config.N_EPOCHS, t() - start)

            # Run a train pass on the dataset
            self.train_pass()

        print("Training complete!\n")

    def run_test(self):
        SAMPLES_PER_CLASS = 50
        N_CLASSES = 10
        TIME = 150
        BIN_SIZE = 10
        DELAY = 50
        DURATION = 10
        SPARSITY = 0.05
        CI_LVL = 0.95
        
        # Determine the output and spatio-temporal response to various patterns, including unknown classes
        for model in ["scratch", "trained"]:
            if model == "trained": # Initially compute test statistics with model initialized from scratch, then do the same with trained model
                try:
                    self.network: Net = load(self.config.RESULT_FOLDER + "/model.pt")
                except FileNotFoundError as e:
                    print("No saved network model found.")
                    raise e
                # Direct network to GPU
                if P.GPU: self.network.to_gpu()
                self.stats_manager = utils.StatsManager(self.network, self.config.CLASSES, self.config.ASSIGNMENTS)
            self.network.train(False)
            print("Testing " + model + " model...")
            
            for type in ["out", "st"]:
                if type == "out": print("Computing output responses for various patterns")
                else: print("Computing spatio-temporal responses for various patterns")
                unk = None
                for k in range(N_CLASSES + 1):
                    pattern_name = str(k) if k < N_CLASSES else "rnd"
                    print("Pattern: " + pattern_name)
                    encoder = PoissonEncoder(time=self.config.TIME, dt=self.config.DT) if type == "out" else utils.CustomEncoder(TIME, DELAY, DURATION, self.config.DT, SPARSITY)
                    dataset = self.data_manager.get_test([k], encoder, SAMPLES_PER_CLASS) if k < N_CLASSES else None
                    # Get next input sample.
                    input_enc = next(iter(dataset))["encoded_image"] if k < N_CLASSES else encoder(torch.cat((
                        torch.rand(SAMPLES_PER_CLASS, *self.config.INPT_SHAPE) * (self.config.INPT_NORM / (.25 * self.config.INPT_SHAPE[1] * self.config.INPT_SHAPE[2]) if self.config.INPT_NORM is not None else 1.),
                        torch.zeros(SAMPLES_PER_CLASS, *self.config.LABEL_SHAPE)), dim=3) * self.config.INTENSITY)
                    if P.GPU: input_enc = input_enc.cuda()
                    # Run the network on the input without labels
                    self.network.run(inputs={"X": input_enc}, time=self.config.TIME if type=="out" else TIME)
                    # Update network activity monitoring
                    res = self.stats_manager.get_class_scores() if type == "out" else self.stats_manager.get_st_resp(bin_size=BIN_SIZE)
                    if k not in self.config.CLASSES and k < N_CLASSES: unk = res if unk is None else torch.cat((unk, res), dim=0)
                    # Reset network state
                    self.network.reset_state_variables()
                    # Save results
                    if type == "out":
                        mean = res.mean(dim=0)
                        std = res.std(dim=0)
                        count = res.size(0)
                        utils.plot_out_resp([mean], [std], [count], [pattern_name + " out"], self.config.CLASSES,
                                            self.config.RESULT_FOLDER + "/" + model + "/out_mean_" + pattern_name + ".png", CI_LVL)
                        utils.plot_out_dist(mean, std, self.config.CLASSES, self.config.RESULT_FOLDER + "/" + model + "/out_dist_" + pattern_name + ".png")
                    else:
                        utils.plot_st_resp([res.mean(dim=0)[:, :, [0, 3, 6, 9]]], [pattern_name + " resp."],
                                           BIN_SIZE, self.config.RESULT_FOLDER + "/" + model + "/st_resp_" + pattern_name + ".png")
                        res = res.mean(dim=3).mean(dim=2)
                        utils.plot_series([res.mean(dim=0)], [res.std(dim=0)], [pattern_name + " resp."],
                                          BIN_SIZE, self.config.RESULT_FOLDER + "/" + model + "/time_resp_" + pattern_name + ".png", CI_LVL)
                print("Pattern: unk")
                if type == "out":
                    mean = unk.mean(dim=0)
                    std = unk.std(dim=0)
                    count = unk.size(0)
                    utils.plot_out_resp([mean], [std], [count], ["unk out"],
                                        self.config.CLASSES, self.config.RESULT_FOLDER + "/" + model + "/out_mean_unk.png", CI_LVL)
                    utils.plot_out_dist(mean, std, self.config.CLASSES, self.config.RESULT_FOLDER + "/" + model + "/out_dist_unk.png")
    
                else:
                    utils.plot_st_resp([unk.mean(dim=0)[:, :, [0, 3, 6, 9]]], ["unk resp."],
                                       BIN_SIZE, self.config.RESULT_FOLDER + "/" + model + "/st_resp_unk.png")
                    unk = unk.mean(dim=3).mean(dim=2)
                    utils.plot_series([unk.mean(dim=0)], [unk.std(dim=0)], ["unk resp."],
                                      BIN_SIZE, self.config.RESULT_FOLDER + "/" + model + "/time_resp_unk.png", CI_LVL)
        
        # Plot kernels
        print("Plotting network kernels")
        connections = {"inpt": ("X", "Y"), "exc": ("Y", "Y"), "inh": ("Z", "Y")}
        lin_coord = self.network.coord_y_disc.view(-1) * self.config.GRID_SHAPE[2] + self.network.coord_x_disc.view(-1)
        knl_idx = [torch.nonzero(lin_coord == i) for i in range(self.config.GRID_SHAPE[1] * self.config.GRID_SHAPE[2])]
        knl_idx = [knl_idx[i][0] if len(knl_idx[i]) > 0 else None for i in range(len(knl_idx))]
        for name, conn in connections.items():
            w = self.network.connections[conn].w.t()
            lin_coord = lin_coord.to(w.device)
            kernels = torch.zeros(self.config.GRID_SHAPE[1] * self.config.GRID_SHAPE[2], self.config.GRID_SHAPE[1], self.config.GRID_SHAPE[2], device=w.device)
            if name != "inpt":
                w = w.view(self.config.NEURON_SHAPE[0] * self.config.NEURON_SHAPE[1], self.config.NEURON_SHAPE[0] * self.config.NEURON_SHAPE[1])
                w_red = torch.zeros(self.config.NEURON_SHAPE[0] * self.config.NEURON_SHAPE[1], self.config.GRID_SHAPE[1] * self.config.GRID_SHAPE[2], device=w.device)
                for i in range(w.size(1)): w_red[:, lin_coord[i]] += w[:, i]
                w = w_red
            w = w.view(self.config.NEURON_SHAPE[0] * self.config.NEURON_SHAPE[1], self.config.GRID_SHAPE[1], self.config.GRID_SHAPE[2])
            for i in range(kernels.size(0)):
                if knl_idx[i] is not None:
                    kernels[i, :, :] = w[knl_idx[i], :, :]
            utils.plot_grid(kernels, path=self.config.RESULT_FOLDER + "/weights_" + name + ".png", num_rows=self.config.GRID_SHAPE[1], num_cols=self.config.GRID_SHAPE[2])
        
        # Calculate accuracy on test set
        print("Evaluating test accuracy...")
        self.eval_pass(self.tst_set, train=False)
        print("Test accuracy: " + str(100 * self.stats_manager.eval_accuracy[-1]) + "%")
        
        print("Finished!")
    
    def run_experiment(self):
        if self.mode == P.MODE_TRN: self.run_train()
        if self.mode == P.MODE_TST: self.run_test()

if __name__=='__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=P.DEFAULT_CONFIG, help="The configuration you want to run for this experiment.")
    parser.add_argument("--mode", default=P.DEFAULT_MODE, choices=[P.MODE_TRN, P.MODE_TST], help='Whether you want to run a training or test experiment.')
    parser.add_argument("--seed", type=int, default=P.DEFAULT_SEED, help="The RNG seed to be used for this experiment.")
    
    args = parser.parse_args()

    # Launch experiment
    config = C.CONFIGURATIONS[args.config]
    experiment = Experiment(config, args.mode, args.seed)
    experiment.run_experiment()

