import numpy as np
import time
from dec_opt.gossip_matrix import GossipMatrix
import pandas as pd

class DecGD:
    def __init__(self, data_reader, hyper_param, model):
        self.A_train = data_reader.A_train
        self.y_train = data_reader.y_train
        self.A_test = data_reader.A_test
        self.y_test = data_reader.y_test

        self.param = hyper_param
        self.model = model

        self.W = GossipMatrix(topology=self.param.topology, n_cores=self.param.n_cores).W

        # initialize parameters for each node. Ax = y is the problem we are solving
        # ----------------------------------------------------------------------------------
        self.losses = np.zeros(self.param.epochs + 1)
        self.num_samples, self.num_features = self.A_train.shape
        INIT_WEIGHT_STD = 1 / np.sqrt(self.num_features)

        np.random.seed(self.param.seed)
        # self.model.x_estimate = np.random.normal(0, INIT_WEIGHT_STD, size=(self.num_features, self.param.n_cores))
        temp = np.random.normal(0, INIT_WEIGHT_STD, size=(self.num_features, 1))
        self.model.x_estimate = np.repeat(temp,self.param.n_cores,axis=1)

        # self.model.x_estimate = np.tile(self.model.x_estimate, (self.param.n_cores, 1)).T

        self.model.Z = np.zeros(self.model.x_estimate.shape)
        self.model.S = np.zeros(self.model.x_estimate.shape)

        # Now Distribute the Data among machines
        # ----------------------------------------
        self.data_partition_ix, self.num_samples_per_machine = self._distribute_data()

        # Decentralized Training
        # --------------------------
        self.train_losses, self.test_losses, self.consensus_error= self._dec_train()

    def _distribute_data(self):
        data_partition_ix = []
        num_samples_per_machine = self.num_samples // self.param.n_cores
        all_indexes = np.arange(self.num_samples)
        # np.random.shuffle(all_indexes)

        for machine in range(0, self.param.n_cores - 1):
            data_partition_ix += [
                all_indexes[num_samples_per_machine * machine: num_samples_per_machine * (machine + 1)]]
        # put the rest in the last machine
        data_partition_ix += [all_indexes[num_samples_per_machine * (self.param.n_cores - 1):]]
        print("All but last machine has {} data points".format(num_samples_per_machine))
        print("length of last machine indices:", len(data_partition_ix[-1]))
        return data_partition_ix, num_samples_per_machine

    def _dec_train(self):
        train_losses = np.zeros(self.param.epochs + 1)
        test_losses = np.zeros(self.param.epochs + 1)
        consensus_error = np.zeros(self.param.epochs + 1)      

        train_losses[0] = self.model.loss(self.A_train, self.y_train)
        test_losses[0] = self.model.loss(self.A_test, self.y_test)
        x_average = np.average(self.model.x_estimate,1)
        x_average=np.transpose(x_average[np.newaxis])
        # Consensus = np.linalg.norm(self.model.x_estimate-x_average,2)
        Consensus = self.model.x_estimate-x_average
        consensus_error[0] = np.average(np.sqrt(np.sum(Consensus**2, axis=0)))
        train_start = time.time()
        for epoch in np.arange(self.param.epochs):
            loss = self.model.loss(self.A_train, self.y_train)
            if np.isinf(loss) or np.isnan(loss):
                print("training exit - diverging")
                break
            lr = self.model.lr(epoch=epoch,
                               iteration=epoch,
                               num_samples=self.num_samples_per_machine)

            # Gradient step
            # --------------------------
            x_plus = np.zeros_like(self.model.x_estimate)


            # Communication step
            # -----------------------------------------
            if self.param.algorithm == 'exact_comm':
                # Xiao, Boyd; Fast Linear Iterations for Distributed Averaging
                # for t in 0...T 1 do in parallel for all workers i ∈[n]
                for machine in range(0, self.param.n_cores):
                    # Compute neg. Gradient (or stochastic gradient) based on algorithm
                    minus_grad = self.model.get_grad(A=self.A_train,
                                                    y=self.y_train,
                                                    stochastic=self.param.stochastic,
                                                    indices=self.data_partition_ix,
                                                    machine=machine)
                    x_plus[:, machine] = lr * minus_grad
                x_cap = self.model.x_estimate + x_plus
                self.model.x_estimate = x_cap @ self.W
            elif self.param.algorithm == 'FedNDL1':
                sig = np.sqrt(self.param.var_proposed) 
                # self.model.Noise = np.zeros(self.model.x_estimate.shape)
                self.model.Noise = np.random.normal(0.0, sig , (self.model.x_estimate.shape[0], self.model.x_estimate.shape[1]))
                W_Modified = np.copy(self.W)
                np.fill_diagonal(W_Modified,0)
                # local gradient update i.e. X_t0
                for loop in range(0,self.param.E):
                    for machine in range(0, self.param.n_cores):
                    # Compute neg. Gradient (or stochastic gradient) based on algorithm
                            minus_grad = self.model.get_grad(A=self.A_train,
                                                    y=self.y_train,
                                                    stochastic=self.param.stochastic,
                                                    indices=self.data_partition_ix,
                                                    machine=machine)
                            x_plus[:, machine] = lr * minus_grad
                    self.model.x_estimate = self.model.x_estimate + x_plus
                    # Gossip Averaging
                self.model.x_estimate =    self.model.x_estimate@ self.W +  self.model.Noise @ W_Modified
                x_average = np.average(self.model.x_estimate,1)
                x_average=np.transpose(x_average[np.newaxis])
                Consensus = self.model.x_estimate-x_average
                consensus_error_temp = np.average(np.sqrt(np.sum(Consensus**2, axis=0)))

            elif self.param.algorithm == 'FedNDL2':
                sig = np.sqrt(self.param.var_proposed) 
                # self.model.Noise = np.zeros(self.model.x_estimate.shape)
                self.model.Noise = np.random.normal(0.0, sig , (self.model.x_estimate.shape[0], self.model.x_estimate.shape[1]))
                W_Modified = np.copy(self.W)
                np.fill_diagonal(W_Modified,0)                # local gradient update i.e. X_t0
                    # Gossip Averaging
                self.model.x_estimate =     self.model.x_estimate  @ self.W +  self.model.Noise @ W_Modified
                
                for loop in range(0,self.param.E):
                    for machine in range(0, self.param.n_cores):
                    # Compute neg. Gradient (or stochastic gradient) based on algorithm
                        minus_grad = self.model.get_grad(A=self.A_train,
                                                    y=self.y_train,
                                                    stochastic=self.param.stochastic,
                                                    indices=self.data_partition_ix,
                                                    machine=machine)
                        x_plus[:, machine] = lr * minus_grad
                    self.model.x_estimate = self.model.x_estimate  + x_plus
                # self.model.x_estimate = self.model.x_estimate  + x_plus
                x_average = np.average(self.model.x_estimate,1)
                x_average=np.transpose(x_average[np.newaxis])
                Consensus = self.model.x_estimate-x_average
                consensus_error_temp = np.average(np.sqrt(np.sum(Consensus**2, axis=0)))

                
            elif self.param.algorithm == 'FedNDL3': # Transfer Gradients
                sig = np.sqrt(self.param.var_proposed) 
                # self.model.Noise = np.zeros(self.model.x_estimate.shape)
                self.model.Noise = np.random.normal(0.0, sig , (self.model.x_estimate.shape[0], self.model.x_estimate.shape[1]))
                W_Modified = np.copy(self.W)
                np.fill_diagonal(W_Modified,0)                # local gradient update i.e. X_t0
                grad_plus = np.zeros_like(self.model.x_estimate)
                temp = np.copy(self.model.x_estimate)

                for machine in range(0, self.param.n_cores):
                    # self.model.x_estimate = np.copy(temp)
                    for loop in range(0,self.param.E):
                                        # Compute neg. Gradient (or stochastic gradient) based on algorithm
                        minus_grad = self.model.get_grad(A=self.A_train,
                                                    y=self.y_train,
                                                    stochastic=self.param.stochastic,
                                                    indices=self.data_partition_ix,
                                                    machine=machine)
                        grad_plus[:, machine] =  minus_grad
                        x_plus[:, machine] = lr * minus_grad
                        self.model.x_estimate = self.model.x_estimate  + x_plus

                    # Gossip Averaging
                # self.model.x_estimate = self.model.x_estimate + lr*(grad_plus + self.model.Noise) @ self.W
                self.model.x_estimate = temp
                self.model.x_estimate = self.model.x_estimate + lr*(grad_plus@ self.W + self.model.Noise@ W_Modified)
                x_average = np.average(self.model.x_estimate,1)
                x_average=np.transpose(x_average[np.newaxis])
                Consensus = self.model.x_estimate-x_average
                consensus_error_temp = np.average(np.sqrt(np.sum(Consensus**2, axis=0)))
                    
                # df = pd.DataFrame(self.model.x_estimate )
                # df = pd.DataFrame(Consensus)
                # df.to_excel(excel_writer = "C:\\Users\\qi106\\OneDrive - Cummins\\Research\\DSGD_Noisy\\test.xlsx")

            elif self.param.algorithm == 'choco-sgd':
                # Koloskove,Stich,Jaggi; Decentralized Stochastic
                # Optimization and Gossip Algorithms with Compressed Communication
                # x_(t+1) = x_(t+1/2) + \gamma W.dot.(x^_j(t+1) - x^_i(t+1))
                # self.model.x_estimate = x_cap + \
                #         self.param.consensus_lr * self.model.x_hat.dot(self.W - np.eye(self.param.n_cores))
                self.model.x_estimate = x_cap + \
                        self.param.consensus_lr * x_cap.dot(self.W - np.eye(self.param.n_cores))
                pass
            else:
                print('Running Plain GD. n_cores = 1 else convergence results not guaranteed')
                # do nothing just plain GD
                self.model.x_estimate = x_cap

            train_losses[epoch + 1] = self.model.loss(self.A_train, self.y_train)
            test_losses[epoch + 1] = self.model.loss(self.A_test, self.y_test)
            consensus_error[epoch + 1] = consensus_error_temp

            # train_acc = compute_accuracy(model=self.model, feature=self.A_train, target=self.y_train)
            # test_acc = compute_accuracy(model=self.model, feature=self.A_test, target=self.y_test)
            print("epoch : {}; loss: {}; consensus: {}".
                  format(epoch, train_losses[epoch + 1],consensus_error[epoch + 1]))
            # print("epoch : {}; loss: {}; Test accuracy : {}".format(epoch, test_losses[epoch + 1], test_acc))
            if np.isinf(train_losses[epoch + 1]) or np.isnan(train_losses[epoch + 1]):
                print("Break training - Diverged")
                break
        print("Training took: {}s".format(time.time() - train_start))
        return train_losses, test_losses, consensus_error

