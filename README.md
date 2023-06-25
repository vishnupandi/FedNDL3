# FedNDL3
On the Convergence of Decentralized Federated Learning Under Imperfect Information Sharing

his repository provides code for Decentralized ML training with noisy communication channel (both deep learning, compatible with PyTorch, and traditional convex machine learning models.

We provide code for the main experiments in the paper - On the Convergence of Decentralized Federated Learning Under Imperfect Information Sharing

Please refer to the folder main and start with the code 'main_file.py' for more details. The oode is highly self-contained and requires only basic packages that are widely available in all versions of python. Please contact me if you have any issues running the code. The results in the paper can be reproduced with this code. The code also has an option to run multiple times (different random seeds) and the results can be aggregated during plotting. Please use the plots_loss to plot the loss vs iterations and plots_consensus to plot the consensus error vs iterations.

The data needs to be generated from the Data_Reader for the first time. 

References
If you use the code, please cite the following papers:

@article{chellapandi2023convergence,
  title={On the convergence of decentralized federated learning under imperfect information sharing},
  author={Chellapandi, Vishnu Pandi and Upadhyay, Antesh and Hashemi, Abolfazl and \.{Z}ak, Stanislaw H},
  journal={arXiv preprint arXiv:2303.10695},
  year={2023}
}
