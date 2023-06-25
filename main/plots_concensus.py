import sys
import os
curr_dir = os.path.dirname(__file__)
# sys.path.append(curr_dir)
sys.path.append(os.getcwd())
from dec_opt.utils import unpickle_dir
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import itertools
import string 

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams["font.family"] = "Times New Roman"

def plot_results(repeats, label, plot='train',
                 optima=0.0, line_style=None, line_width=5, marker=None, scale=1,marker_size=None,markevery=5,alpha=1,zorder=5):
    scores = []
    for result in repeats:
        if plot == 'train':
            loss_val = result[0] 
        elif plot=='consensus':
            loss_val = result[2]
        else: 
            result[1]
        loss_val = loss_val - optima
        scores += [loss_val]

    scores = np.array(scores)
    mean = np.mean(scores, axis=0)
    x = np.arange(mean.shape[0]) * scale
    UB = mean + np.std(scores, axis=0)
    LB = mean - np.std(scores, axis=0)

    plt.plot(x, mean, label=label, linewidth=line_width, linestyle=line_style, marker=marker,markersize=marker_size,markevery=markevery,alpha=alpha,zorder=zorder)
    plt.fill_between(x, LB, UB, alpha=0.2, linewidth=1)


if __name__ == '__main__':
    plt.close('all')
    plt.figure()
    fig = plt.gcf()
    # data_set = 'syn2/'
    # optimal_baseline = baselines[data_set]

    # baseline - Gradient Descent Vanilla Centralized
    # baselines = unpickle_dir(d='./results/baselines')
    # repeats_baseline = baselines[data_set + '_gd']
    # no communication
    # repeats_disconnected = baselines[data_set + '_dis']

    # Specify what result runs you want to plot together
    # this is what you need to modify

    # MNIST
    results_dir = '.'

        
        
    data_loop = ['syn1','syn2'] #,'mnist']
    data_loop = ['mnist']

    for data1 in data_loop:

        topo_loop = ['ring' ] #, 'torus'] #,'fully_connected']
        topo_loop = ['ring', 'torus','fully_connected']
        # topo_loop = [ 'fully_connected']
        for topo in topo_loop:

            print(data1)
            print('')
            print(topo)
            plt.close('all')
            plt.figure()
            fig = plt.gcf()

            var_loop=[0.002,0.005,0.0075]
            
            for loop in (var_loop):

                
                # data = unpickle_dir(d='../results/' + data_set + results_dir)
                # plt.title('Non-Convex Objective(SYN-2): DeLiCoCo vs Choco-GD', fontsize=14)
                topo = topo #'ring'
                # topo = 'torus'
                # dataset = 'fmnist'
                dataset = data1 #'mnist'
                alg = '1'  
                n_cores = '16'  
                var1 = '0'
                var= str(loop) #'0'
                # var = '0.1'
                # var = '0.001'
                # var='0.0001' 
                # var = '1'
                pt = 'consensus'

                if dataset=='syn1': 
                    obj = 'Convex Objective'
                else:    
                    obj = 'Non-Convex Objective'

                data = unpickle_dir(d=curr_dir+'\\results'  + dataset + '//'+topo+'//'+results_dir)

                # plt.title(str(obj+'('+(dataset.upper())+')'+ '-'+topo.upper()+' : FedNDL Approaches - Consensus Error'), fontsize=12)
                # plt.subplot(1, 1, 1)
                # b=str(dataset+'.a_FedNDL'+alg+'.n_16.t_'+topo+'.var1e-08')

                # plot_results(repeats=data[str(dataset+'.a_choco-sgd.n_16.t_'+topo+'.var0.0')],       label='Choco-SGD', line_width=width, scale=1)    
                width = 3;
                markersize=12;
                # if  var=='0':  
                # 

                plot_results(repeats=data[str(dataset+'.a_FedNDL'+'1'+'.n_16.t_'+topo+'.var0.0')],    plot=pt,      label=str('FedNDL1'+'_Var=0.0'), line_width=1, scale=1,marker='o',marker_size=markersize,line_style='--',markevery=(1,20),zorder=4)
                # plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2'+'.n_16.t_'+topo+'.var0.0')],          label=str('FedNDL2'+'_Var=0.0'), line_width=width, scale=1,marker='*',line_style='--',marker_size=markersize)
                plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2'+'.n_16.t_'+topo+'.var0.0')],   plot=pt,       label=str('FedNDL2'+'_Var=0.0'),  line_width=1, scale=1,marker='P',marker_size=markersize,line_style='--',markevery=(7,20),zorder=4)
                # plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2-Nedic'+'.n_16.t_'+topo+'.var0.0')],          label=str('FedNDL2-Nedic'+'_Var=0.0'), line_width=width, scale=1,marker='*',line_style='--')
                # plot_results(repeats=data[str(dataset+'.a_FedNDL'+'3'+'.n_16.t_'+topo+'.var0.0')],          label=str('FedNDL3'+'_Var=0.0'), line_width=width, scale=1,marker='v',line_style='--',marker_size=markersize)
                plot_results(repeats=data[str(dataset+'.a_FedNDL'+'3'+'.n_16.t_'+topo+'.var0.0')],   plot=pt,       label=str('FedNDL3'+'_Var=0.0'), line_width=1, scale=1,marker='X',marker_size=markersize,line_style='--',markevery=(15,20))

                if  var=='0.005':   
                    # plot_results(repeats=data[str(dataset+'.a_FedNDL'+'1'+'.n_16.t_'+topo+'.var0.005')],          label=str('FedNDL1'+'_Var=0.005'), line_width=0, scale=1,marker='o',marker_size=markersize,zorder=1)
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'1'+'.n_16.t_'+topo+'.var0.005')],  plot=pt,        label=str('FedNDL1'+'_Var=0.005'), line_width=width, scale=1,line_style='--',zorder=3)
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2'+'.n_16.t_'+topo+'.var0.005')],  plot=pt,        label=str('FedNDL2'+'_Var=0.005'), line_width=width, scale=1,line_style='--',zorder=2)
                    # plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2-Nedic'+'.n_16.t_'+topo+'.var0.005')],          label=str('FedNDL2'+'_Var=0.005'), line_width=width, scale=1,line_style='--')
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'3'+'.n_16.t_'+topo+'.var0.005')],  plot=pt,        label=str('FedNDL3'+'_Var=0.005'), line_width=width, scale=1,line_style='--',alpha=0.7,zorder=1)


                    # plot_results(repeats=data[str(dataset+'.a_FedNDL'+'1'+'.n_16.t_'+topo+'.var0.0')], plot=pt,         label=str('FedNDL1'+'_Var=0.0'), line_width=width, scale=1,marker='o',marker_size=markersize)
                # plot_results(repeats=data[str(dataset+'.a_FedNDL'+'1'+'.n_16.t_'+topo+'.var0.0')],  plot=pt,         label=str('FedNDL1'+'_Var=0.0'), line_width=width, scale=1,marker='o',marker_size=markersize)
                # plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2'+'.n_16.t_'+topo+'.var0.0')],  plot=pt,         label=str('FedNDL2'+'_Var=0.0'), line_width=width, scale=1,line_style='--',marker_size=markersize)
                # plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2-Nedic'+'.n_16.t_'+topo+'.var0.0')],          label=str('FedNDL2-Nedic'+'_Var=0.0'), line_width=width, scale=1,marker='*',line_style='--')
                # plot_results(repeats=data[str(dataset+'.a_FedNDL'+'3'+'.n_16.t_'+topo+'.var0.0')],  plot=pt,         label=str('FedNDL3'+'_Var=0.0'), line_width=width, scale=1)
                # else:
            #     plot_results(repeats=data[str(dataset+'.a_FedNDL'+'1'+'.n_16.t_'+topo+'.var0.0')],          label=str('FedNDL1'+'_Var=0.0'), line_width=width, scale=1)

                if  var=='0.1':   
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'1'+'.n_16.t_'+topo+'.var0.1')], plot=pt,          label=str('FedNDL1'+'_Var=0.1'), line_width=width, scale=1,marker='o',marker_size=markersize)
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2'+'.n_16.t_'+topo+'.var0.1')], plot=pt,         label=str('FedNDL2'+'_Var=0.1'), line_width=width, scale=1,line_style='--')
                    # plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2-Nedic'+'.n_16.t_'+topo+'.var0.1')],          label=str('FedNDL2'+'_Var=0.1'), line_width=width, scale=1,line_style='--')
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'3'+'.n_16.t_'+topo+'.var0.1')],   plot=pt,        label=str('FedNDL3'+'_Var=0.1'), line_width=width, scale=1)

                if  var=='0.05':   
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'1'+'.n_16.t_'+topo+'.var0.05')], plot=pt,          label=str('FedNDL1'+'_Var=0.05'), line_width=width, scale=1,marker='o',marker_size=markersize)
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2'+'.n_16.t_'+topo+'.var0.05')], plot=pt,         label=str('FedNDL2'+'_Var=0.05'), line_width=width, scale=1,line_style='--')
                    # plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2-Nedic'+'.n_16.t_'+topo+'.var0.05')],          label=str('FedNDL2'+'_Var=0.05'), line_width=width, scale=1,line_style='--')
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'3'+'.n_16.t_'+topo+'.var0.05')],   plot=pt,        label=str('FedNDL3'+'_Var=0.05'), line_width=width, scale=1)

                if  var=='0.01':   
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'1'+'.n_16.t_'+topo+'.var0.01')], plot=pt,          label=str('FedNDL1'+'_Var=0.01'), line_width=width, scale=1,marker='o',marker_size=markersize)
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2'+'.n_16.t_'+topo+'.var0.01')], plot=pt,         label=str('FedNDL2'+'_Var=0.01'), line_width=width, scale=1,line_style='--')
                    # plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2-Nedic'+'.n_16.t_'+topo+'.var0.01')],          label=str('FedNDL2'+'_Var=0.01'), line_width=width, scale=1,line_style='--')
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'3'+'.n_16.t_'+topo+'.var0.01')],   plot=pt,        label=str('FedNDL3'+'_Var=0.01'), line_width=width, scale=1)

                if  var=='0.001':   
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'1'+'.n_16.t_'+topo+'.var0.001')], plot=pt,          label=str('FedNDL1'+'_Var=0.001'), line_width=width, scale=1,marker='o',marker_size=markersize)
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2'+'.n_16.t_'+topo+'.var0.001')], plot=pt,         label=str('FedNDL2'+'_Var=0.001'), line_width=width, scale=1,line_style='--')
                    # plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2-Nedic'+'.n_16.t_'+topo+'.var0.001')],          label=str('FedNDL2'+'_Var=0.001'), line_width=width, scale=1,line_style='--')
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'3'+'.n_16.t_'+topo+'.var0.001')],   plot=pt,        label=str('FedNDL3'+'_Var=0.001'), line_width=width, scale=1)

                if  var=='0.002':   
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'1'+'.n_16.t_'+topo+'.var0.002')], plot=pt,          label=str('FedNDL1'+'_Var=0.002'), line_width=width, scale=1,marker='o',marker_size=markersize)
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2'+'.n_16.t_'+topo+'.var0.002')], plot=pt,         label=str('FedNDL2'+'_Var=0.002'), line_width=width, scale=1,line_style='--')
                    # plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2-Nedic'+'.n_16.t_'+topo+'.var0.002')],          label=str('FedNDL2'+'_Var=0.002'), line_width=width, scale=1,line_style='--')
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'3'+'.n_16.t_'+topo+'.var0.002')],   plot=pt,        label=str('FedNDL3'+'_Var=0.002'), line_width=width, scale=1)
                # if  var=='0.005':   
                #     plot_results(repeats=data[str(dataset+'.a_FedNDL'+'1'+'.n_16.t_'+topo+'.var0.005')], plot=pt,          label=str('FedNDL1'+'_Var=0.005'), line_width=width, scale=1,marker='o',marker_size=markersize)
                #     plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2'+'.n_16.t_'+topo+'.var0.005')], plot=pt,         label=str('FedNDL2'+'_Var=0.005'), line_width=width, scale=1,line_style='--')
                #     # plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2-Nedic'+'.n_16.t_'+topo+'.var0.005')],          label=str('FedNDL2'+'_Var=0.005'), line_width=width, scale=1,line_style='--')
                #     plot_results(repeats=data[str(dataset+'.a_FedNDL'+'3'+'.n_16.t_'+topo+'.var0.005')],   plot=pt,        label=str('FedNDL3'+'_Var=0.005'), line_width=width, scale=1)
                if  var=='0.0075':   
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'1'+'.n_16.t_'+topo+'.var0.0075')], plot=pt,          label=str('FedNDL1'+'_Var=0.0075'), line_width=width, scale=1,marker='o',marker_size=markersize)
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2'+'.n_16.t_'+topo+'.var0.0075')], plot=pt,         label=str('FedNDL2'+'_Var=0.0075'), line_width=width, scale=1,line_style='--')
                    # plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2-Nedic'+'.n_16.t_'+topo+'.var0.0075')],          label=str('FedNDL2'+'_Var=0.0075'), line_width=width, scale=1,line_style='--')
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'3'+'.n_16.t_'+topo+'.var0.0075')],   plot=pt,        label=str('FedNDL3'+'_Var=0.0075'), line_width=width, scale=1)


                if  var=='0.0001':   

                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'1'+'.n_16.t_'+topo+'.var0.0001')],  plot=pt,         label=str('FedNDL1'+'_Var=1e-4'), line_width=width, scale=1,marker='o',marker_size=markersize)
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2'+'.n_16.t_'+topo+'.var0.0001')],  plot=pt,         label=str('FedNDL2'+'_Var=1e-4'), line_width=width, scale=1,line_style='--')
                    # plot_results(repeats=data[str(dataset+'.a_FedNDL'+'2-Nedic'+'.n_16.t_'+topo+'.var0.0001')],          label=str('FedNDL2'+'_Var=1e-4'), line_width=width, scale=1,line_style='--')
                    plot_results(repeats=data[str(dataset+'.a_FedNDL'+'3'+'.n_16.t_'+topo+'.var0.0001')],  plot=pt,         label=str('FedNDL3'+'_Var=1e-4'), line_width=width, scale=1)
            
                # plt.legend(fontsize=16,loc='best')
                # plt.legend(fontsize=14,loc='upper right', bbox_to_anchor=(1,1.02),borderaxespad=0)
                # plt.yscale("log")
                plt.ylim(bottom=-0.1, top= 2)
                # plt.ylim(bottom=-0.25, top=1.0)
                # plt.xlim(left=0, right=100)
                # plt.xlabel('Total Bits Communicated', fontsize=14)
                plt.xlabel('Iterations', fontsize=20)
                # plt.ylabel('$f - f^*$', fontsize=20)
                plt.ylabel('Consensus Error', fontsize=20)
                plt.grid(axis='both')
                plt.tick_params(labelsize=18)
                # plt.tick_params(axis='y',labelsize=18,labelcolor='blue')

                # plt.show()
                plt.savefig(str('figures/'+dataset+'_'+topo+'_var'+var+'_plot'+pt+'.pdf'),bbox_inches  ='tight') 
                plt.savefig(str('figures_png/'+dataset+'_'+topo+'_var'+var+'_plot'+pt+'.png'),bbox_inches  ='tight')
                # plt.show(block=False)
                # plt.pause(2)
                plt.close()
