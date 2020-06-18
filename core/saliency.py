import numpy as np
from ensemble import get_ensemble_predicted_output as get_ensemble_output
from ensemble import get_ensemble_predicted_outputs as get_ensemble_outputs
import matplotlib.pyplot as plt
from vis.visualization import get_saliency_optimizer
from vis.visualization import visualize_saliency_with_optimizer
from vis.visualization import get_cam_optimizer
from vis.visualization import visualize_cam_with_optimizer
from tqdm import tqdm

# Get optimisers for each network's output node to speed up saliency processing
def get_ensemble_optimisers(ensemble, grad_cam = False):
    optimisers = []
    for i in range(0,len(ensemble)):
        network_optmisers = []
        for j in range(0,10):
            if not grad_cam:
                opt = get_saliency_optimizer(ensemble[i],9,filter_indices=j)
            else:
                opt = get_cam_optimizer(ensemble[i],9,filter_indices=j)
            network_optmisers.append(opt)
        optimisers.append(network_optmisers)

    return optimisers

# ------ Functions for one input ------

# Calculate saliency maps for each network in an ensemble for a single input
def generate_saliency_maps_for_one_input(ensemble,input,optimisers,visualised_layer, grad_cam = False):
    output_node = get_ensemble_output(ensemble,input,len(optimisers))
    saliency_maps = np.zeros((len(ensemble),input.shape[0],input.shape[1]))
    for i in range(0,len(ensemble)):
        if not grad_cam:
            saliency_maps[i] = visualize_saliency_with_optimizer(model = ensemble[i],layer_idx = visualised_layer, opt = optimisers[i][output_node], seed_input = input)
        else:
            saliency_maps[i] = visualize_cam_with_optimizer(model = ensemble[i],layer_idx = visualised_layer, opt = optimisers[i][output_node], seed_input = input)
    return(saliency_maps)

# Compute difference of saliency maps
def generate_uncertainty_map(saliency_maps):
    return(np.divide(np.std(saliency_maps,axis=0),np.average(saliency_maps,axis=0), out = np.zeros_like(np.std(saliency_maps, axis=0)), where = (np.average(saliency_maps, axis = 0) != 0)))

# Compute difference of saliency maps
def calculate_uncertainty_with_maps(saliency_maps):
    return(np.mean(generate_uncertainty_map(saliency_maps)))

# ------ Functions for many inputs ------

# Calculate saliency maps for a single network for many inputs
def generate_saliency_maps_for_multiple_inputs(network,inputs,output_nodes,
                                               network_optimisers,visualised_layer,
                                               grad_cam = False):

    saliency_maps = np.zeros(np.shape(inputs)[:-1])
    for input_idx in tqdm(range(0,np.size(inputs,axis=0))):
        input = inputs[input_idx]
        output_node = output_nodes[input_idx]
        if not grad_cam:
            saliency_maps[input_idx] = visualize_saliency_with_optimizer(model = network,
                                                                     layer_idx = 9,
                                                                     opt = network_optimisers[output_node],
                                                                     seed_input = input)
        else:
            saliency_maps[input_idx] = visualize_cam_with_optimizer(model = network,
                                                                     layer_idx = 9,
                                                                     opt = network_optimisers[output_node],
                                                                     seed_input = input)

    return(saliency_maps)

# Calculate saliency maps for each network in an ensemble for many inputs
def generate_ensemble_saliency_maps_for_multiple_inputs(ensemble,
                                                        inputs,output_nodes,
                                                        optimisers,visualised_layer, grad_cam = False):
    saliency_maps = []
    for network_idx in range(0,len(ensemble)):
        saliency_maps.append(generate_saliency_maps_for_multiple_inputs(
                                ensemble[network_idx],
                                inputs,
                                output_nodes,
                                optimisers[network_idx],
                                9, grad_cam))

    saliency_maps = np.swapaxes(saliency_maps,0,1)
    return(saliency_maps)

# Calculate uncertainty values with maps as input
def calculate_uncertainties_with_maps(saliency_maps):
    uncertainties = np.zeros(np.size(saliency_maps,axis=0))

    for i in range(0,np.size(saliency_maps, axis=0)):
        uncertainties[i] = calculate_uncertainty_with_maps(saliency_maps[i])

    return uncertainties

#------- wrapper functions -------

# Wrapper function to arrive at uncertainty output using ensemble and input
def calculate_uncertainty_for_one_input(ensemble,input,optimisers,visualised_layer, grad_cam = False):

    saliency_maps = generate_saliency_maps_for_one_input(ensemble = ensemble,
                                  input = input,
                                  optimisers = optimisers,
                                  visualised_layer = visualised_layer,
                                  grad_cam = grad_cam)

    uncertainty_map = generate_uncertainty_map(saliency_maps)

    return(np.average(uncertainty_map))

# Wrapper function to arrive at uncertainty outputs using ensemble and inputs
def calculate_uncertainties_for_multiple_inputs(ensemble,inputs,output_nodes,optimisers,visualised_layer, grad_cam = False):

    saliency_maps = generate_ensemble_saliency_maps_for_multiple_inputs(ensemble = ensemble,
                                  inputs = inputs,
                                  output_nodes = output_nodes,
                                  optimisers = optimisers,
                                  visualised_layer = visualised_layer,
                                  grad_cam = grad_cam)

    uncertainties = calculate_uncertainties_with_maps(saliency_maps)

    return uncertainties

#------- visualizations ------

#Function to visualise the multiple saliency maps
def visualize_saliency_maps(input,saliency_maps):
    fig, ax = plt.subplots(nrows=1, ncols=len(saliency_maps)+1, figsize = (15,15))
    i = 1
    for s_map in saliency_maps:
        ax[i].imshow(s_map)
        i = i+1
    ax[0].imshow(input)
    plt.show()
