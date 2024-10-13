import yaml
import numpy as np
import os, sys
import shutil
from typing import Union
from pathlib import Path
import datetime

from universal_transcoder.auxiliars.typing import ArrayLike
from universal_transcoder.auxiliars.typing import NpArray

from universal_transcoder.auxiliars.get_cloud_points import (
    get_all_sphere_points, 
    get_equi_t_design_points
)
from universal_transcoder.auxiliars.get_input_channels import (
    get_input_channels_ambisonics,
    get_input_channels_vbap
)
from universal_transcoder.plots_and_logs.all_plots import plots_general
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.calculations.optimization import optimize
from constants import *

###############################################################################

def load_yaml(file_path: Union[os.PathLike, str]) -> dict:
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return {}
        

def get_ambisonics_matrix(data: dict, path_to_t_design: Union[os.PathLike, str]) -> tuple:
    order = data.get(CFG_AMBISONICS_ORDER)
    assert(order is not None)

    cloud_optimization  = get_equi_t_design_points(path_to_t_design, False)
    matrix              = get_input_channels_ambisonics(cloud_optimization, order)
    name                = f"{order}OA"
    
    return order, cloud_optimization, matrix, name


def get_surround_layout(surround_data: dict, surround_layout: str) -> tuple:
    positions = surround_data.get(surround_layout)
    assert(positions is not None)

    matrix_array    = np.array([tuple(position) for position in positions.get(CFG_SURROUND_POSITIONS)])
    matrix          = MyCoordinates.mult_points(matrix_array) 
    name            = f"{surround_layout}"

    return matrix, name


def process_audio_format(config: dict, optimisation_parameters: dict, format_option: str):
    settings    = config.get(CFG_SETTINGS)
    data        = {}
    
    if format_option == CFG_INPUT: 
        data = config.get(CFG_INPUT)

    elif format_option == CFG_OUTPUT:
        data = config.get(CFG_OUTPUT)   

    else:
        raise ValueError("Incorrect format option.")
    
    assert(data is not None)
    assert(settings is not None)

    name                    = ""
    matrix                  = ""
    audio_format_type       = data.get(CFG_TYPE_FORMAT)
    path_to_layouts         = "config_files/surround_layouts.yaml"
    
    if audio_format_type == CFG_TYPE_AMBISONICS:
        basepath            = Path(__file__).resolve().parents[1]
        path_to_t_design    = (
        basepath /
        "encoders" / 
        "t-design" /
        settings[CFG_T_DESIGN_PATH]
        )

        order, cloud_optimization, matrix, name = get_ambisonics_matrix(data, path_to_t_design)
        cloud_plots                             = get_all_sphere_points(1, False)

        optimisation_parameters[CFG_CLOUD_OPTIMIZATION] = cloud_optimization
        optimisation_parameters[CFG_CLOUD_PLOTS]        = cloud_plots

        if format_option == "input":
            optimisation_parameters[CFG_INPUT_MATRIX_PLOTS] = get_input_channels_ambisonics(cloud_plots, order)

    elif audio_format_type == CFG_TYPE_SURROUND:
        config_layouts  = load_yaml(path_to_layouts)
        assert(config_layouts is not None)
        
        matrix, name    = get_surround_layout(config_layouts, data.get(CFG_SURROUND_LAYOUT))

    elif audio_format_type == CFG_TYPE_VBAP:
        pass

    elif audio_format_type == CFG_TYPE_OBJECT:
        pass

    else: 
        raise ValueError("Unknown audio format.")
    

    if format_option == "input":
        optimisation_parameters[CFG_INPUT_MATRIX_OPTIMIZATION]  = matrix

    else:
        optimisation_parameters[CFG_OUTPUT_LAYOUT]              = matrix


    return optimisation_parameters, name

            


def init_config(path: Union[os.PathLike, str]) -> dict:
    ############################################################################
    # LOAD FILES
    config          = load_yaml(path)
    settings        = config.get(CFG_SETTINGS)
    coefficients    = config.get(CFG_COEFFICIENTS)

    assert(settings is not None)
    assert(coefficients is not None)

    ############################################################################
    # LOAD INPUT AND OUTPUT
    
    optimisation_parameters                 = {}
    optimisation_parameters, input_name     = process_audio_format(config, optimisation_parameters, CFG_INPUT)
    optimisation_parameters, output_name    = process_audio_format(config, optimisation_parameters, CFG_OUTPUT)

    ############################################################################
    # OPTIMISATION
    current_time        = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_file_name   = f"{input_name}_to_{output_name}_{current_time}.yaml"

    optimisation_parameters[CFG_COEFFICIENTS]           = coefficients
    optimisation_parameters[CFG_DIRECTIONAL_WEIGHTS]    = settings[CFG_DIRECTIONAL_WEIGHTS]
    optimisation_parameters[CFG_SHOW_RESULTS]           = settings[CFG_SHOW_RESULTS]
    optimisation_parameters[CFG_SAVE_RESULTS]           = settings[CFG_SAVE_RESULTS]
    optimisation_parameters[CFG_RESULTS_FILE_NAME]      = results_file_name

    return optimisation_parameters


if __name__ == "__main__":

    # Load parameters from config file
    optimisation_parameters = init_config("config_files/config.yaml")
    
    # Duplicate config parameters for future reference
    config_file_name = f"config_files/CONFIG_{optimisation_parameters[CFG_RESULTS_FILE_NAME]}"
    shutil.copy("config_files/config.yaml", config_file_name)

    # Run optimisation
    optimize(optimisation_parameters)