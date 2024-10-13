######################################################################################################################################
# James Bedson

# Example Transcodings: Ambisonics to Traditional Layouts
# Input Encodings: 1st, 2nd, 3rd order Ambisonics
# Output Decodings: 9.0.6, 7.0.4, 5.0.4
######################################################################################################################################

import os
import numpy as np
from pathlib import Path

from universal_transcoder.auxiliars.get_cloud_points import (
    get_all_sphere_points,
    get_equi_t_design_points,
)
from universal_transcoder.auxiliars.get_input_channels import (
    get_input_channels_ambisonics,
)

from universal_transcoder.plots_and_logs.all_plots import plots_general
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.calculations.optimization import optimize
from universal_transcoder.plots_and_logs.import_allrad_dec import get_allrad_decoder

basepath = Path(__file__).resolve().parents[1]

######################################################################################################################################
# POINT DISTRIBUTION + INPUT INITIALISATION

t_design = (
    basepath / "universal_transcoder" /
    "encoders" / "t-design" /
    "des.3.56.9.txt"                    # 3 Dimensional, 56-point 9-Design (integration up to 9th degree polynomials)
)

cloud_optimization          = get_equi_t_design_points(t_design, False)
ambisonics_orders           = [1, 2, 3]
input_matrix_optimizations  = [get_input_channels_ambisonics(cloud_optimization, order) for order in ambisonics_orders]

######################################################################################################################################
# LAYOUTS

# 9.0.6 Layout
# https://www.dolby.com/siteassets/about/support/guide/setup-guides/9.1.6-overhead-speaker-placement/9_1_6_overhead_speaker_setup.pdf
output_layout_9_0_6 = MyCoordinates.mult_points(
    np.array(
        [
            (30, 0, 1),     # L
            (-30, 0, 1),    # R
            
            (0, 0, 1),      # C
            
            (90, 0, 1),     # Ls
            (-90, 0, 1),    # Rs

            (135, 0, 1),    # Lb
            (-135, 0, 1),   # Rb
            
            (45, 45, 1),    # Tfl
            (-45, 45, 1),   # Tfr
            
            (90, 45, 1),    # Tcl
            (-90, 45, 1),   # Tcr

            (135, 45, 1),  # Tbl
            (-135, 45, 1), # Tbr
        ]
    )
)

# 7.0.4 Layout
output_layout_7_0_4 = MyCoordinates.mult_points(
    np.array(
        [
            (30, 0, 1),     # L
            (-30, 0, 1),    # R
            
            (0, 0, 1),      # C
            
            (90, 0, 1),     # Ls
            (-90, 0, 1),    # Rs
            
            (120, 0, 1),    # Lb
            (-120, 0, 1),   # Rb
            
            (45, 45, 1),    # Tfl
            (-45, 45, 1),   # Tfr
            
            (135, 45, 1),   # Tbl
            (-135, 45, 1),  # Tbr
        ]
    )
)

# 5.0.4 Layout
# https://www.dolby.com/siteassets/about/support/guide/setup-guides/5.1.4-overhead-speaker-placement/sell-sheet-5.1.4-mounted.pdf

output_layout_5_0_4 = MyCoordinates.mult_points(
    np.array(
        [
            (25, 0, 1),     # L
            (-25, 0, 1),    # R
            
            (0, 0, 1),      # C
            
            (115, 0, 1),    # Lb
            (-115, 0, 1),   # Rb
            
            (45, 45, 1),    # Tfl
            (-45, 45, 1),   # Tfr
            
            (135, 45, 1),   # Tbl
            (-135, 45, 1),  # Tbr
        ]
    )
)

all_output_layouts = [
    output_layout_9_0_6,  
    output_layout_7_0_4,
    output_layout_5_0_4
    ]

######################################################################################################################################
# OPTIMISATION
cloud_plots     = get_all_sphere_points(1, False)
input_names     = ["10A", "20A", "30A"]
output_names    = ["906", "704", "504"]

for input_idx, optimised_input_matrix in enumerate(input_matrix_optimizations):
    
    input_name          = input_names[input_idx] 
    input_matrix_plots  = get_input_channels_ambisonics(cloud_plots, ambisonics_orders[input_idx])

    for output_idx, output_layout in enumerate(all_output_layouts):

        output_name     = output_names[output_idx]  
        save_plot_name  = f"{input_name}_{output_name}_USAT"

        dictionary = {
            "input_matrix_optimization": optimised_input_matrix,
            "cloud_optimization": cloud_optimization,
            "output_layout": output_layout,
            "coefficients": {
                "energy": 5,
                "radial_intensity": 2,
                "transverse_intensity": 1,
                "pressure": 0,
                "radial_velocity": 0,
                "transverse_velocity": 0,
                "in_phase_quad": 10,
                "symmetry_quad": 2,
                "in_phase_lin": 0,
                "symmetry_lin": 0,
                "total_gains_lin": 0,
                "total_gains_quad": 0,
            },

            "directional_weights": 1,
            "show_results": False,
            "results_file_name": save_plot_name,
            "save_results": True,
            "input_matrix_plots": input_matrix_plots,
            "cloud_plots": cloud_plots,
        }

        optimize(dictionary)
        ######################################################################################################################################