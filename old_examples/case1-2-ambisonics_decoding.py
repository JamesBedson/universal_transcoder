"""
Copyright (c) 2024 Dolby Laboratories, Amaia Sagasti
Copyright (c) 2023 Dolby Laboratories

Redistribution and use in source and binary forms, with or without modification, are permitted 
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions 
and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or 
promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED 
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
"""

import os
from pathlib import Path

import numpy as np
from universal_transcoder.auxiliars.get_cloud_points import (
    get_all_sphere_points,
    get_equi_t_design_points,
    get_equi_circumference_points,
)
from universal_transcoder.auxiliars.get_input_channels import (
    get_input_channels_ambisonics,
)
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.calculations.optimization import optimize


order = 5
output_layout = MyCoordinates.mult_points(
    np.array(
        [
            (-90, 0, 1),
            (0, 0, 1),
            (90, 0, 1),
            (180, 0, 1),
        ]
    )
)
output_layout = MyCoordinates.mult_points(
    np.array(
        [
            (-135, 0, 1),
            (-45, 0, 1),
            (45, 0, 1),
            (135, 0, 1),
        ]
    )
)
layout_704 = MyCoordinates.mult_points(
    np.array(
        [
            (-135, 45, 1),
            (-120, 0, 1),
            (-90, 0, 1),
            (-45, 45, 1),
            (-30, 0, 1),
            (0, 0, 1),
            (30, 0, 1),
            (45, 45, 1),
            (90, 0, 1),
            (120, 0, 1),
            (135, 45, 1),
        ]
    )
)

layout_50 = MyCoordinates.mult_points(
    np.array(
        [
            (-120, 0, 1),
            (-30, 0, 1),
            (0, 0, 1),
            (30, 0, 1),
            (120, 0, 1),
        ]
    )
)

# cloud_optimization = get_equi_circumference_points(36, False)
# cloud_plots = get_equi_circumference_points(360, False)

basepath = Path(__file__).resolve().parents[0]
t_design = (
    basepath / "universal_transcoder" / "encoders" / "t-design" / "des.3.56.9.txt"
)
cloud_optimization = get_equi_t_design_points(t_design, False)
cloud_plots = get_all_sphere_points(1, False)
# cloud_optimization = get_equi_circumference_points(36, False)
# cloud_plots = get_equi_circumference_points(360, False)
input_matrix_optimization = get_input_channels_ambisonics(cloud_optimization, order)
input_matrix_plots = get_input_channels_ambisonics(cloud_plots, order)

output_layout = layout_704
dictionary = {
    "input_matrix_optimization": input_matrix_optimization,
    "cloud_optimization": cloud_optimization,
    "output_layout": layout_704,
    "coefficients": {
        "energy": 0,
        "radial_intensity": 0,
        "transverse_intensity": 0,
        "pressure": 1,
        "radial_velocity": 0.0001,
        "transverse_velocity": 0.0001,
        "in_phase_quad": 0,
        "symmetry_quad": 0,
        "in_phase_lin": 0,
        "symmetry_lin": 0.0001,
        "total_gains_lin": 0,
        "total_gains_quad": 0,
    },
    "directional_weights": 1,
    "show_results": False,
    "results_file_name": "ex2_ambi5OAto704_pv",
    "save_results": True,
    "input_matrix_plots": input_matrix_plots,
    "cloud_plots": cloud_plots,
}

decoding_matrix = optimize(dictionary)
print(decoding_matrix)


# '''
from universal_transcoder.calculations.energy_intensity import (
    energy_calculation,
    intensity_calculation,
    radial_I_calculation,
    transverse_I_calculation,
)
from universal_transcoder.calculations.pressure_velocity import (
    pressure_calculation,
    velocity_calculation,
    radial_V_calculation,
    transversal_V_calculation,
)
from universal_transcoder.plots_and_logs.paper_plots_to_R import (
    save_physics_to_file,
)

input_matrix_plots = get_input_channels_ambisonics(cloud_plots, order)

speaker_signals = np.dot(input_matrix_plots, decoding_matrix.T)

energy = energy_calculation(speaker_signals)
radial_I = radial_I_calculation(cloud_plots, speaker_signals, output_layout)
transverse_I = transverse_I_calculation(cloud_plots, speaker_signals, output_layout)
pressure = pressure_calculation(speaker_signals)
radial_V = radial_V_calculation(cloud_plots, speaker_signals, output_layout)
transverse_V = transversal_V_calculation(cloud_plots, speaker_signals, output_layout)

coordinates = cloud_plots.sph_deg()
azimuth = coordinates[:, 0]
elevation = coordinates[:, 1]

save_physics_to_file(
    azimuth,
    elevation,
    pressure,
    radial_V,
    transverse_V,
    energy,
    radial_I,
    transverse_I,
    "ex2_ambi5OAto704_pv",
)
# '''
