"""
Copyright (c) 2024 Dolby Laboratories, Amaia Sagasti
Copyright (c) 2023 Dolby Laboratories
Modified and extended by James Bedson, 2025

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

from typing import Union, cast
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.mpl.geoaxes as cgeo
from matplotlib.ticker import FuncFormatter

from universal_transcoder.plots_and_logs.plot_utils import *
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.auxiliars.typing import ArrayLike, Array
from universal_transcoder.plots_and_logs.common_plots_functions import save_plot


def plot_pv_2D(
    pressure: Array,
    radial_v: Array,
    transverse_v: Array,
    cloud_points: MyCoordinates,
    save_results: bool,
    results_file_name: Union[bool, str] = False,
) -> None:
    """
    Function to plot the pressure and velocity when
    decoding from an input format to an output layout. 2D plots.

    Args:
        pressure (jax.numpy Array): contains the real pressure values for each virtual source (1xL)
        radial_v (jax.numpy array): contains the real adial velocity values for each virtual
                source (1xL)
        transverse_v (jax.numpy array): contains the real transverse velocity values for each virtual
                source (1xL)
        cloud(MyCoordinates): set of points sampling the sphere (L)
        save_results (bool): Flag to save plots
        results_file_name(str): Path where to save the plots
    """

    azimuth = cloud_points.sph_rad()[:, 0]
    elevation = cloud_points.sph_rad()[:, 1]
    pressure, radial_v, transverse_v = map(np.asarray, (pressure, radial_v, transverse_v))
    mask_horizon = np.isclose(np.abs(elevation), np.min(np.abs(elevation))) & (
        elevation >= 0.0
    )

    azimuth = azimuth[mask_horizon]
    azimuth_polar = close_loop(azimuth)

    pressure = pressure[mask_horizon]
    pressure_polar = close_loop(pressure)

    radial_v = radial_v[mask_horizon]
    radial_v_polar = close_loop(radial_v)

    transverse_v = transverse_v[mask_horizon]
    transverse_v_polar = close_loop(transverse_v)

    # Calculations
    v = np.sqrt(radial_v**2 + transverse_v**2)

    # Plot Limits
    lim = compute_plot_limit(pressure, radial_v, transverse_v)

    # -- Plot --
    fig1 = plt.figure(figsize=(17, 9))

    # XY
    ax1 = fig1.add_subplot(121)
    ax1.plot(azimuth, pressure, label="Pressure")
    ax1.plot(azimuth, radial_v, label="Radial Velocity")
    ax1.plot(azimuth, transverse_v, label="Transverse Velocity")
    ax1.legend(bbox_to_anchor=(1.5, 1.1))
    ax1.set_title("Pressure and Velocity (Linear)")
    ax1.set_ylim(-0.01, lim)
    
    # Polar
    ax2 = fig1.add_subplot(122, projection="polar")
    ax2.set_theta_zero_location("N")
    ax2.plot(azimuth_polar, pressure_polar, label="Pressure")
    ax2.plot(azimuth_polar, radial_v_polar, label="Radial Velocity")
    ax2.plot(azimuth_polar, transverse_v_polar, label="Transverse Velocity")
    ax2.legend(bbox_to_anchor=(1.5, 1.1))
    ax2.set_ylim(0, lim)

    plt.tight_layout()

    # Save plots
    if save_results and isinstance(results_file_name, str):
        save_plot(plt, results_file_name, "plot_pressure_velocity_2D.png")

    # -- dB Scale --
    fig = plt.figure(figsize=(17, 9))
    pressure_db = 20 * np.log10(pressure)
    pressure_polar_db = 20 * np.log10(pressure_polar)

    # XY
    ax3 = fig.add_subplot(121)
    ax3.plot(azimuth, pressure_db, label="Pressure (dB)")
    ax3.legend(bbox_to_anchor=(1.5, 1.1))
    ax3.set_ylim(-24, +6)
    ax3.set_title("Pressure (dB)")

    # Polar
    ax4 = fig.add_subplot(122, projection="polar")
    ax4.set_theta_zero_location("N")
    ax4.plot(azimuth_polar, pressure_polar_db, label="Pressure (dB)")
    ax4.legend(bbox_to_anchor=(1.3, 1.1))
    ax4.set_ylim(-24, +6)

    if save_results and isinstance(results_file_name, str):
        save_plot(plt, results_file_name, "plot_pressure_velocity_2D_dB.png")


def plot_pv_3D(
    pressure: ArrayLike,
    radial_v: ArrayLike,
    transverse_v: ArrayLike,
    cloud_points: MyCoordinates,
    save_results: bool,
    results_file_name: Union[bool, str] = False,
) -> None:
    """
    Function to plot the pressure and velocity when
    decoding from an input format to an output layout.
    3D projection on 2D plots.

    Args:
        pressure (jax.numpy Array): contains the real pressure values for each virtual source (1xL)
        radial_v (jax.numpy array): contains the real adial velocity values for each virtual
                source (1xL)
        transverse_v (jax.numpy array): contains the real transverse velocity values for each virtual
                source (1xL)
        cloud(MyCoordinates): set of points sampling the sphere (L)
        save_results (bool): Flag to save plots
        results_file_name(str): Path where to save the plots
    """
    # Preparation
    points = cloud_points.sph_deg()
    pressure, radial_v, transverse_v = map(np.asarray, (pressure, radial_v, transverse_v))
    x = points[:, 0]
    y = points[:, 1]
    data_crs = ccrs.PlateCarree()
    proj = ccrs.Robinson(central_longitude=0)
    fig = plt.figure(figsize=(17,9))

    # Pressure
    ax1 = fig.add_subplot(311, projection=proj)
    cartopy_scatter(x, y, data_crs, ax1, pressure, "Pressure", "Pressure", (0, 2))
    
    # Radial Velocity
    ax2 = fig.add_subplot(312, projection=proj)
    cartopy_scatter(x, y, data_crs, ax2, radial_v, "Radial Velocity", "Radial Velocity", (0, 1))

    # Transverse Velocity
    ax3 = fig.add_subplot(313, projection=proj)
    cartopy_scatter(x, y, data_crs, ax3, transverse_v, "Transverse Velocity", "Transverse Velocity", (0, 1))
    
    plt.tight_layout()

    if save_results and isinstance(results_file_name, str):
        save_plot(plt, results_file_name, "plot_pressure_velocity_3D.png")