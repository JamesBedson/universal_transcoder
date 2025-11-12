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

import math
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from typing import cast

from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
from matplotlib.projections.polar import PolarAxes

from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.auxiliars.typing import Array
from universal_transcoder.auxiliars.typing import ArrayLike
from universal_transcoder.calculations.energy_intensity import (
    angular_error,
    width_angle,
)
from universal_transcoder.plots_and_logs.common_plots_functions import save_plot
from universal_transcoder.plots_and_logs.plot_utils import *

def plot_polar_energy_intensity(
    azimuth: np.ndarray,
    energy: np.ndarray,
    radial_i: np.ndarray,
    transverse_i: np.ndarray,
    color_values: np.ndarray,
    color_label: str,
    cmap: str,
    vmax: float,
    lim: float,
    title: str,
    deg_scale: float = 1.0,
    ax: PolarAxes | None = None,
) -> PolarAxes:
    
    if ax is None:
        ax = plt.subplot(projection="polar")
    ax = cast(PolarAxes, ax)

    # Main curves
    ax.plot(azimuth, energy, label="Energy")
    ax.plot(azimuth, radial_i, label="Radial Intensity")
    ax.plot(azimuth, transverse_i, label="Transverse Intensity")

    # Color-coded bars
    radii = np.ones(color_values.size) * lim
    cmap_obj = plt.get_cmap(cmap)
    norm = Normalize(vmin=0.0, vmax=vmax)
    for t, r, w in zip(azimuth, radii, color_values):
        color = cmap_obj(norm(w))
        bar = ax.bar(t, r, width=w, bottom=0.0, edgecolor="none")
        bar[0].set_facecolor(color)

    # Colorbar formatter
    def deg_formatter(x, pos):
        return f"{np.rad2deg(x) * deg_scale:.0f}°"

    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj),
        ax=ax,
        pad=0.15,
        shrink=0.75,
        format=FuncFormatter(deg_formatter),
    )
    cbar.ax.set_ylabel(color_label)

    ax.legend(bbox_to_anchor=(1.5, 1.2))
    ax.set_theta_zero_location("N")
    ax.set_ylim(0, lim)
    ax.set_title(title)

    return ax

def plot_ei_2D(
    energy: Array,
    radial_i: Array,
    transverse_i: Array,
    cloud_points: MyCoordinates,
    save_results: bool,
    results_file_name=False,
) -> None:
    """
    Function to plot the energy and intensity when
    decoding from an input format to an output layout. 2D plots.

    Args:
        energy (Array): contains the energy values for each virtual source (1xL)
        radial_i (Array): contains the radial intensity values for each virtual
                source (1xL)
        transverse_i (array): contains the transverse intensity values for each virtual
                source (1xL)
        cloud(MyCoordinates): set of points sampling the sphere (L)
        save_results (bool): Flag to save plots
        results_file_name(str): Path where to save the plots
    """
    azimuth = cloud_points.sph_rad()[:, 0]
    elevation = cloud_points.sph_rad()[:, 1]
    energy, radial_i, transverse_i = map(np.asarray, (energy, radial_i, transverse_i))
    mask_horizon = (elevation < 0.01) * (elevation > -0.01)

    azimuth = azimuth[mask_horizon]
    azimuth = close_loop(azimuth)

    elevation = elevation[mask_horizon]
    elevation = close_loop(elevation)

    # Energy
    energy_db = 10 * np.log10(energy)
    energy_db = energy_db[mask_horizon]
    energy_db = close_loop(energy_db)

    energy = energy[mask_horizon]
    energy = close_loop(energy)

    # Radial Intensity
    radial_i = radial_i[mask_horizon]
    radial_i = close_loop(radial_i)

    # Transverse Intensity
    transverse_i = transverse_i[mask_horizon]
    transverse_i = close_loop(transverse_i)

    # Angular Error
    ang_err_deg = angular_error(radial_i, transverse_i)
    ang_err_rad = np.deg2rad(ang_err_deg)
    ang_err_rad = close_loop(ang_err_rad)

    # Source Width
    width_deg = width_angle(radial_i)
    width_rad = np.deg2rad(width_deg / 3)
    width_rad = close_loop(width_rad)
    
    # Plot limit
    lim = compute_plot_limit(energy, radial_i, transverse_i)

    fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "polar"}, figsize=(17,9))
    # Plot 1 - Energy, Radial, Transverse and Angular Error
    plot_polar_energy_intensity(
        azimuth,
        energy,
        radial_i,
        transverse_i,
        ang_err_rad,
        "Angular Error",
        "Greens",
        vmax=math.pi / 4,
        lim=lim,
        title="Energy, Intensity and Angular Error",
        deg_scale=1.0,
        ax=axs[0,0]
    )

    # Plot 2 - Energy, Radial, Transverse and Width Angle
    plot_polar_energy_intensity(
        azimuth,
        energy,
        radial_i,
        transverse_i,
        width_rad,
        "Angular Width",
        "Purples",
        vmax=math.pi / 6,
        lim=lim,
        title="Energy, Intensity and Width",
        deg_scale=3.0,
        ax=axs[1,0]
    )

    # Plot 3 - Energy dB
    ax_db = axs[0, 1]
    ax_db.plot(azimuth, energy_db)
    ax_db.set_theta_zero_location("N")
    ax_db.set_ylim(-24, 6)
    ax_db.set_title("Energy (dB)")
    axs[1,1].axis("off")
    plt.tight_layout()

    # Save plots
    if save_results and isinstance(results_file_name, str):
        save_plot(plt, results_file_name, "plot_energy_intensity_2D.png") # type: ignore[arg-type]


def plot_ei_3D(
    energy: ArrayLike,
    radial_i: ArrayLike,
    transverse_i: ArrayLike,
    cloud_points: MyCoordinates,
    save_results: bool,
    results_file_name=False,
) -> None:
    """
    Function to plot the energy and intensity when
    decoding from an input format to an output layout.
    3D projection on 2D plots.

    Args:
        energy (jax.numpy Array): contains the energy values for each virtual source (1xL)
        radial_i (jax.numpy array): contains the radial intensity values for each virtual
                source (1xL)
        transverse_i (jax.numpy array): contains the transverse intensity values for each virtual
                source (1xL)
        cloud(MyCoordinates): set of points sampling the sphere (L)
        save_results (bool): Flag to save plots
        results_file_name(str): Path where to save the plots
    """
    # Prep
    points = cloud_points.sph_deg()
    energy, radial_i, transverse_i = map(np.asarray, (energy, radial_i, transverse_i))

    x = points[:, 0]
    y = points[:, 1]
    data_crs = ccrs.PlateCarree()
    proj = ccrs.Robinson(central_longitude=0)
    fig = plt.figure(figsize=(17, 9))

    # Energy
    ax1 = fig.add_subplot(321, projection=proj)
    cartopy_scatter(x, y, data_crs, ax1, energy, "Energy", "Energy", (0, 2))

    # Radial Intensity
    ax2 = fig.add_subplot(323, projection=proj)
    cartopy_scatter(x, y, data_crs, ax2, radial_i, "Radial Intensity", "Radial Intensity", (0, 1))
    
    # Transverse Intensity
    ax3 = fig.add_subplot(325, projection=proj)
    cartopy_scatter(x, y, data_crs, ax3, transverse_i, "Transverse Intensity", "Transverse Intensity", (0,1))

    # Source Width
    width_deg = width_angle(radial_i)
    ax5 = fig.add_subplot(324, projection=proj)
    cartopy_scatter(x, y, data_crs, ax5, width_deg, "Source Width", "Source Width (Degrees)", (0, 45))

    # Angular Error
    ang_err = angular_error(radial_i, transverse_i)
    ax4 = fig.add_subplot(326, projection=proj)
    cartopy_scatter(x, y, data_crs, ax4, ang_err, "Angular Error", "Angular Error (Degrees)", (0, 45))

    plt.tight_layout()
    
    if save_results and isinstance(results_file_name, str):
        file_name = "plot_energy_intensity_3D.png"
        save_plot(plt, results_file_name, file_name) # type: ignore