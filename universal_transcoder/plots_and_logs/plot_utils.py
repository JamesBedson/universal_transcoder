"""
Copyright (c) 2025 James Bedson

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

"""
Common plotting utilities for USAT visualizations.

Includes:
- Cartopy map configuration
- Polar and azimuthal plotting helpers
- Scaling and formatting utilities for energy/intensity and pressure/velocity plots
"""

from typing import Any, cast
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.mpl.geoaxes as cgeo
from matplotlib.ticker import FuncFormatter


def setup_cartopy_axes(ax: plt.Axes) -> cgeo.GeoAxes:
    """
    Configures gridlines and label formatting for Cartopy GeoAxes.
    """
    ax = cast(cgeo.GeoAxes, ax)

    gl = ax.gridlines(
        draw_labels=True,
        xlocs=np.arange(-180, 182, 60),
        ylocs=np.arange(-90, 91, 45),
        linestyle="--",
        linewidth=0.5,
        color="black",
        alpha=0.5,
    )

    gl.xformatter = FuncFormatter(lambda x, _: f"{int(x):+d}°")
    gl.yformatter = FuncFormatter(lambda y, _: f"{int(y):+d}°")
    gl.top_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = True

    gl.xlabel_style = {"size": 8, "rotation": 0}
    gl.ylabel_style = {"size": 8, "rotation": 0}
    gl.xpadding = 2
    return ax


def cartopy_scatter(
    x: ArrayLike,
    y: ArrayLike,
    data_crs: ccrs.Projection,
    ax: plt.Axes,
    data: ArrayLike,
    title: str,
    label: str,
    clim: tuple[float, float],
) -> plt.Axes:
    
    """
    Utility function to plot a 2D scatter on Cartopy projection.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    data = np.asarray(data)
    ax = setup_cartopy_axes(ax)

    sc = ax.scatter(
        x,
        y,
        s=60 - 0.6 * np.abs(y),
        c=data,
        cmap="coolwarm",
        alpha=0.8,
        edgecolors="none",
        transform=data_crs,
    )

    sc.set_clim(*clim)
    cbar = plt.colorbar(sc, ax=ax, label=label)
    ax.set_title(title)
    return ax


def close_loop(arr: np.ndarray) -> np.ndarray:
    """
    Ensures that circular data close properly by repeating the first value.
    """
    return np.hstack((arr, arr[0]))


def compute_plot_limit(*arrays: np.ndarray) -> float:
    """
    Determines a suitable plot limit based on the maxima of several arrays.
    Ensures small headroom above 1.0 for normalized data.
    """
    maxima = np.max([np.max(a) for a in arrays])
    if maxima < 0.9:
        return 1.0
    elif np.isclose(maxima, 1.0, rtol=1e-9, atol=1e-9):
        return 1.1
    else:
        return float(maxima + 0.1)