import numpy as np
import matplotlib.pyplot as plt


def map_hist(x, y, h, bins):
    xi = np.digitize(x, bins[0]) - 1
    yi = np.digitize(y, bins[1]) - 1
    inds = np.ravel_multi_index((xi, yi),
                                (len(bins[0]) - 1, len(bins[1]) - 1),
                                mode='clip')
    vals = h.flatten()[inds]
    bads = ((x < bins[0][0]) | (x > bins[0][-1]) |
            (y < bins[1][0]) | (y > bins[1][-1]))
    vals[bads] = np.NaN
    return vals


def scatter_hist2d(x, y,
                   s=20, marker=u'o',
                   mode='mountain',
                   bins=10, range=None,
                   normed=False, weights=None,  # np.histogram2d args
                   ax=None, dens_func=None,
                   **kwargs):
    """
    Make a scattered-histogram plot.

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data

    s : scalar or array_like, shape (n, ), optional, default: 20
        size in points^2.

    marker : `~matplotlib.markers.MarkerStyle`, optional, default: 'o'
        See `~matplotlib.markers` for more information on the different
        styles of markers scatter supports. `marker` can be either
        an instance of the class or the text shorthand for a particular
        marker.

    mode: [None | 'mountain' | 'valley' | 'clip']
       Possible values are:

       - None : The points are plotted as one scatter object, in the
         order in-which they are specified at input.

       - 'mountain' : The points are sorted/plotted in the order of
         the number of points in their 'bin'. This means that points
         in the highest density will be plotted on-top of others. This
         cleans-up the edges a bit, the points near the edges will
         overlap.

       - 'valley' : The reverse order of 'mountain'. The low density
         bins are plotted on top of the high ones.

    bins : int or array_like or [int, int] or [array, array], optional
        The bin specification:

          * If int, the number of bins for the two dimensions (nx=ny=bins).
          * If array_like, the bin edges for the two dimensions
            (x_edges=y_edges=bins).
          * If [int, int], the number of bins in each dimension
            (nx, ny = bins).
          * If [array, array], the bin edges in each dimension
            (x_edges, y_edges = bins).
          * A combination [int, array] or [array, int], where int
            is the number of bins and array is the bin edges.

    range : array_like, shape(2,2), optional
        The leftmost and rightmost edges of the bins along each dimension
        (if not specified explicitly in the `bins` parameters):
        ``[[xmin, xmax], [ymin, ymax]]``. All values outside of this range
        will be considered outliers and not tallied in the histogram.

    normed : bool, optional
        If False, returns the number of samples in each bin. If True,
        returns the bin density ``bin_count / sample_count / bin_area``.

    weights : array_like, shape(N,), optional
        An array of values ``w_i`` weighing each sample ``(x_i, y_i)``.
        Weights are normalized to 1 if `normed` is True. If `normed` is
        False, the values of the returned histogram are equal to the sum of
        the weights belonging to the samples falling into each bin.

    ax : an axes instance to plot into.

    dens_func : function or callable (default: None)
        A function that modifies (inputs and returns) the dens
        values (e.g., np.log10). The default is to not modify the
        values.

    kwargs : these are all passed on to scatter.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`
        The scatter instance.
    """
    if ax is None:
        ax = plt.gca()

    
    ax.set_facecolor('xkcd:salmon')
    h, xe, ye = np.histogram2d(x, y, bins=bins,
                               range=range, density=normed,
                               weights=weights)
    # bins = (xe, ye)
    dens = map_hist(x, y, h, bins=(xe, ye))
    if dens_func is not None:
        dens = dens_func(dens)
    iorder = slice(None)  # No ordering by default
    if mode == 'mountain':
        iorder = np.argsort(dens)
    elif mode == 'valley':
        iorder = np.argsort(dens)[::-1]
    x = x[iorder]
    y = y[iorder]
    dens = dens[iorder]
    return ax.scatter(x, y,
                      s=s, c=dens,
                      marker=marker,
                      **kwargs)


def scatter_hexbin(
        x, y, s=20, C=None, marker='o',
        gridsize=100, mode='mountain',
        xscale='linear', yscale='linear', extent=None,
        cmap=None, norm=None, vmin=None, vmax=None,
        reduce_C_function=np.mean,
        ax=None,
        **kwargs):
    """
    Make a scatter plot where points are grouped into 'hexagonal bins'.
    If *C* is *None*, the color-value of the points in each bin is
    determined by the number of points in the hexagon (i.e., a
    hexagonal histogram). Otherwise, *C* specifies values at the
    coordinate (x[i], y[i]). For each hexagon, these values are
    reduced using *reduce_C_function*. The 'color-values' are then
    mapped to colors using the specified colormap.

    Parameters
    ----------

    x, y : array-like
        The data positions. *x* and *y* must be of the same length.

    s : array-like, default: 20
        The marker size in points**2. Default is 20.

    C : array-like, optional
        If given, these values are accumulated in the bins. Otherwise,
        every point has a value of 1. Must be of the same length as *x*
        and *y*.

    marker : MarkerStyle, default: 'o'
        The marker style. This is passed directly to scatter.

    gridsize : int or (int, int), default: 100
        If a single int, the number of hexagons in the *x*-direction.
        The number of hexagons in the *y*-direction is chosen such that
        the hexagons are approximately regular.
        Alternatively, if a tuple (*nx*, *ny*), the number of hexagons
        in the *x*-direction and the *y*-direction.

    mode: [None | 'mountain' | 'valley']
       Possible values are:

       - None : The points are plotted as one scatter object, in the
         order in-which they are specified at input. This leaves the
         edges of hexbins somewhat chaotic.
       - 'mountain' : The points are sorted/plotted in the order of
         the number of points in their 'bin'. This means that points
         in the highest density will be plotted on-top of others. This
         cleans-up the edges a bit, the points near the edges will
         overlap.
       - 'valley' : The reverse order of 'mountain'. The low density
         bins are plotted on top of the high ones.

    xscale : {'linear', 'log'}, default: 'linear'
        Use a linear or log10 scale on the horizontal axis.

    yscale : {'linear', 'log'}, default: 'linear'
        Use a linear or log10 scale on the vertical axis.

    extent : 4-tuple of float, default: *None*
        The limits of the bins (xmin, xmax, ymin, ymax).
        The default assigns the limits based on
        *gridsize*, *x*, *y*, *xscale* and *yscale*.
        If *xscale* or *yscale* is set to 'log', the limits are
        expected to be the exponent for a power of 10. E.g. for
        x-limits of 1 and 50 in 'linear' scale and y-limits
        of 10 and 1000 in 'log' scale, enter (1, 50, 1, 3).

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`
        The scatter instance.

    Other Parameters
    ----------------
    cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
        The Colormap instance or registered colormap name used to map
        the bin values to colors.
    norm : `~matplotlib.colors.Normalize`, optional
        The Normalize instance scales the bin values to the canonical
        colormap range [0, 1] for mapping to colors. By default, the data
        range is mapped to the colorbar range using linear scaling.
    vmin, vmax : float, default: None
        The colorbar range. If *None*, suitable min/max values are
        automatically chosen by the `.Normalize` instance (defaults to
        the respective min/max values of the bins in case of the default
        linear scaling).
        It is an error to use *vmin*/*vmax* when *norm* is given.
    reduce_C_function : callable, default: `numpy.mean`
        The function to aggregate *C* within the bins. It is ignored if
        *C* is not given. This must have the signature::
            def reduce_C_function(C: array) -> float
        Commonly used functions are:
        - `numpy.mean`: average of the points
        - `numpy.sum`: integral of the point values
        - `numpy.amax`: value taken from the largest point

    **kwargs : These are all passed on to scatter

    Notes
    -----
    For best results, you typically have to make the marker size
    (``s``) much smaller than the hexbin size. Setting
    edgecolor='none' can also be helpful to reduce markersize.

    See Also
    --------
    scatter : a scatter plot.
    hexbin : 2D hexagonal binning plot of points x, y.
    hist2d : A 2D histogram
    scatter_hist2d : a 2D 'scatter histogram' with rectangular bins.
    """

    if ax is None:
        ax = plt.gca()

    # Much of this was copied <matplotlib>/lib/matplotlib/axes/_axes.py

    # Set the size of the hexagon grid
    if np.iterable(gridsize):
        nx, ny = gridsize
    else:
        nx = gridsize
        ny = int(nx / np.sqrt(3))

    # Count the number of data in each hexagon
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # Will be log()'d if necessary, and then rescaled.
    tx = x
    ty = y

    if xscale == 'log':
        if np.any(x <= 0.0):
            raise ValueError("x contains non-positive values, so can not "
                             "be log-scaled")
        tx = np.log10(tx)
    if yscale == 'log':
        if np.any(y <= 0.0):
            raise ValueError("y contains non-positive values, so can not "
                             "be log-scaled")
        ty = np.log10(ty)
    if extent is not None:
        xmin, xmax, ymin, ymax = extent
    else:
        xmin, xmax = (tx.min(), tx.max()) if len(x) else (0, 1)
        ymin, ymax = (ty.min(), ty.max()) if len(y) else (0, 1)

    #####
    # Start hexagon magic
    nx1 = nx + 1
    ny1 = ny + 1
    nx2 = nx
    ny2 = ny
    n = nx1 * ny1 + nx2 * ny2

    # In the x-direction, the hexagons exactly cover the region from
    # xmin to xmax. Need some padding to avoid roundoff errors.
    padding = 1.e-9 * (xmax - xmin)
    xmin -= padding
    xmax += padding
    sx = (xmax - xmin) / nx
    sy = (ymax - ymin) / ny
    # Positions in hexagon index coordinates.
    ix = (tx - xmin) / sx
    iy = (ty - ymin) / sy
    ix1 = np.round(ix).astype(int)
    iy1 = np.round(iy).astype(int)
    ix2 = np.floor(ix).astype(int)
    iy2 = np.floor(iy).astype(int)
    # flat indices, plus one so that out-of-range points go to position 0.
    i1 = np.where((0 <= ix1) & (ix1 < nx1) & (0 <= iy1) & (iy1 < ny1),
                  ix1 * ny1 + iy1 + 1, 0)
    i2 = np.where((0 <= ix2) & (ix2 < nx2) & (0 <= iy2) & (iy2 < ny2),
                  ix2 * ny2 + iy2 + 1, 0)


    d1 = (ix - ix1) ** 2 + 3.0 * (iy - iy1) ** 2
    d2 = (ix - ix2 - 0.5) ** 2 + 3.0 * (iy - iy2 - 0.5) ** 2
    # Which points are in the 'first set' vs. 'second set'?
    bdist = (d1 < d2)

    # This is the index of the hexagon 'number'
    ihex = i2 + nx1 * ny1 - 1
    ihex[bdist] = i1[bdist] - 1

    # End hexagon magic
    #####
    
    if C is None:  # [1:] drops out-of-range points.
        counts1 = np.bincount(i1[bdist], minlength=1 + nx1 * ny1)[1:]
        counts2 = np.bincount(i2[~bdist], minlength=1 + nx2 * ny2)[1:]
        accum = np.concatenate([counts1, counts2]).astype(float)
        C = np.ones(len(x))
    else:
        # store the C values in a list per hexagon index
        Cs_at_i1 = [[] for _ in range(1 + nx1 * ny1)]
        Cs_at_i2 = [[] for _ in range(1 + nx2 * ny2)]
        for i in range(len(x)):
            if bdist[i]:
                Cs_at_i1[i1[i]].append(C[i])
            else:
                Cs_at_i2[i2[i]].append(C[i])
        accum = np.array(
            [reduce_C_function(acc) if len(acc) > mincnt else np.nan
             for Cs_at_i in [Cs_at_i1, Cs_at_i2]
             for acc in Cs_at_i[1:]],  # [1:] drops out-of-range points.
            float)
    good_idxs = ~np.isnan(accum)

    offsets = np.zeros((n, 2), float)
    offsets[:nx1 * ny1, 0] = np.repeat(np.arange(nx1), ny1)
    offsets[:nx1 * ny1, 1] = np.tile(np.arange(ny1), nx1)
    offsets[nx1 * ny1:, 0] = np.repeat(np.arange(nx2) + 0.5, ny2)
    offsets[nx1 * ny1:, 1] = np.tile(np.arange(ny2), nx2) + 0.5
    offsets[:, 0] *= sx
    offsets[:, 1] *= sy
    offsets[:, 0] += xmin
    offsets[:, 1] += ymin
    # remove accumulation bins with no data
    offsets = offsets[good_idxs, :]
    accum = accum[good_idxs]
    
    cvals = accum[ihex]

    iorder = slice(None)  # No ordering by default
    if mode == 'mountain':
        iorder = np.argsort(cvals)
    elif mode == 'valley':
        iorder = np.argsort(cvals)[::-1]

    x = x[iorder]
    y = y[iorder]
    cvals = cvals[iorder]
    return ax.scatter(x, y,
                      s=s, c=cvals, marker=marker,
                      cmap=None, norm=None, vmin=None, vmax=None,
                      **kwargs)


if __name__ == '__main__':

    import matplotlib as mpl
    mpl.use('TkAgg')
    plt.ion()

    randgen = np.random.RandomState(84309242)
    npoint = 10000
    x = randgen.randn(npoint)
    y = 2 * randgen.randn(npoint) + x

    extent = [-10, 10, -10, 10]
    
    bins = np.linspace(extent[0], extent[1], 50)
    hexbin_gridsize = [50, 20]
    
    fig, axs = plt.subplots(3, 2, figsize=[6, 8],
                            gridspec_kw=dict(hspace=0.3),
                            sharex=True, sharey=True)

    ax = axs[0, 0]
    ax.plot(x, y, '.', color='b', )
    ax.set_title("Traditional Scatterplot")

    ax = axs[1, 0]
    ax.hist2d(x, y, bins=[bins, bins])
    ax.set_title("Traditional 2-D Histogram")

    ax = axs[2, 0]
    scatter_hist2d(x, y, bins=[ bins, bins], ax=ax, s=5, edgecolor='none')
    ax.set_title("Scatter histogram combined!")

    axs[0,1].set_visible(False)
    
    ax = axs[1, 1]
    ax.hexbin(x, y, gridsize=hexbin_gridsize, extent=extent, linewidths=0.2)
    ax.set_title("A 'hexbin' histogram")

    ax = axs[2, 1]
    scatter_hexbin(x, y, gridsize=hexbin_gridsize, extent=extent, ax=ax, s=5, edgecolor='none')
    ax.set_title("Scatter hexbin combined!")

    ax.set_xlim([-10,10])
    ax.set_ylim([-5,5])

    fig.text(0.5, 0.02, "Note: The hexbin (right) doesn't have as much 'striping'\n"
             "as a rectangular 2-D histogram (left).", ha='center', va='bottom', size='large')
    
    fig.savefig('ScatterHist_Example.png', dpi=200)
