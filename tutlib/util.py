from itertools import product

import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def ternary_to_xy(comps, normalize=True):
    '''Ternary composition to Cartesian coordinate'''

    if not (comps.shape[1] == 3):
        raise ValueError('Must specify exactly three components')

    if normalize:
        comps = comps / comps.sum(1)[:, np.newaxis]

    # Convert ternary data to cartesian coordinates.
    xy = np.zeros((comps.shape[0], 2))
    xy[:, 1] = comps[:, 1] * np.sin(60. * np.pi / 180.)
    xy[:, 0] = comps[:, 0] + xy[:, 1] * np.sin(30. * np.pi / 180.) / np.sin(60 * np.pi / 180)
    return xy


def xy_to_ternary(xy, base=1.0):
    '''Ternary composition to Cartesian coordinate'''

    # Convert ternary data to cartesian coordinates.
    ternary = np.zeros((xy.shape[0], 3))
    ternary[:, 1] = xy[:, 1] / np.sin(60. * np.pi / 180.)
    ternary[:, 0] = xy[:, 0] - xy[:, 1] * np.sin(30. * np.pi / 180.) / np.sin(60 * np.pi / 180)
    ternary[:, 2] = 1.0 - ternary[:, 1] - ternary[:, 0]

    ternary *= base

    return ternary


def composition_grid_ternary(pts_per_row=50, basis=1.0, dim=3, eps=1e-9):
    pts = []
    for i in product(*[np.linspace(0, 1.0, pts_per_row)] * (dim - 1)):
        if sum(i) > (1.0 + eps):
            continue

        j = 1.0 - sum(i)

        if j < (0.0 - eps):
            continue
        pt = [k * basis for k in [*i, j]]
        pts.append(pt)
    return np.array(pts)


def make_ordinal_labels(labels):
    encoder = OrdinalEncoder()
    labels_ordinal = encoder.fit_transform(labels.values.reshape(-1, 1)).flatten()
    labels_ordinal = labels.copy(data=labels_ordinal)
    return labels_ordinal


from shapely import MultiPoint
from shapely import concave_hull


def trace_boundaries(
        dataset,
        hull_tracing_ratio=0.2,
        drop_phases=None,
        reset=True,
        component_attr='components',
        label_attr='labels',
        segementize_length=0.025
):
    if drop_phases is None:
        drop_phases = []

    hulls = {}

    label_variable = dataset.attrs[label_attr]
    for label, sds in dataset.groupby(label_variable):
        if label in drop_phases:
            continue
        comps = sds[sds.attrs[component_attr]].to_array('component').transpose(..., 'component')
        xy = ternary_to_xy(comps.values)
        mp = MultiPoint(xy)
        hull = concave_hull(mp, ratio=hull_tracing_ratio)
        hull = hull.segmentize(segementize_length)
        hulls[label] = hull

    return hulls


from sklearn.metrics import pairwise_distances_argmin_min


def calculate_perimeter_score_v1(ds, gt_xy, hull_tracing_ratio=0.2, component_attr='components', label_attr='labels'):
    hulls = trace_boundaries(ds, hull_tracing_ratio=hull_tracing_ratio, component_attr=component_attr,
                             label_attr=label_attr)

    if len(hulls) == 1:
        return np.nan, np.nan

    means = []
    stds = []
    for hull in hulls.values():
        try:
            xy = np.vstack(hull.boundary.xy).T
        except NotImplementedError:
            pass
        else:
            _, dist = pairwise_distances_argmin_min(gt_xy, xy, metric='euclidean')
            means.append(dist.mean())
            stds.append(dist.std())
    if len(means) > 0:
        idxmin = np.argmin(means)
        return means[idxmin], stds[idxmin]
    else:
        return np.nan, np.nan


def get_xy(hull):
    try:
        x, y = hull.boundary.xy
    except NotImplementedError:
        if hasattr(hull, 'geoms'):
            x = [];
            y = []
            for geom in list(hull.boundary.geoms):
                xx, yy = geom.xy
                x.extend(xx)
                y.extend(yy)
        else:
            x, y = None, None
    return x, y


def calculate_perimeter_score_v2(hull1, hull2):
    hull1_xy = np.vstack(get_xy(hull1)).T
    hull2_xy = np.vstack(get_xy(hull2)).T

    try:
        idx1, dist1 = pairwise_distances_argmin_min(hull2_xy, hull1_xy, metric='euclidean')
        idx2, dist2 = pairwise_distances_argmin_min(hull1_xy, hull2_xy, metric='euclidean')
    except:
        import pdb
        pdb.set_trace()

    # build index list of closest pairs
    idx1 = np.vstack([np.arange(idx1.shape[0]), idx1]).T
    idx2 = np.vstack([idx2, np.arange(idx2.shape[0])]).T

    dist = np.hstack([dist1, dist2])
    idx = np.vstack([idx1, idx2])

    # remove duplicate pairs from idx and dist
    unique = np.unique(idx, axis=0, return_index=True)[1]
    idx = idx[unique]
    dist = dist[unique]

    # build list of distance pair coordinates for plotting
    coord = []
    for i, j in idx:
        coord.extend([hull1_xy[j], hull2_xy[i], [np.nan, np.nan]])
    coord = np.array(coord)

    out = {
        'mean': dist.mean(),
        'std': dist.std(),
        'pair_dist': dist,
        'pair_idx': idx,
        'pair_coord': coord,
        'hull1_xy': hull1_xy,
        'hull2_xy': hull2_xy,
    }

    return out
