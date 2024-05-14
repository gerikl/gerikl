from math import log

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


def _make_diffs(inters):
    return np.array(list(map(lambda x: inters[x] - inters[x - 1], range(1, len(inters)))))


def _calc_sdnn(intervals):
    ints = np.array(intervals)
    return np.std(ints)


def _calc_rmssd(intervals):
    ints = np.array(intervals)
    diffs = _make_diffs(ints) ** 2
    return np.sqrt(np.mean(diffs))


def _calc_nn50(intervals):
    ints = np.array(intervals)
    diffs = np.abs(_make_diffs(ints))
    return np.sum(diffs > 0.05)


def _calc_pnn50(intervals):
    return _calc_nn50(intervals) / (len(intervals) - 1)


def _calc_cv(intervals):
    ints = np.array(intervals)
    return _calc_sdnn(intervals) / np.mean(ints) * 100


def _calc_d(intervals):
    ints = np.array(intervals)
    return _calc_rmssd(ints) ** 2


def _calc_as(intervals):
    ints = np.array(intervals)
    return np.mean((ints - np.mean(ints)) ** 3)


def _calc_ex(intervals):
    ints = np.array(intervals)
    return np.mean((ints - np.mean(ints)) ** 4)


def _make_main_gist(ints, verbose=False):
    n, bins, rr = plt.hist(ints, bins=28, range=(0.3, 1.7))
    if verbose:
        plt.gca().set_yticklabels(['{:.0f}%'.format(x / sum(n) * 100) for x in plt.gca().get_yticks()])
    else:
        plt.gcf().set_visible(not plt.gcf().get_visible())
    return n, bins, rr


def _make_help_gist(ints, nclass, verbose=False):
    n, bins, rr = plt.hist(ints, bins=nclass, range=(0.3, 1.7))
    if verbose:
        plt.gca().set_yticklabels(['{:.0f}%'.format(x / sum(n) * 100) for x in plt.gca().get_yticks()])
    else:
        plt.gcf().set_visible(not plt.gcf().get_visible())
    return n, bins, rr


def _make_dots(n):
    x = np.array(range(len(n))) * (1.4 / len(n)) + 0.3 + (1.4 / len(n)) / 2
    y = np.array(n) / sum(n)
    return x, y


def _make_interpolation(x, y, verbose=True):
    temp = interpolate.splrep(x, y, s=0)
    xnew = np.linspace(0.3, 1.7, 400)
    ynew = interpolate.splev(xnew, temp, der=0)
    if verbose:
        plt.plot(xnew, ynew)
    else:
        plt.gcf().set_visible(not plt.gcf().get_visible())
    return xnew, ynew


def _calc_amo(y):
    return max(y)


def _calc_mo(x, y):
    return x[np.argmax(y)]


def _calc_mxdmn(ints):
    ints = np.array(ints)
    return np.percentile(ints, 98) - np.percentile(ints, 2)


def _calc_in(ints):
    n, bins, rr = _make_main_gist(ints, verbose=False)
    nclass = _calc_nclass(n)
    n, bins, rr = _make_help_gist(ints, nclass, verbose=False)
    x, y = _make_dots(n)
    xnew, ynew = _make_interpolation(x, y, verbose=False)
    amo = _calc_amo(ynew)
    mo = _calc_mo(xnew, ynew)
    mxdmn = _calc_mxdmn(ints)
    plt.close()
    return amo / (2 * mo * mxdmn)


def _calc_nclass(n):
    return int(1 + 4.44 * log(sum(n)))


def _calc_amo_full(ints):
    n, bins, rr = _make_main_gist(ints, verbose=False)
    nclass = _calc_nclass(n)
    n, bins, rr = _make_help_gist(ints, nclass, verbose=False)
    x, y = _make_dots(n)
    xnew, ynew = _make_interpolation(x, y, verbose=False)
    plt.close()
    return _calc_amo(ynew)


def _calc_mo_full(ints):
    n, bins, rr = _make_main_gist(ints, verbose=False)
    nclass = _calc_nclass(n)
    n, bins, rr = _make_help_gist(ints, nclass, verbose=False)
    x, y = _make_dots(n)
    xnew, ynew = _make_interpolation(x, y, verbose=False)
    plt.close()
    return _calc_mo(xnew, ynew)


def _calc_heartbeat(ints):
    return len(ints)


def _calc_mean(ints):
    return np.mean(ints)


def _calc_max_sig(signals):
    return (np.max(signals) - np.mean(signals)) / (np.max(signals) - np.min(signals))


def _make_features(dataset):
    dataset["sdnn"] = dataset["ints"].apply(_calc_sdnn)
    dataset["rmssd"] = dataset["ints"].apply(_calc_rmssd)
    dataset["nn50"] = dataset["ints"].apply(_calc_nn50)
    dataset["pnn50"] = dataset["ints"].apply(_calc_pnn50)
    dataset["cv"] = dataset["ints"].apply(_calc_cv)
    dataset["d"] = dataset["ints"].apply(_calc_d)
    dataset["as"] = dataset["ints"].apply(_calc_as)
    dataset["ex"] = dataset["ints"].apply(_calc_ex)
    dataset["mxdmn"] = dataset["ints"].apply(_calc_mxdmn)
    dataset["in"] = dataset["ints"].apply(_calc_in)
    dataset["amo"] = dataset["ints"].apply(_calc_amo_full)
    dataset["mo"] = dataset["ints"].apply(_calc_mo_full)
    dataset["beat"] = dataset["ints"].apply(_calc_heartbeat)
    dataset["mean"] = dataset["ints"].apply(_calc_mean)
    dataset["max"] = dataset["sigs"].apply(_calc_max_sig)
    return dataset
