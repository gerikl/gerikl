import typing as tp
import warnings

import mne
import numpy as np
import pandas as pd
from ecgdetectors import Detectors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from wfdb import processing

from errors import AbnormalECGError
from feature_calcs import _make_features


def _normalize(signals):
    signals = np.array(signals)
    diap = np.max(signals) - np.min(signals)
    return signals * (1.2 / diap)


def _read_edf(path):
    data = mne.io.read_raw_edf(path)
    raw_data = data.get_data()[0]
    raw_data = _normalize(raw_data)
    fs = data.info["sfreq"]
    return raw_data, fs


def _calc_qrs_inds(signals, fs):
    signals = processing.normalize_bound(signals)
    xqrs = processing.XQRS(signals, fs=fs)
    xqrs.detect(verbose=False)
    if len(xqrs.qrs_inds) < 3:
        detectors = Detectors(fs)
        rs = detectors.engzee_detector(signals)
        if len(rs) < 2:
            raise AbnormalECGError("Detectors did not find R-peaks")
        return detectors.engzee_detector(signals)
    else:
        return xqrs.qrs_inds


def _make_card_int(r_ids, fs):
    return np.array(list(map(lambda x: r_ids[x] - r_ids[x - 1], range(1, len(r_ids))))) / fs


def _make_data_sigs(signals, fs):
    signals = _normalize(signals)
    r_ids = _calc_qrs_inds(np.array(signals), fs)
    ints = _make_card_int(r_ids, fs)
    data = pd.DataFrame(columns=["sigs", "ints", "fs"])
    data.loc[0] = [signals, ints, fs]
    data = _make_features(data).drop(columns=["fs", "sigs", "ints"])
    return data


def _predict_5_sigs(signals, fs, model):
    data = _make_data_sigs(signals, fs)
    return model.predict(data)[0]


def _make_long_ecg(sigs, fs):
    needed_len = 5 * 60 * fs
    new_sigs = [0] * needed_len
    new_sigs[:len(sigs)] = sigs
    new_sigs[len(sigs):] = sigs[:needed_len - len(sigs)]
    return new_sigs


def predict_edf(path: str, model) -> bool:
    """
        Predict arrhythmia by .edf file with ECG. Expected duration of ECG is 5 minutes.
        If duration is longer, then samples intervals_count 5-minute intervals from signal.
        If duration is shorter, pad the ending of the recording by recording's beginning.
    """
    sigs, fs = _read_edf(path)
    return predict_sigs(sigs, fs, model)


def predict_sigs(sigs: list[float],
                 fs: int,
                 model: tp.Union[LogisticRegression, RandomForestClassifier],
                 intervals_count: int = 3) -> bool:
    """
    Predict arrhythmia by ECG signal and frequency. Expected duration of ECG is 5 minutes.
    If duration is longer, then samples intervals_count 5-minute intervals from signal.
    If duration is shorter, pad the ending of the recording by recording's beginning.
    """
    if len(sigs) < fs * 5 * 30 * 0.997:
        raise AbnormalECGError("Recording is too short for analysis (less than 2.5 minutes)")
    elif len(sigs) < fs * 5 * 60 * 0.997:
        warnings.warn('The ECG is shorter than it should be, the model may not work well')
        sigs = _make_long_ecg(sigs, fs)
        return bool(_predict_5_sigs(sigs, fs, model))
    elif len(sigs) > fs * 5 * 60 * 1.003:
        warnings.warn('The ECG is longer than it should be, the model may not work well')
        needed_len = fs * 5 * 60
        np.random.seed(123)
        starts = np.random.choice(range(len(sigs) - needed_len), intervals_count)
        for start in starts:
            if _predict_5_sigs(sigs[start:start + needed_len], fs, model) == 1:
                return True
        return False
    else:
        return bool(_predict_5_sigs(sigs, fs, model))
