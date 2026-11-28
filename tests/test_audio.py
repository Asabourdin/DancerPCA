"""Tests for the audio offset"""

import numpy as np

from rpca_dance.audio import amplitude_envelope, audio_offset_frames, audio_peak_check, envelope_offset


def test_envelope_is_the_mean_loudness_per_block():
    samples = np.array([0.5, -0.5, 0.0, 0.0, -1.0, 1.0, 0.2])  # the last sample does not fill a block
    assert amplitude_envelope(samples, 2, 1).tolist() == [0.5, 0.0, 1.0]


def test_envelope_offset_finds_a_shift():
    random_generator = np.random.default_rng(0)
    reference = random_generator.random(600)
    learner = reference[37:537]  # the learner starts 37 values later in the song
    best_lag, _, correlations = envelope_offset(reference, learner, 50, 300)
    assert best_lag == 37
    assert np.isclose(correlations.max(), 1.0)


def test_lag_in_frames():
    assert audio_offset_frames(62, 100, 8) == 5  # 0.62 s at 8 frames per second is 4.96 frames
    assert audio_offset_frames(-30, 100, 8) == -2


def test_audio_peak_check_rejects_weak_and_edge_peaks():
    lags = np.arange(-50, 51)
    clear = np.zeros(101)
    clear[60] = 1.0  # lag 10
    clear[20] = 0.5
    assert audio_peak_check(lags, clear, 10, 10, 1.3)["reliable"]
    weak = clear.copy()
    weak[20] = 0.9  # next peak too close in height
    assert not audio_peak_check(lags, weak, 10, 10, 1.3)["reliable"]
    edge = np.full(101, 0.1)
    edge[98] = 1.0  # lag 48, at the edge
    check = audio_peak_check(lags, edge, 10, 10, 1.3)
    assert check["at edge"] and not check["reliable"]
