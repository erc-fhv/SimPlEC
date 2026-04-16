"""Tests for Simulation.fresh_copy() — running the same scenario over
multiple periods with freshly initialised models."""

import pandas as pd
import numpy as np
from simplec import Simulation


# ---------------------------------------------------------------------------
# Minimal stateful models
# ---------------------------------------------------------------------------

class ConstantSource:
    inputs = []
    outputs = ['trigger']
    delta_t = 60

    def __init__(self, name):
        self.name = name

    def step(self, time):
        return {'trigger': 1}


class StatefulCounter:
    """Increments an internal counter on every step — state that must be
    reset between independent simulation runs."""
    inputs = ['trigger']
    outputs = ['count']
    delta_t = 60

    def __init__(self, name):
        self.name = name
        self.count = 0

    def step(self, time, trigger):
        self.count += 1
        return {'count': self.count}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PERIOD_1 = pd.date_range('2021-01-01', periods=3, freq='1min', tz='UTC')
PERIOD_2 = pd.date_range('2021-06-01', periods=3, freq='1min', tz='UTC')


def _build_sim():
    """Set up a simulation with source → counter, watching 'count'."""
    sim = Simulation()
    source = ConstantSource('source')
    counter = StatefulCounter('counter')
    sim.add_model(source)
    sim.add_model(counter, watch_values=['count'])
    sim.connect(source, counter, ('trigger', 'trigger'))
    return sim, counter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fresh_copy_resets_state():
    """Each period must start with a fresh model (count begins at 1)."""
    sim, _ = _build_sim()

    results = []
    for period in [PERIOD_1, PERIOD_2]:
        sim_copy = sim.fresh_copy()
        sim_copy.simulate(period, enable_progress_bar=False)
        results.append(sim_copy.df)

    for df in results:
        counts = df[('counter', 'outputs', 'count')].values
        np.testing.assert_array_equal(counts, [1, 2, 3])


def test_fresh_copy_matches_individual_setup():
    """Results from fresh_copy loop must match separately constructed sims."""
    # --- via fresh_copy ---
    sim, _ = _build_sim()
    copy_results = []
    for period in [PERIOD_1, PERIOD_2]:
        sim_copy = sim.fresh_copy()
        sim_copy.simulate(period, enable_progress_bar=False)
        copy_results.append(sim_copy.df)

    # --- via independent construction ---
    indep_results = []
    for period in [PERIOD_1, PERIOD_2]:
        sim2, _ = _build_sim()
        sim2.simulate(period, enable_progress_bar=False)
        indep_results.append(sim2.df)

    for df_copy, df_indep in zip(copy_results, indep_results):
        pd.testing.assert_frame_equal(df_copy, df_indep)


def test_fresh_copy_does_not_mutate_original():
    """Running simulate on a copy must not change the original sim."""
    sim, counter = _build_sim()

    sim_copy = sim.fresh_copy()
    sim_copy.simulate(PERIOD_1, enable_progress_bar=False)

    # Original counter must still be at 0
    assert counter.count == 0
    # Original sim should not have a populated df
    assert not hasattr(sim, 'array') or not hasattr(sim.array, 'shape')
