"""Run the qualitative invariants of the reduced bioenergetic model under pytest."""

import test_sanity as s


def test_healthy_vs_disease():
    s.healthy_vs_disease()


def test_monotonic_in_frataxin():
    s.monotonic_in_frataxin()


def test_positivity_and_stability():
    s.positivity_and_stability()


def test_nrf2_inducer_rescues():
    s.nrf2_inducer_rescues()


def test_nad_precursor_improves_throughput():
    s.nad_precursor_improves_throughput()


def test_substrate_alone_worsens_redox():
    s.substrate_alone_worsens_redox()


def test_iron_loading_worsens():
    s.iron_loading_worsens()
