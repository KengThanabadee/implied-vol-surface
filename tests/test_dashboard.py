import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from iv_surface import fetcher, surface_builder, visualization


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("The dashboard must not use the network")

    monkeypatch.setattr(fetcher.requests.sessions.Session, "request", fail)


def test_dashboard_renders_grid_and_recalculates_rate(monkeypatch):
    results = []
    build = surface_builder.build_combined_surface_from_chain

    def record(chain, r=0):
        result = build(chain, r=r)
        results.append((r, result))
        return result

    monkeypatch.setattr(surface_builder, "build_combined_surface_from_chain", record)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert not app.error
    assert not app.selectbox
    assert results[-1][0] == 0
    result = results[-1][1]
    counts = [int(metric.value.replace(",", "")) for metric in app.metric]
    assert sum(counts) == result.iv_surface.size
    assert counts[2] == np.isnan(result.iv_surface).sum()
    assert len(app.get("plotly_chart")) == 3
    assert app.sidebar.radio[0].value == "Core (0.8–1.2)"
    assert app.main.radio[0].value == "Surface"
    assert [tab.label for tab in app.tabs] == ["Heatmap", "3D surface", "IV smiles"]
    assert any(
        "211 observed · 163 interpolated · 26 blank" in caption.value
        for caption in app.caption
    )
    assert any("Frozen snapshot" in caption.value for caption in app.caption)

    app.main.radio[0].set_value("Observed").run()
    assert not app.exception
    assert app.main.radio[0].value == "Observed"
    assert any(
        "211 observed · 0 interpolated · 189 blank" in caption.value
        for caption in app.caption
    )

    app.sidebar.radio[0].set_value("Extended (0.7–1.3)").run()
    assert not app.exception
    assert any(
        "253 observed · 231 interpolated · 56 blank" in caption.value
        for caption in app.caption
    )

    app.number_input[0].set_value(0.25).run()
    assert not app.exception
    assert results[-1][0] == 0.0025
    assert not np.allclose(
        results[0][1].iv_surface, results[-1][1].iv_surface, equal_nan=True,
    )


def test_smile_selection_preserves_observations_and_other_views(monkeypatch):
    figures = {}
    for name in ["make_iv_heatmap", "make_iv_surface_figure", "make_smile_figure"]:
        build = getattr(visualization, name)

        def record(*args, _name=name, _build=build, **kwargs):
            figure = _build(*args, **kwargs)
            figures[_name] = figure
            return figure

        monkeypatch.setattr(visualization, name, record)

    snapshot = fetcher.load_chain_snapshot(APP_PATH.parent / "data/bybit_btc_options_snapshot.json")
    chain = fetcher.parse_chain_snapshot(snapshot)
    result = surface_builder.build_combined_surface_from_chain(chain)
    ordered_expiries = sorted(result.expiries)
    dates = chain.drop_duplicates("tau").set_index("tau")["expiry_dt"]
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert not app.multiselect
    assert [box.label for box in app.checkbox] == [
        dates[tau].strftime("%Y-%m-%d") for tau in ordered_expiries
    ]
    expected_defaults = [
        tau for tau in ordered_expiries
        if np.isfinite(result.iv_surface[result.expiries.index(tau)]).any()
    ][:3]
    assert [tau for tau, box in zip(ordered_expiries, app.checkbox) if box.value] == expected_defaults
    other_views = {
        name: figures[name].to_json()
        for name in ["make_iv_heatmap", "make_iv_surface_figure"]
    }

    selected = ordered_expiries[-1]
    for tau, box in zip(ordered_expiries, app.checkbox):
        box.set_value(tau == selected)
    app.run()
    assert not app.exception
    smile = figures["make_smile_figure"]
    assert len(smile.data) == 1
    trace = smile.data[0]
    row = result.expiries.index(selected)
    order = np.argsort(result.strikes)
    np.testing.assert_allclose(trace.y, result.iv_surface[row, order], equal_nan=True)
    np.testing.assert_allclose(trace.x, np.asarray(result.strikes)[order] / result.spot_price)
    assert np.isnan(trace.y).any()
    assert trace.connectgaps is False
    assert trace.name == dates[selected].strftime("%Y-%m-%d")
    np.testing.assert_array_equal(
        np.asarray(trace.customdata)[:, 1], result.source_grid[row, order]
    )
    for field in ["strike", "K / S", "IV", "source", "fullData.name"]:
        assert field in trace.hovertemplate
    for name, before in other_views.items():
        assert figures[name].to_json() == before

    app.sidebar.radio[0].set_value("Extended (0.7–1.3)").run()
    assert not app.exception
    assert figures["make_smile_figure"].to_json() == smile.to_json()
    assert [tau for tau, box in zip(ordered_expiries, app.checkbox) if box.value] == [selected]

    for box in app.checkbox:
        box.uncheck()
    app.run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 2
    assert any("Select an expiry" in info.value for info in app.info)


@pytest.mark.parametrize("finite_count", [0, 1, 2, 4])
def test_smile_defaults_skip_expiries_without_finite_iv(monkeypatch, finite_count):
    build = surface_builder.build_combined_surface_from_chain
    expected = []
    ordered_expiries = []

    def sparse_result(chain, r=0):
        result = build(chain, r=r)
        ordered_expiries[:] = sorted(result.expiries)
        # Keep later expiries usable to catch default selection of blank early rows.
        chosen = sorted(result.expiries)[-finite_count:] if finite_count else []
        expected[:] = chosen[:3]
        for index, tau in enumerate(result.expiries):
            if tau not in chosen:
                result.iv_surface[index, :] = np.nan
        return result

    monkeypatch.setattr(surface_builder, "build_combined_surface_from_chain", sparse_result)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert [tau for tau, box in zip(ordered_expiries, app.checkbox) if box.value] == expected
    assert len(app.get("plotly_chart")) == (3 if finite_count else 2)


@pytest.mark.parametrize("case", ["file", "schema", "ticker", "quotes"])
def test_dashboard_input_errors_are_visible(monkeypatch, case):
    if case == "file":
        def load(path):
            raise OSError("missing file")
    else:
        snapshot = json.loads(
            (APP_PATH.parent / "data/bybit_btc_options_snapshot.json").read_text()
        )
        if case == "schema":
            snapshot["schema_version"] = -1
        elif case == "ticker":
            snapshot["response"]["result"]["list"][0]["symbol"] = None
        else:
            for ticker in snapshot["response"]["result"]["list"]:
                ticker["bid1Price"] = ""
                ticker["ask1Price"] = ""

        def load(path):
            return snapshot

    monkeypatch.setattr(fetcher, "load_chain_snapshot", load)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert len(app.error) == 1
    assert not app.get("plotly_chart")
