import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from iv_surface import fetcher, surface_builder


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
    assert len(app.get("plotly_chart")) == 2
    assert app.radio[0].value == "Core (0.8–1.2)"
    assert app.radio[1].value == "Surface"
    assert [tab.label for tab in app.tabs] == ["Heatmap", "3D surface"]
    assert any(
        "211 observed · 150 interpolated · 39 blank" in caption.value
        for caption in app.caption
    )
    assert any("Frozen snapshot" in caption.value for caption in app.caption)

    app.radio[1].set_value("Observed").run()
    assert not app.exception
    assert app.radio[1].value == "Observed"

    app.radio[0].set_value("Extended (0.7–1.3)").run()
    assert not app.exception
    assert any(
        "253 observed · 192 interpolated · 95 blank" in caption.value
        for caption in app.caption
    )

    app.number_input[0].set_value(0.25).run()
    assert not app.exception
    assert results[-1][0] == 0.0025
    assert not np.allclose(
        results[0][1].iv_surface, results[-1][1].iv_surface, equal_nan=True,
    )


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
