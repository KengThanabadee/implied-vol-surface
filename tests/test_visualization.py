from datetime import datetime
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from iv_surface.surface_transforms import (
    atm_term_structure_frame,
    interpolate_surface_2d,
    interpolate_smiles_1d,
    surface_to_long_frame,
)
from iv_surface.visualization import (
    make_iv_heatmap,
    make_iv_surface_figure,
    make_smile_figure,
    make_atm_term_structure_figure,
)


def _long_frame():
    return surface_to_long_frame(
        np.array([[0.20, np.nan], [0.25, 0.30]]),
        expiries=[0.5, 0.25],
        strikes=[110, 90],
        spot_price=100,
    )


def test_surface_transforms_do_not_require_plotly():
    script = """
import importlib.abc
import sys

class BlockPlotly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "plotly" or fullname.startswith("plotly."):
            raise ImportError("plotly intentionally unavailable")
        return None

sys.meta_path.insert(0, BlockPlotly())
from iv_surface.surface_transforms import (
    atm_term_structure_frame,
    interpolate_surface_2d,
    interpolate_smiles_1d,
    surface_to_long_frame,
)

frame = surface_to_long_frame([[0.2]], [0.25], [100], 100)
interpolate_smiles_1d(frame)
interpolate_surface_2d(frame)
atm_term_structure_frame(frame)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_surface_to_long_frame_preserves_shape_order_and_nan():
    frame = _long_frame()

    assert len(frame) == 4
    assert {"tau", "strike", "iv", "spot_price", "moneyness"} <= set(
        frame.columns
    )
    assert frame[["tau", "strike"]].values.tolist() == [
        [0.5, 110.0],
        [0.5, 90.0],
        [0.25, 110.0],
        [0.25, 90.0],
    ]
    assert frame["moneyness"].tolist() == [1.1, 0.9, 1.1, 0.9]
    assert np.isnan(frame.loc[1, "iv"])


def test_surface_to_long_frame_rejects_bad_shape_and_spot_price():
    with pytest.raises(ValueError, match="shape"):
        surface_to_long_frame(np.ones((1, 2)), [0.25, 0.5], [90, 100], 100)

    for spot_price in [0, -1, np.nan, np.inf]:
        with pytest.raises(ValueError, match="spot_price"):
            surface_to_long_frame(np.ones((1, 1)), [0.25], [100], spot_price)


def test_source_grid_preserves_alignment_and_existing_output():
    args = ([[0.2, np.nan], [0.3, 0.4]], [0.5, 0.25], [110, 100], 100)
    metadata = ["2026-07-01", "2026-06-01"]
    baseline = surface_to_long_frame(*args, expiry_datetimes=metadata)
    frame = surface_to_long_frame(
        *args, expiry_datetimes=metadata,
        source_grid=[["call", "missing"], ["put_fallback", "near_atm_average"]],
    )
    assert frame.source.tolist() == [
        "call", "missing", "put_fallback", "near_atm_average"
    ]
    pd.testing.assert_frame_equal(frame.drop(columns="source"), baseline)
    assert "source" not in baseline.columns


@pytest.mark.parametrize("sources", ["call", ["call", "put"], [["call"], ["put"]]])
def test_source_grid_rejects_mismatched_shape(sources):
    with pytest.raises(ValueError, match="source_grid shape"):
        surface_to_long_frame([[0.2, 0.3]], [0.25], [90, 110], 100,
                              source_grid=sources)


def test_interpolate_smiles_1d_fills_only_interior_gaps_per_expiry():
    frame = pd.DataFrame(
        {
            "tau": [0.25] * 5 + [0.5] * 5,
            "strike": [80, 90, 100, 110, 120] * 2,
            "moneyness": [0.8, 0.9, 1.0, 1.1, 1.2] * 2,
            "iv": [np.nan, 0.30, np.nan, 0.50, np.nan,
                   np.nan, np.nan, 0.40, np.nan, np.nan],
            "source": ["missing", "put", "missing", "call", "missing"] * 2,
        }
    )

    result = interpolate_smiles_1d(frame)

    near = result[result["tau"] == 0.25].sort_values("moneyness")
    far = result[result["tau"] == 0.5].sort_values("moneyness")
    assert np.isnan(near["iv"].iloc[0])
    assert near["iv"].iloc[2] == pytest.approx(0.40)
    assert np.isnan(near["iv"].iloc[4])
    assert near["surface_value_type"].tolist() == [
        "Missing", "Observed", "Interpolated", "Observed", "Missing"
    ]
    assert near["observed_iv"].iloc[1] == 0.30
    assert np.isnan(near["observed_iv"].iloc[2])
    assert near["observed_iv"].iloc[3] == 0.50
    assert near["source"].tolist() == frame.iloc[:5]["source"].tolist()
    assert np.isnan(far["iv"].iloc[[0, 1, 3, 4]]).all()
    assert far["surface_value_type"].tolist() == [
        "Missing", "Missing", "Observed", "Missing", "Missing"
    ]


def test_interpolate_smiles_1d_handles_duplicate_row_indices():
    near = surface_to_long_frame(
        [[np.nan, 0.30, np.nan, 0.50, np.nan]],
        [0.25], [80, 90, 100, 110, 120], 100,
        source_grid=[["missing", "put", "missing", "call", "missing"]],
    )
    far = surface_to_long_frame(
        [[np.nan, np.nan, 0.70, np.nan, np.nan]],
        [0.5], [80, 90, 100, 110, 120], 100,
        source_grid=[["missing", "missing", "near_atm_average", "missing", "missing"]],
    )
    frame = pd.concat([near, far]).sample(frac=1, random_state=7)
    frame.index.name = "input_row"
    original = frame.copy(deep=True)
    assert not frame.index.is_unique

    result = interpolate_smiles_1d(frame)

    pd.testing.assert_index_equal(result.index, original.index)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(
        result.drop(columns=["iv", "observed_iv", "surface_value_type"]),
        original.drop(columns="iv"),
    )
    pd.testing.assert_series_equal(
        result["observed_iv"], original["iv"], check_names=False,
    )
    near_result = result[result["tau"] == 0.25].sort_values("moneyness")
    far_result = result[result["tau"] == 0.5].sort_values("moneyness")
    np.testing.assert_allclose(
        near_result["iv"], [np.nan, 0.30, 0.40, 0.50, np.nan], equal_nan=True,
    )
    np.testing.assert_allclose(
        far_result["iv"], [np.nan, np.nan, 0.70, np.nan, np.nan], equal_nan=True,
    )
    assert near_result["surface_value_type"].tolist() == [
        "Missing", "Observed", "Interpolated", "Observed", "Missing"
    ]
    assert far_result["surface_value_type"].tolist() == [
        "Missing", "Missing", "Observed", "Missing", "Missing"
    ]


def test_interpolate_smiles_1d_clips_range_and_rejects_ambiguous_inputs():
    result = interpolate_smiles_1d(_long_frame(), 0.95, 1.2)
    assert result["moneyness"].tolist() == [1.1, 1.1]

    for bounds in [(1.0, 1.0), (1.1, 1.0), (np.nan, 1.0)]:
        with pytest.raises(ValueError, match="moneyness bounds"):
            interpolate_smiles_1d(_long_frame(), *bounds)

    duplicate = pd.concat([_long_frame(), _long_frame().iloc[[0]]])
    with pytest.raises(ValueError, match="unique tau and moneyness"):
        interpolate_smiles_1d(duplicate)


def test_chart_sources_follow_sorted_cells_and_selected_expiry():
    frame = surface_to_long_frame(
        [[0.2, np.nan, 0.22], [0.3, 0.4, 0.35]],
        [0.5, 0.25], [110, 90, 100], 100,
        source_grid=[
            ["call", "missing", "near_atm_call_only"],
            ["put_fallback", "put", "near_atm_average"],
        ],
    ).sample(frac=1, random_state=7)
    heatmap = make_iv_heatmap(frame).data[0]
    assert heatmap.customdata.tolist() == [
        [[90.0, "put"], [100.0, "near_atm_average"], [110.0, "put_fallback"]],
        [[90.0, "missing"], [100.0, "near_atm_call_only"], [110.0, "call"]],
    ]
    assert np.isnan(heatmap.z[1, 0])
    assert heatmap.z[0].tolist() == [0.4, 0.35, 0.3]
    smile = make_smile_figure(frame, selected_expiries=[0.5]).data[0]
    assert smile.customdata.tolist() == heatmap.customdata[1].tolist()
    assert np.isnan(smile.y[0])
    assert smile.y[1:].tolist() == [0.22, 0.2]
    for trace in [heatmap, smile]:
        assert "source: %{customdata[1]}" in trace.hovertemplate
        assert "strike: %{hovertext}" in trace.hovertemplate
        assert trace.connectgaps is False
    for figure in [make_iv_heatmap(_long_frame()), make_smile_figure(_long_frame())]:
        assert all("source:" not in trace.hovertemplate for trace in figure.data)


def test_surface_to_long_frame_aligns_and_normalizes_expiry_datetimes():
    surface = np.array([[0.20], [0.25], [np.nan]])
    expiries = [0.5, 0.25, 0.125]
    without_metadata = surface_to_long_frame(surface, expiries, [100], 100)

    with_metadata = surface_to_long_frame(
        surface,
        expiries,
        [100],
        100,
        expiry_datetimes=[
            "2026-06-30T20:00:00-04:00",
            datetime(2026, 7, 2, 8),
            None,
        ],
    )

    assert str(with_metadata["expiry_datetime"].dtype) == "datetime64[ns, UTC]"
    assert with_metadata["expiry_datetime"].tolist() == [
        pd.Timestamp("2026-07-01T00:00:00Z"),
        pd.Timestamp("2026-07-02T08:00:00Z"),
        pd.NaT,
    ]
    assert "expiry_datetime" not in without_metadata.columns
    pd.testing.assert_frame_equal(
        with_metadata.drop(columns="expiry_datetime"),
        without_metadata,
    )


def test_surface_to_long_frame_rejects_invalid_expiry_datetimes():
    with pytest.raises(ValueError, match=r"expiry_datetimes.*len\(expiries\)"):
        surface_to_long_frame(
            np.ones((2, 1)),
            [0.25, 0.5],
            [100],
            100,
            expiry_datetimes=["2026-07-01"],
        )

    with pytest.raises(ValueError, match="expiry_datetimes.*valid"):
        surface_to_long_frame(
            np.ones((1, 1)),
            [0.25],
            [100],
            100,
            expiry_datetimes=["not-a-date"],
        )


def test_surface_to_long_frame_accepts_mixed_valid_datetime_formats():
    frame = surface_to_long_frame(
        np.ones((2, 1)),
        [0.25, 0.5],
        [100],
        100,
        expiry_datetimes=["2026-07-01", "2026-07-02T08:00:00Z"],
    )

    assert frame["expiry_datetime"].tolist() == [
        pd.Timestamp("2026-07-01T00:00:00Z"),
        pd.Timestamp("2026-07-02T08:00:00Z"),
    ]


def test_atm_term_structure_selects_nearest_and_accepts_bound():
    frame = pd.DataFrame(
        {
            "tau": [0.5, 0.25, 0.25],
            "strike": [102, 94, 102],
            "iv": [0.30, 0.24, 0.20],
            "moneyness": [1.02, 0.94, 1.02],
        }
    )

    result = atm_term_structure_frame(frame)

    assert result["tau"].tolist() == [0.25, 0.5]
    assert result["strike"].tolist() == [102.0, 102.0]
    assert result["selection_status"].tolist() == ["nearest", "nearest"]
    assert np.allclose(result["atm_distance"], [0.02, 0.02])


@pytest.mark.parametrize("right_moneyness", [1.02, 1.0200000000005])
def test_atm_term_structure_averages_exact_and_near_ties(right_moneyness):
    frame = pd.DataFrame(
        {
            "tau": [0.25, 0.25],
            "strike": [98, 102],
            "iv": [0.20, 0.24],
            "moneyness": [0.98, right_moneyness],
        }
    )

    result = atm_term_structure_frame(frame).iloc[0]

    assert result["selection_status"] == "symmetric_average"
    assert result["strike"] == 100
    assert result["iv"] == pytest.approx(0.22)
    assert result["moneyness"] == pytest.approx(
        (0.98 + right_moneyness) / 2
    )


def test_atm_term_structure_does_not_average_distinct_distances():
    frame = pd.DataFrame(
        {
            "tau": [0.25, 0.25],
            "strike": [98, 102],
            "iv": [0.20, 0.24],
            "moneyness": [0.98, 1.020000000002],
        }
    )

    result = atm_term_structure_frame(frame).iloc[0]

    assert result["selection_status"] == "nearest"
    assert result["strike"] == 98
    assert result["iv"] == 0.20


def test_atm_term_structure_preserves_outside_bound_and_no_finite_iv():
    frame = pd.DataFrame(
        {
            "tau": [0.25, 0.5],
            "strike": [120, 100],
            "iv": [0.30, np.nan],
            "moneyness": [1.20, 1.0],
        }
    )

    result = atm_term_structure_frame(frame)
    outside = result.iloc[0]
    missing = result.iloc[1]

    assert outside["selection_status"] == "outside_bound"
    assert np.isnan(outside[["strike", "iv", "moneyness"]].astype(float)).all()
    assert outside["atm_distance"] == pytest.approx(0.20)
    assert missing["selection_status"] == "no_finite_iv"
    assert np.isnan(
        missing[["strike", "iv", "moneyness", "atm_distance"]].astype(float)
    ).all()


def test_atm_term_structure_rejects_nearest_just_outside_default_two_percent():
    frame = pd.DataFrame(
        {
            "tau": [0.25],
            "strike": [102.0000000002],
            "iv": [0.20],
            "moneyness": [1.020000000002],
        }
    )

    result = atm_term_structure_frame(frame).iloc[0]

    assert result["selection_status"] == "outside_bound"
    assert result["atm_distance"] == pytest.approx(0.020000000002)
    assert np.isnan(result["iv"])


def test_atm_term_structure_preserves_expiry_metadata_for_rejected_rows():
    long_frame = pd.DataFrame(
        {
            "tau": [0.5, 0.25, 0.125],
            "strike": [120, 100, 100],
            "iv": [0.30, np.nan, 0.20],
            "moneyness": [1.20, 1.0, 1.0],
        }
    )
    with_metadata = long_frame.assign(
        expiry_datetime=pd.to_datetime(
            [
                "2026-07-01T08:00:00Z",
                "2026-06-01T08:00:00Z",
                "2026-05-01T08:00:00Z",
            ],
            utc=True,
        )
    )

    baseline = atm_term_structure_frame(long_frame)
    result = atm_term_structure_frame(with_metadata)

    pd.testing.assert_frame_equal(
        result.drop(columns="expiry_datetime"), baseline
    )
    assert result["expiry_datetime"].tolist() == [
        pd.Timestamp("2026-05-01T08:00:00Z"),
        pd.Timestamp("2026-06-01T08:00:00Z"),
        pd.Timestamp("2026-07-01T08:00:00Z"),
    ]
    assert result["selection_status"].tolist() == [
        "nearest",
        "no_finite_iv",
        "outside_bound",
    ]


@pytest.mark.parametrize("distance", [-0.01, np.nan, np.inf, -np.inf])
def test_atm_term_structure_rejects_invalid_distance_bound(distance):
    with pytest.raises(ValueError, match="finite and non-negative"):
        atm_term_structure_frame(_long_frame(), distance)


def test_make_iv_heatmap_sorts_grid_keeps_nan_and_strike_hover():
    figure = make_iv_heatmap(_long_frame())

    assert isinstance(figure, go.Figure)
    trace = figure.data[0]
    assert isinstance(trace, go.Heatmap)
    assert trace.x.tolist() == [0.9, 1.1]
    assert trace.y.tolist() == [0.25, 0.5]
    assert np.asarray(trace.z).shape == (2, 2)
    assert np.asarray(trace.z)[0].tolist() == [0.30, 0.25]
    assert np.isnan(np.asarray(trace.z)[1, 0])
    assert np.asarray(trace.z)[1, 1] == 0.20
    assert np.asarray(trace.customdata).tolist() == [[90.0, 110.0], [90.0, 110.0]]
    assert trace.connectgaps is False
    assert "strike" in trace.hovertemplate
    assert figure.layout.xaxis.title.text == "K / S"
    assert figure.layout.yaxis.title.text == "tau (years)"


def test_surface_heatmap_uses_equal_expiry_rows_and_value_type_hover():
    frame = interpolate_smiles_1d(
        surface_to_long_frame(
            [[0.30, np.nan, 0.50], [0.25, 0.35, 0.45]],
            [0.25, 0.5],
            [90, 100, 110],
            100,
            expiry_datetimes=["2026-06-01", "2026-09-01"],
            source_grid=[
                ["put", "missing", "call"],
                ["put", "near_atm_average", "call"],
            ],
        )
    )

    figure = make_iv_heatmap(frame, equal_expiry_spacing=True)
    trace = figure.data[0]
    assert list(trace.y) == ["2026-06-01", "2026-09-01"]
    assert figure.layout.yaxis.title.text == "Expiry (UTC)"
    assert figure.layout.yaxis.type == "category"
    assert list(figure.layout.yaxis.categoryarray) == [
        "2026-06-01", "2026-09-01"
    ]
    assert np.asarray(trace.z)[0].tolist() == pytest.approx([0.30, 0.40, 0.50])
    assert trace.customdata[0][1].tolist() == [
        100.0, "1D interpolation within expiry", "Interpolated"
    ]
    assert "Surface value: %{customdata[2]}" in trace.hovertemplate
    assert "Value source: %{customdata[1]}" in trace.hovertemplate
    surface = make_iv_surface_figure(frame).data[0]
    assert surface.customdata.tolist() == trace.customdata.tolist()
    assert surface.hovertemplate is None
    assert surface.hoverinfo == "text"
    assert "Value source: 1D interpolation within expiry" in surface.text[0][1]
    # Display provenance changes must not rewrite the original quote provenance.
    assert frame.loc[frame["surface_value_type"].eq("Interpolated"), "source"].tolist() == [
        "missing"
    ]
    assert trace.customdata[0][0].tolist() == [90.0, "put", "Observed"]

    observed_frame = frame.copy()
    observed_frame["iv"] = observed_frame["observed_iv"]
    observed_frame["surface_value_type"] = np.where(
        np.isfinite(observed_frame["observed_iv"]), "Observed", "Missing"
    )
    observed_heatmap = make_iv_heatmap(observed_frame).data[0]
    assert observed_heatmap.customdata[0][1].tolist() == [
        100.0, "missing", "Missing"
    ]


def test_3d_surface_uses_same_values_and_true_tau_axis_as_heatmap():
    frame = interpolate_smiles_1d(
        surface_to_long_frame(
            [[0.30, np.nan, 0.50], [0.25, 0.35, 0.45]],
            [0.25, 0.5], [90, 100, 110], 100,
            expiry_datetimes=["2026-06-01", "2026-09-01"],
        )
    )
    heatmap = make_iv_heatmap(frame, equal_expiry_spacing=True).data[0]
    surface = make_iv_surface_figure(frame)
    compact = make_iv_surface_figure(frame, compact_layout=True)
    np.testing.assert_allclose(compact.data[0].z, surface.data[0].z, equal_nan=True)
    np.testing.assert_array_equal(compact.data[0].y, surface.data[0].y)
    np.testing.assert_array_equal(compact.data[0].text, surface.data[0].text)
    assert compact.layout.height < surface.layout.height
    trace = surface.data[0]
    observed = surface.data[1]

    assert isinstance(trace, go.Surface)
    assert isinstance(observed, go.Scatter3d)
    assert trace.y.tolist() == [0.25, 0.5]
    assert np.asarray(trace.z).tolist() == np.asarray(heatmap.z).tolist()
    assert trace.connectgaps is False
    assert list(surface.layout.scene.yaxis.ticktext) == [
        "2026<br>06-01", "2026<br>09-01"
    ]
    assert surface.layout.scene.aspectratio.x == 1.7
    assert surface.layout.scene.aspectratio.y == 1.2
    assert surface.layout.scene.aspectratio.z == 0.72
    assert surface.layout.scene.dragmode == "turntable"
    assert surface.layout.height == 640
    assert trace.colorbar.thickness == 18
    assert trace.colorbar.len == 0.72
    assert trace.lighting.ambient == 0.78
    assert "Surface value: Interpolated" in trace.text[0][1]
    assert observed.name == "Observed IV"
    assert observed.z.tolist() != pytest.approx(
        [0.30, 0.50, 0.25, 0.35, 0.45]
    )
    assert np.asarray(observed.customdata)[:, 1].astype(float).tolist() == (
        pytest.approx([0.30, 0.50, 0.25, 0.35, 0.45])
    )
    assert np.all(
        np.asarray(observed.z, dtype=float)
        > np.asarray(observed.customdata)[:, 1].astype(float)
    )
    assert observed.marker.size == 3
    assert observed.marker.color == "#F97316"
    assert observed.hovertemplate is None
    assert observed.hoverinfo == "text"
    assert "Surface value: Observed" in observed.text[0]


def test_surface_hover_text_is_aligned_on_a_nonsquare_grid():
    frame = surface_to_long_frame(
        [[.21, .22, .23, .24], [.31, .32, .33, .34]],
        [.5, .25], [110, 90, 100, 120], 100,
        source_grid=[["call", "put", "near_atm_average", "call_fallback"],
                     ["call", "put", "near_atm_put_only", "call"]],
    ).sample(frac=1, random_state=4)
    trace = make_iv_surface_figure(frame).data[0]
    assert np.asarray(trace.text).shape == (2, 4)
    assert "strike: 90<br>Combined source: put" in trace.text[0][0]
    assert "strike: 120<br>Combined source: call" in trace.text[0][3]
    assert "strike: 120<br>Combined source: call_fallback" in trace.text[1][3]
    assert "tau: 0.2500 years" in trace.text[0][3]
    assert "IV: 34.00%" in trace.text[0][3]
    assert trace.hovertemplate is None
    assert trace.hoverinfo == "text"
    serialized = json.loads(trace.to_json())
    assert len(serialized["text"]) == 2
    assert len(serialized["text"][0]) == 4
    assert "strike: 120" in serialized["text"][1][3]


def test_make_smile_figure_filters_and_sorts_without_changing_iv_values():
    figure = make_smile_figure(_long_frame(), selected_expiries=[0.5])

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 1
    trace = figure.data[0]
    assert trace.name == "tau=0.5"
    assert trace.x.tolist() == [0.9, 1.1]
    assert np.isnan(trace.y[0])
    assert trace.y[1] == 0.20
    assert trace.customdata.tolist() == [90.0, 110.0]
    assert trace.connectgaps is False
    assert figure.layout.yaxis.tickformat == ".0%"
    assert len(figure.layout.shapes) == 1
    assert figure.layout.shapes[0].x0 == 1.0
    assert figure.layout.shapes[0].x1 == 1.0


def test_make_smile_figure_orders_all_expiries_ascending():
    figure = make_smile_figure(_long_frame())

    assert [trace.name for trace in figure.data] == ["tau=0.25", "tau=0.5"]
    assert all(trace.connectgaps is False for trace in figure.data)


def test_smile_axis_fits_selected_observations_without_removing_gaps():
    frame = surface_to_long_frame(
        [[np.nan, .4, np.nan, .5, np.nan], [.6, np.nan, np.nan, np.nan, .7]],
        [.25, .5], [10, 90, 100, 110, 400], 100,
    )
    figure = make_smile_figure(frame, selected_expiries=[.25])
    assert list(figure.layout.xaxis.range) == pytest.approx([.89, 1.11])
    assert len(figure.data[0].x) == 5
    np.testing.assert_allclose(figure.data[0].y, frame.iloc[:5]["iv"], equal_nan=True)
    assert figure.data[0].connectgaps is False
    assert figure.layout.legend.orientation == "h"
    assert figure.layout.height == 430
    all_expiries = make_smile_figure(frame)
    selected_later = make_smile_figure(frame, selected_expiries=[.5])
    assert selected_later.data[0].line.color == all_expiries.data[1].line.color


@pytest.mark.parametrize("values", [[np.nan, np.nan], [.4, np.nan]])
def test_smile_axis_handles_missing_or_single_observation(values):
    frame = surface_to_long_frame([values], [.25], [100, 400], 100)
    figure = make_smile_figure(frame)
    expected = [.8, 1.2] if np.isnan(values).all() else [.99, 1.01]
    assert list(figure.layout.xaxis.range) == pytest.approx(expected)


def test_make_atm_term_structure_figure_keeps_gap_and_hover_fields():
    atm_frame = atm_term_structure_frame(
        pd.DataFrame(
            {
                "tau": [0.5, 0.25],
                "strike": [120, 100],
                "iv": [0.30, np.nan],
                "moneyness": [1.20, 1.0],
            }
        )
    )

    figure = make_atm_term_structure_figure(atm_frame)

    assert isinstance(figure, go.Figure)
    trace = figure.data[0]
    assert trace.x.tolist() == [0.25, 0.5]
    assert np.isnan(trace.y).all()
    assert trace.connectgaps is False
    assert np.asarray(trace.customdata).shape == (2, 4)
    assert "tau: 0.2500 years" in trace.text[0]
    for label in ["IV", "strike", "K / S", "ATM distance", "selection"]:
        assert label in trace.hovertemplate
    assert figure.layout.yaxis.tickformat == ".0%"


def test_figures_use_date_labels_and_tau_fallback_without_moving_coordinates():
    long_frame = surface_to_long_frame(
        np.array([[0.20, np.nan], [0.25, 0.30]]),
        expiries=[0.5, 0.25],
        strikes=[110, 90],
        spot_price=100,
        expiry_datetimes=["2026-07-01T08:00:00Z", None],
    )

    heatmap = make_iv_heatmap(long_frame)
    heatmap_trace = heatmap.data[0]
    assert heatmap_trace.y.tolist() == [0.25, 0.5]
    assert list(heatmap.layout.yaxis.tickvals) == [0.25, 0.5]
    assert list(heatmap.layout.yaxis.ticktext) == ["tau=0.25", "2026-07-01"]
    assert np.isnan(np.asarray(heatmap_trace.z)[1, 0])
    assert heatmap_trace.connectgaps is False

    smiles = make_smile_figure(long_frame)
    assert [trace.name for trace in smiles.data] == [
        "tau=0.25",
        "2026-07-01",
    ]
    assert [trace.meta for trace in smiles.data] == [0.25, 0.5]
    assert all("tau:" in trace.text[0] for trace in smiles.data)
    assert all(trace.connectgaps is False for trace in smiles.data)

    selected = make_smile_figure(long_frame, selected_expiries=[0.5])
    assert len(selected.data) == 1
    assert selected.data[0].name == "2026-07-01"
    assert selected.data[0].x.tolist() == [0.9, 1.1]
    assert np.isnan(selected.data[0].y[0])
    assert selected.data[0].y[1] == 0.20

    atm = make_atm_term_structure_figure(atm_term_structure_frame(long_frame))
    atm_trace = atm.data[0]
    assert atm_trace.x.tolist() == [0.25, 0.5]
    assert list(atm.layout.xaxis.tickvals) == [0.25, 0.5]
    assert list(atm.layout.xaxis.ticktext) == ["tau=0.25", "2026-07-01"]
    assert np.isnan(atm_trace.y).all()
    assert atm_trace.connectgaps is False


def test_all_views_share_source_expiry_and_frozen_maturity_hover():
    frame = surface_to_long_frame(
        [[.21, .22, np.nan], [.31, .32, .33]],
        [.5, .25], [90, 100, 110], 100,
        expiry_datetimes=['2027-03-26T01:00:00-07:00', None],
    ).sample(frac=1, random_state=7)
    expected = [
        'Expiry (UTC): Unavailable<br>Time to expiry: 91.31 days<br>tau: 0.2500 years',
        'Expiry (UTC): 2027-03-26<br>Time to expiry: 182.62 days<br>tau: 0.5000 years',
    ]
    for categorical in [False, True]:
        trace = make_iv_heatmap(frame, equal_expiry_spacing=categorical).data[0]
        assert np.asarray(trace.text).shape == (2, 3)
        assert trace.text[:, 0].tolist() == expected
        assert '%{text}' in trace.hovertemplate
    surface = make_iv_surface_figure(frame)
    for index, maturity in enumerate(expected):
        assert all(text.startswith(maturity + '<br>') for text in surface.data[0].text[index])
    points = frame.loc[np.isfinite(frame['iv'])]
    assert all(
        text.startswith(expected[int(tau == .5)] + '<br>')
        for text, tau in zip(surface.data[1].text, points['tau'])
    )
    for trace, maturity in zip(make_smile_figure(frame).data, expected):
        assert list(trace.text) == [maturity] * 3
    atm = make_atm_term_structure_figure(atm_term_structure_frame(frame)).data[0]
    assert list(atm.text) == expected
    assert list(atm.x) == [.25, .5]


def test_surface_mesh_and_observed_markers_share_identical_hover_for_same_quote():
    frame = interpolate_smiles_1d(surface_to_long_frame(
        [[.30, np.nan, .50], [.25, .35, .45]],
        [.25, .5], [90, 100, 110], 100,
        expiry_datetimes=['2026-06-01', '2026-09-01'],
        source_grid=[['put', 'missing', 'call'], ['put', 'near_atm_average', 'call']],
    )).sample(frac=1, random_state=6)
    surface, markers = make_iv_surface_figure(frame).data
    taus, moneyness = list(surface.y), list(surface.x)
    for x, tau, text, lifted_iv, custom in zip(
        markers.x, markers.y, markers.text, markers.z, markers.customdata,
    ):
        assert text == surface.text[taus.index(tau)][moneyness.index(x)]
        assert f'IV: {float(custom[2]):.2%}' in text
        assert lifted_iv > float(custom[2])
    assert 'Surface value: Interpolated' in surface.text[0][1]
    assert 'Value source: 1D interpolation within expiry' in surface.text[0][1]


@pytest.mark.parametrize("with_source", [False, True])
def test_all_chart_strike_labels_preserve_fractions_and_numeric_data(with_source):
    frame = surface_to_long_frame(
        [[.20, .21], [np.nan, .22]], [.25, .5], [82000, 82000.25], 82000,
        source_grid=(
            [["near_atm_average", "call"], ["missing", "call"]]
            if with_source else None
        ),
    ).sample(frac=1, random_state=9)
    original = frame.copy(deep=True)
    labels = ["82000", "82000.25"]
    heatmap = make_iv_heatmap(frame).data[0]
    assert np.asarray(heatmap.hovertext).tolist() == [labels, labels]
    assert json.loads(heatmap.to_json())["hovertext"] == [labels, labels]
    strikes = np.asarray(heatmap.customdata)
    if with_source:
        strikes = strikes[:, :, 0]
    np.testing.assert_array_equal(strikes.astype(float), [[82000, 82000.25]] * 2)
    assert "strike: %{hovertext}" in heatmap.hovertemplate
    for trace in make_smile_figure(frame).data:
        assert list(trace.hovertext) == labels
        assert "strike: %{hovertext}" in trace.hovertemplate
        numeric = np.asarray(trace.customdata)
        if with_source:
            numeric = numeric[:, 0]
        np.testing.assert_array_equal(numeric.astype(float), [82000, 82000.25])
    surface, markers = make_iv_surface_figure(frame).data
    for row in surface.text:
        for label, text in zip(labels, row):
            assert f"strike: {label}<br>" in text
    for custom, text in zip(markers.customdata, markers.text):
        label = labels[int(float(custom[0]) == 82000.25)]
        assert f"strike: {label}<br>" in text
    atm_frame = atm_term_structure_frame(frame)
    atm = make_atm_term_structure_figure(atm_frame).data[0]
    assert list(atm.hovertext) == labels
    assert "strike: %{hovertext}" in atm.hovertemplate
    np.testing.assert_array_equal(np.asarray(atm.customdata)[:, 0].astype(float),
                                  [82000, 82000.25])
    assert np.isnan(heatmap.z[1, 0])
    pd.testing.assert_frame_equal(frame, original)
