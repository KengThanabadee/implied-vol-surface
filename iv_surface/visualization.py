import numpy as np
import pandas as pd
import plotly.graph_objects as go


_LONG_FRAME_COLUMNS = {"tau", "strike", "iv", "moneyness"}
_ATM_FRAME_COLUMNS = {
    "tau",
    "strike",
    "iv",
    "moneyness",
    "atm_distance",
    "selection_status",
}


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")

    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _format_expiry_label(tau, expiry_datetime=pd.NaT) -> str:
    if pd.isna(expiry_datetime):
        return f"tau={tau:g}"

    timestamp = pd.Timestamp(expiry_datetime)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime("%Y-%m-%d")


def _expiry_labels(frame: pd.DataFrame, taus) -> list[str]:
    labels = []
    has_metadata = "expiry_datetime" in frame.columns
    for tau in taus:
        expiry_datetime = pd.NaT
        if has_metadata:
            values = frame.loc[
                frame["tau"] == tau, "expiry_datetime"
            ].dropna()
            if not values.empty:
                expiry_datetime = values.iloc[0]
        labels.append(_format_expiry_label(tau, expiry_datetime))
    return labels


def _spaced_tau_ticks(taus, minimum_spacing_fraction=0.08):
    values = np.asarray(taus, dtype=float)
    if len(values) <= 2:
        return values

    spacing = (values.max() - values.min()) * minimum_spacing_fraction
    visible = [0]
    for index in range(1, len(values)):
        if values[index] - values[visible[-1]] >= spacing:
            visible.append(index)
    if visible[-1] != len(values) - 1:
        visible.append(len(values) - 1)
    return values[visible]


def _heatmap_customdata(long_frame, iv_grid, strike_grid):
    customdata = strike_grid.to_numpy(dtype=float)
    strike_hover = "strike: %{customdata:g}<br>"
    source_hover = ""
    value_type_hover = ""

    has_source = "source" in long_frame.columns
    has_value_type = "surface_value_type" in long_frame.columns
    if not has_source and not has_value_type:
        return customdata, strike_hover, source_hover, value_type_hover

    fields = [customdata]
    if has_source:
        display_source = long_frame["source"].copy()
        if has_value_type:
            display_source = display_source.mask(
                long_frame["surface_value_type"].eq("Interpolated"),
                "1D interpolation within expiry",
            )
        source_grid = long_frame.assign(source=display_source).pivot(
            index="tau", columns="moneyness", values="source"
        ).reindex(index=iv_grid.index, columns=iv_grid.columns)
        fields.append(source_grid.to_numpy())
    if has_value_type:
        value_type_grid = long_frame.pivot(
            index="tau", columns="moneyness", values="surface_value_type"
        ).reindex(index=iv_grid.index, columns=iv_grid.columns)
        fields.append(value_type_grid.to_numpy())

    combined = np.empty((*iv_grid.shape, len(fields)), dtype=object)
    for index, values in enumerate(fields):
        combined[:, :, index] = values

    customdata = combined
    strike_hover = "strike: %{customdata[0]:g}<br>"
    next_index = 1
    if has_source:
        source_label = "Value source" if has_value_type else "Combined source"
        source_hover = f"{source_label}: %{{customdata[{next_index}]}}<br>"
        next_index += 1
    if has_value_type:
        value_type_hover = f"Surface value: %{{customdata[{next_index}]}}<br>"
    return customdata, strike_hover, source_hover, value_type_hover


def make_iv_heatmap(
    long_frame: pd.DataFrame, *, equal_expiry_spacing: bool = False
) -> go.Figure:
    """Create a moneyness-by-expiry heatmap without filling IV itself."""
    _require_columns(long_frame, _LONG_FRAME_COLUMNS, "long_frame")

    iv_grid = (
        long_frame.pivot(index="tau", columns="moneyness", values="iv")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    strike_grid = (
        long_frame.pivot(index="tau", columns="moneyness", values="strike")
        .reindex(index=iv_grid.index, columns=iv_grid.columns)
    )

    customdata, strike_hover, source_hover, value_type_hover = (
        _heatmap_customdata(long_frame, iv_grid, strike_grid)
    )
    expiry_labels = _expiry_labels(long_frame, iv_grid.index)
    y_values = expiry_labels if equal_expiry_spacing else iv_grid.index.to_numpy(
        dtype=float
    )
    expiry_hover = (
        "Expiry: %{y}<br>"
        if equal_expiry_spacing
        else "tau: %{y:.4f} years<br>"
    )

    trace = go.Heatmap(
        x=iv_grid.columns.to_numpy(dtype=float),
        y=y_values,
        z=iv_grid.to_numpy(dtype=float),
        customdata=customdata,
        connectgaps=False,
        xgap=1,
        ygap=1,
        colorbar={"title": "implied volatility", "tickformat": ".0%"},
        hovertemplate=(
            "K / S: %{x:.4f}<br>"
            + expiry_hover
            + strike_hover
            + source_hover
            + value_type_hover
            + "IV: %{z:.2%}<extra></extra>"
        ),
        colorscale="Viridis",
    )
    figure = go.Figure(data=[trace])
    figure.update_layout(
        xaxis_title="K / S",
        yaxis_title="Expiry" if equal_expiry_spacing else "tau (years)",
        plot_bgcolor="rgba(128, 128, 128, 0.18)",
    )
    if equal_expiry_spacing:
        figure.update_yaxes(
            type="category",
            categoryorder="array",
            categoryarray=expiry_labels,
        )
    else:
        figure.update_yaxes(
            tickmode="array",
            tickvals=iv_grid.index.to_numpy(dtype=float),
            ticktext=expiry_labels,
        )
    return figure


def make_iv_surface_figure(long_frame: pd.DataFrame) -> go.Figure:
    """Create a 3D IV surface from the same rectangular frame as the heatmap."""
    _require_columns(long_frame, _LONG_FRAME_COLUMNS, "long_frame")

    iv_grid = (
        long_frame.pivot(index="tau", columns="moneyness", values="iv")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    strike_grid = (
        long_frame.pivot(index="tau", columns="moneyness", values="strike")
        .reindex(index=iv_grid.index, columns=iv_grid.columns)
    )
    customdata, strike_hover, source_hover, value_type_hover = (
        _heatmap_customdata(long_frame, iv_grid, strike_grid)
    )
    visible_taus = _spaced_tau_ticks(iv_grid.index)
    trace = go.Surface(
        x=iv_grid.columns.to_numpy(dtype=float),
        y=iv_grid.index.to_numpy(dtype=float),
        z=iv_grid.to_numpy(dtype=float),
        customdata=customdata,
        connectgaps=False,
        colorscale="Viridis",
        colorbar={
            "title": {"text": "Implied volatility", "side": "right"},
            "tickformat": ".0%",
            "len": 0.72,
            "thickness": 18,
            "x": 0.96,
            "y": 0.52,
        },
        lighting={
            "ambient": 0.78,
            "diffuse": 0.88,
            "specular": 0.16,
            "roughness": 0.82,
            "fresnel": 0.08,
        },
        lightposition={"x": 80, "y": -120, "z": 180},
        opacity=0.98,
        name="Interpolated surface",
        showlegend=False,
        hovertemplate=(
            "K / S: %{x:.4f}<br>"
            "tau: %{y:.4f} years<br>"
            + strike_hover
            + source_hover
            + value_type_hover
            + "IV: %{z:.2%}<extra></extra>"
        ),
    )
    figure = go.Figure(data=[trace])
    observed_column = (
        "observed_iv" if "observed_iv" in long_frame.columns else "iv"
    )
    observed_values = long_frame[observed_column].to_numpy(dtype=float)
    observed = long_frame.loc[np.isfinite(observed_values)].copy()
    if not observed.empty:
        original_observed_iv = observed[observed_column].to_numpy(dtype=float)
        finite_surface = iv_grid.to_numpy(dtype=float)
        finite_surface = finite_surface[np.isfinite(finite_surface)]
        surface_span = np.ptp(finite_surface) if finite_surface.size else 0.0
        marker_lift = max(float(surface_span) * 0.006, 0.0005)
        observed_customdata = observed[
            ["strike", observed_column]
        ].to_numpy(dtype=float)
        observed_strike_hover = "strike: %{customdata[0]:g}<br>"
        observed_source_hover = ""
        observed_iv_index = 1
        if "source" in observed.columns:
            observed_customdata = observed[
                ["strike", "source", observed_column]
            ].to_numpy(dtype=object)
            observed_strike_hover = "strike: %{customdata[0]:g}<br>"
            observed_source_hover = (
                "Combined source: %{customdata[1]}<br>"
            )
            observed_iv_index = 2
        figure.add_trace(
            go.Scatter3d(
                x=observed["moneyness"].to_numpy(dtype=float),
                y=observed["tau"].to_numpy(dtype=float),
                z=original_observed_iv + marker_lift,
                customdata=observed_customdata,
                mode="markers",
                name="Observed IV",
                marker={
                    "size": 5,
                    "color": "#F97316",
                    "opacity": 1,
                    "line": {
                        "color": "rgba(15, 23, 42, 0.85)",
                        "width": 0.8,
                    },
                },
                hovertemplate=(
                    "Observed Combined IV<br>"
                    "K / S: %{x:.4f}<br>"
                    "tau: %{y:.4f} years<br>"
                    + observed_strike_hover
                    + observed_source_hover
                    + f"IV: %{{customdata[{observed_iv_index}]:.2%}}"
                    + "<extra></extra>"
                ),
            )
        )
    axis_style = {
        "showbackground": False,
        "gridcolor": "rgba(148, 163, 184, 0.18)",
        "zeroline": False,
        "showspikes": False,
    }
    figure.update_layout(
        scene={
            "bgcolor": "rgba(0, 0, 0, 0)",
            "xaxis": {"title": "K / S", **axis_style},
            "yaxis": {
                "title": "Expiry (tau)",
                "tickmode": "array",
                "tickvals": visible_taus,
                "ticktext": _expiry_labels(long_frame, visible_taus),
                **axis_style,
            },
            "zaxis": {
                "title": "Implied volatility",
                "tickformat": ".0%",
                **axis_style,
            },
            "camera": {
                "eye": {"x": 1.32, "y": -1.48, "z": 0.78},
                "center": {"x": 0, "y": 0, "z": -0.06},
            },
            "aspectmode": "manual",
            "aspectratio": {"x": 1.7, "y": 1.2, "z": 0.72},
            "dragmode": "turntable",
        },
        autosize=True,
        height=700,
        margin={"l": 0, "r": 36, "t": 8, "b": 0},
        paper_bgcolor="rgba(0, 0, 0, 0)",
        legend={
            "orientation": "h",
            "x": 0.02,
            "y": 0.98,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(0, 0, 0, 0)",
        },
        uirevision="iv-surface-camera",
    )
    return figure


def make_smile_figure(
    long_frame: pd.DataFrame, selected_expiries=None
) -> go.Figure:
    """Create one IV smile trace per selected expiry."""
    _require_columns(long_frame, _LONG_FRAME_COLUMNS, "long_frame")

    selected = None if selected_expiries is None else set(selected_expiries)
    figure = go.Figure()
    for tau in sorted(long_frame["tau"].dropna().unique()):
        if selected is not None and tau not in selected:
            continue
        smile = long_frame[long_frame["tau"] == tau].sort_values("moneyness")
        expiry_label = _expiry_labels(long_frame, [tau])[0]
        customdata = smile["strike"].to_numpy(dtype=float)
        strike_hover = "strike: %{customdata:g}<br>"
        source_hover = ""
        if "source" in smile.columns:
            customdata = smile[["strike", "source"]].to_numpy(dtype=object)
            strike_hover = "strike: %{customdata[0]:g}<br>"
            source_hover = "source: %{customdata[1]}<br>"
        figure.add_trace(
            go.Scatter(
                x=smile["moneyness"].to_numpy(dtype=float),
                y=smile["iv"].to_numpy(dtype=float),
                customdata=customdata,
                mode="lines+markers",
                name=expiry_label,
                meta=tau,
                connectgaps=False,
                hovertemplate=(
                    "tau: %{meta:.4f} years<br>"
                    "K / S: %{x:.4f}<br>"
                    + strike_hover
                    + source_hover
                    + "IV: %{y:.2%}<extra>%{fullData.name}</extra>"
                ),
            )
        )

    figure.add_vline(x=1.0, line_dash="dash", line_color="gray")
    figure.update_layout(xaxis_title="K / S", yaxis_title="implied volatility")
    figure.update_yaxes(tickformat=".0%")
    return figure


def make_atm_term_structure_figure(atm_frame: pd.DataFrame) -> go.Figure:
    """Create an ATM-proxy term structure while preserving rejected expiries."""
    _require_columns(atm_frame, _ATM_FRAME_COLUMNS, "atm_frame")

    data = atm_frame.sort_values("tau")
    customdata = data[
        ["strike", "moneyness", "atm_distance", "selection_status"]
    ].to_numpy()
    trace = go.Scatter(
        x=data["tau"].to_numpy(dtype=float),
        y=data["iv"].to_numpy(dtype=float),
        customdata=customdata,
        mode="lines+markers",
        connectgaps=False,
        hovertemplate=(
            "tau: %{x:.4f} years<br>"
            "IV: %{y:.2%}<br>"
            "strike: %{customdata[0]}<br>"
            "K / S: %{customdata[1]}<br>"
            "ATM distance: %{customdata[2]}<br>"
            "selection: %{customdata[3]}<extra></extra>"
        ),
    )
    figure = go.Figure(data=[trace])
    figure.update_layout(
        xaxis_title="tau (years)", yaxis_title="implied volatility"
    )
    figure.update_xaxes(
        tickmode="array",
        tickvals=data["tau"].to_numpy(dtype=float),
        ticktext=_expiry_labels(data, data["tau"]),
    )
    figure.update_yaxes(tickformat=".0%")
    return figure
