from html import escape

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


def _maturity_hover(frame, taus):
    """Format source expiry and frozen maturity consistently across charts."""
    return [
        f"Expiry (UTC): {label if not label.startswith('tau=') else 'Unavailable'}<br>"
        f"Time to expiry: {float(tau) * 365.25:.2f} days<br>"
        f"tau: {float(tau):.4f} years"
        for tau, label in zip(taus, _expiry_labels(frame, taus))
    ]


def _surface_hover_text(
    maturity, moneyness, strike, iv, *,
    source_label=None, source=None, value_type=None,
):
    parts = [maturity, f"K / S: {moneyness:.4f}", f"strike: {strike:g}"]
    if source_label is not None:
        parts.append(f"{source_label}: {escape(str(source))}")
    if value_type is not None:
        parts.append(f"Surface value: {escape(str(value_type))}")
    parts.append(f"IV: {iv:.2%}")
    return "<br>".join(parts)


def _heatmap_customdata(long_frame, iv_grid, strike_grid):
    customdata = strike_grid.to_numpy(dtype=float)
    strike_hover = "strike: %{customdata:g}<br>"
    source_hover = ""
    value_type_hover = ""

    has_source = (
        "source" in long_frame.columns or "interpolation_method" in long_frame.columns
    )
    has_value_type = "surface_value_type" in long_frame.columns
    if not has_source and not has_value_type:
        return customdata, strike_hover, source_hover, value_type_hover

    fields = [customdata]
    if has_source:
        display_source = (
            long_frame["source"].copy()
            if "source" in long_frame.columns
            else pd.Series("Observed IV", index=long_frame.index).mask(
                long_frame["surface_value_type"].eq("Missing"), "missing"
            )
        )
        if has_value_type:
            display_source = display_source.mask(
                long_frame["surface_value_type"].eq("Interpolated"),
                long_frame["interpolation_method"]
                if "interpolation_method" in long_frame.columns
                else "1D interpolation within expiry",
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
    maturity_text = np.repeat(
        np.asarray(_maturity_hover(long_frame, iv_grid.index), dtype=object)[:, None],
        len(iv_grid.columns), axis=1,
    )

    trace = go.Heatmap(
        x=iv_grid.columns.to_numpy(dtype=float),
        y=y_values,
        z=iv_grid.to_numpy(dtype=float),
        customdata=customdata,
        text=maturity_text,
        connectgaps=False,
        xgap=1,
        ygap=1,
        colorbar={
            "title": {"text": "IV", "side": "top"},
            "tickformat": ".0%",
            "thickness": 14,
            "xpad": 8,
        },
        hovertemplate=(
            "%{text}<br>"
            "K / S: %{x:.4f}<br>"
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
        yaxis_title="Expiry (UTC)" if equal_expiry_spacing else "tau (years)",
        plot_bgcolor="rgba(128, 128, 128, 0.18)",
        height=430,
        margin={"l": 12, "r": 12, "t": 24, "b": 12},
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


def make_iv_surface_figure(
    long_frame: pd.DataFrame, *, compact_layout: bool = False
) -> go.Figure:
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
    visible_taus = _spaced_tau_ticks(
        iv_grid.index, minimum_spacing_fraction=0.30 if compact_layout else 0.18
    )
    maturity_text = _maturity_hover(long_frame, iv_grid.index)
    # Surface hover handles its native 2D text grid reliably. Keep customdata
    # for inspection, but avoid renderer-dependent compound-array indexing.
    hover_text = []
    for row_index, row in enumerate(customdata):
        text_row = []
        for column_index, cell in enumerate(row):
            values = cell if customdata.ndim == 3 else [cell]
            field = 1
            source_label = None
            source = None
            if source_hover:
                source_label = "Value source" if value_type_hover else "Combined source"
                source = values[field]
                field += 1
            text_row.append(_surface_hover_text(
                maturity_text[row_index], iv_grid.columns[column_index],
                values[0], iv_grid.iloc[row_index, column_index],
                source_label=source_label, source=source,
                value_type=values[field] if value_type_hover else None,
            ))
        hover_text.append(text_row)
    trace = go.Surface(
        x=iv_grid.columns.to_numpy(dtype=float),
        y=iv_grid.index.to_numpy(dtype=float),
        z=iv_grid.to_numpy(dtype=float),
        customdata=customdata,
        # Plotly's text validator stringifies nested Python lists row-by-row;
        # a 2D object array retains the per-cell shape through serialization.
        text=np.asarray(hover_text, dtype=object),
        hoverlabel={"align": "left"},
        connectgaps=False,
        colorscale="Viridis",
        colorbar={
            "title": {"text": "IV", "side": "top"},
            "tickformat": ".0%",
            "len": 0.72,
            "thickness": 18,
            "x": 1.02,
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
        # Native surface text lookup is row/column aware; hovertemplate's generic
        # array lookup can invert these indices, including for the text token.
        hoverinfo="text",
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
        if "source" in observed.columns:
            observed_customdata = observed[
                ["strike", "source", observed_column]
            ].to_numpy(dtype=object)
        observed_text = [
            _surface_hover_text(
                maturity, row.moneyness, row.strike, iv,
                source_label=(
                    "Value source" if value_type_hover else "Combined source"
                ) if source_hover else None,
                source=getattr(row, "source", "Observed IV"),
                value_type="Observed" if value_type_hover else None,
            )
            for maturity, row, iv in zip(
                _maturity_hover(observed, observed["tau"]),
                observed.itertuples(index=False), original_observed_iv,
            )
        ]
        figure.add_trace(
            go.Scatter3d(
                x=observed["moneyness"].to_numpy(dtype=float),
                y=observed["tau"].to_numpy(dtype=float),
                z=original_observed_iv + marker_lift,
                customdata=observed_customdata,
                text=observed_text,
                hoverlabel={"align": "left"},
                mode="markers",
                name="Observed IV",
                marker={
                    "size": 3,
                    "color": "#F97316",
                    "opacity": 1,
                    "line": {
                        "color": "rgba(15, 23, 42, 0.85)",
                        "width": 0.4,
                    },
                },
                hoverinfo="text",
            )
        )
    axis_style = {
        "showbackground": False,
        "gridcolor": "rgba(148, 163, 184, 0.18)",
        "zeroline": False,
        "showspikes": False,
        "tickfont": {"size": 11},
    }
    figure.update_layout(
        scene={
            "domain": {"x": [0, 1]},
            "bgcolor": "rgba(0, 0, 0, 0)",
            "xaxis": {"title": {"text": "K / S", "font": {"size": 12}}, **axis_style},
            "yaxis": {
                "title": {
                    "text": "" if compact_layout else "Expiry (UTC)",
                    "font": {"size": 12},
                },
                "tickmode": "array",
                "tickvals": visible_taus,
                "ticktext": [
                    label.replace("-", "<br>", 1)
                    for label in _expiry_labels(long_frame, visible_taus)
                ],
                **axis_style,
            },
            "zaxis": {
                # The IV colorbar and percentage ticks identify this axis;
                # omitting the duplicate title avoids edge clipping on rotation.
                "title": "",
                "tickformat": ".0%",
                **axis_style,
            },
            "camera": {
                "eye": (
                    {"x": 2.7, "y": 2.7, "z": 0.796}
                    if compact_layout else {"x": 1.9, "y": 1.9, "z": 0.56}
                ),
                "center": {"x": 0, "y": 0, "z": -0.06},
            },
            "aspectmode": "manual",
            "aspectratio": {"x": 1.7, "y": 1.2, "z": 0.72},
            "dragmode": "turntable",
        },
        autosize=True,
        height=430 if compact_layout else 640,
        margin={
            "l": 40 if compact_layout else 20,
            "r": 20 if compact_layout else 12,
            "t": 24,
            "b": 24,
        },
        paper_bgcolor="rgba(0, 0, 0, 0)",
        legend={
            "orientation": "h",
            "x": 0.02,
            "y": 0.98,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(0, 0, 0, 0)",
        },
        uirevision="iv-surface-camera-v18",
    )
    return figure


def make_smile_figure(
    long_frame: pd.DataFrame, selected_expiries=None
) -> go.Figure:
    """Create one IV smile trace per selected expiry."""
    _require_columns(long_frame, _LONG_FRAME_COLUMNS, "long_frame")

    selected = None if selected_expiries is None else set(selected_expiries)
    figure = go.Figure()
    observed_moneyness = []
    colors = [
        "#38BDF8", "#FB923C", "#C084FC", "#34D399", "#F472B6",
        "#FACC15", "#818CF8", "#2DD4BF", "#F87171", "#A3E635",
    ]
    for expiry_index, tau in enumerate(sorted(long_frame["tau"].dropna().unique())):
        if selected is not None and tau not in selected:
            continue
        smile = long_frame[long_frame["tau"] == tau].sort_values("moneyness")
        observed_moneyness.extend(
            smile.loc[np.isfinite(smile["iv"]), "moneyness"].to_numpy(dtype=float)
        )
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
                text=[_maturity_hover(smile, [tau])[0]] * len(smile),
                mode="lines+markers",
                line=dict(width=2, color=colors[expiry_index % len(colors)]),
                marker=dict(size=5),
                name=expiry_label,
                meta=tau,
                connectgaps=False,
                hovertemplate=(
                    "%{text}<br>"
                    "K / S: %{x:.4f}<br>"
                    + strike_hover
                    + source_hover
                    + "IV: %{y:.2%}<extra>%{fullData.name}</extra>"
                ),
            )
        )

    if observed_moneyness:
        lower, upper = min(observed_moneyness), max(observed_moneyness)
        padding = max((upper - lower) * 0.05, 0.01)
        x_range = [lower - padding, upper + padding]
    else:
        x_range = [0.8, 1.2]
    figure.add_vline(x=1.0, line_dash="dot", line_color="gray", line_width=1)
    figure.update_layout(
        xaxis_title="Moneyness (K / S)",
        yaxis_title="IV",
        height=430,
        margin=dict(l=12, r=12, t=65, b=12),
        legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom"),
        hovermode="closest",
    )
    figure.update_xaxes(range=x_range, tickformat=".2f", zeroline=False)
    figure.update_yaxes(tickformat=".0%", zeroline=False)
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
        text=_maturity_hover(data, data["tau"]),
        customdata=customdata,
        mode="lines+markers",
        connectgaps=False,
        hovertemplate=(
            "%{text}<br>"
            "IV: %{y:.2%}<br>"
            "strike: %{customdata[0]:g}<br>"
            "K / S: %{customdata[1]:.4f}<br>"
            "ATM distance: %{customdata[2]:.2%}<br>"
            "selection: %{customdata[3]}<extra></extra>"
        ),
    )
    figure = go.Figure(data=[trace])
    figure.update_layout(
        height=430,
        margin=dict(l=12, r=12, t=24, b=12),
        xaxis_title=(
            "Expiry (UTC) · spaced by maturity"
            if "expiry_datetime" in data.columns else "tau (years)"
        ),
        yaxis_title="IV",
    )
    ticks = _spaced_tau_ticks(data["tau"], minimum_spacing_fraction=0.18)
    figure.update_xaxes(
        tickmode="array",
        tickvals=ticks,
        ticktext=_expiry_labels(data, ticks),
        automargin=True,
    )
    figure.update_yaxes(tickformat=".0%")
    return figure
