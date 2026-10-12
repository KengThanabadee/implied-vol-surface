"""Offline BTC volatility dashboard. Run with streamlit run app.py."""

from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st

from iv_surface.fetcher import load_chain_snapshot, parse_chain_snapshot
from iv_surface.surface_builder import build_combined_surface_from_chain
from iv_surface.surface_transforms import (
    interpolate_smiles_1d,
    surface_to_long_frame,
)
from iv_surface.visualization import make_iv_heatmap, make_iv_surface_figure


SNAPSHOT_PATH = (
    Path(__file__).resolve().parent / "data/bybit_btc_options_snapshot.json"
)
MONEYNESS_RANGES = {
    "Core (0.8–1.2)": (0.8, 1.2),
    "Extended (0.7–1.3)": (0.7, 1.3),
}


def main():
    st.set_page_config(page_title="BTC volatility", layout="wide")
    st.title("BTC volatility")
    rate_percent = st.sidebar.number_input(
        "Risk-free rate (% p.a., continuously compounded)",
        value=0.00,
        step=0.25,
        format="%.2f",
    )
    if not np.isfinite(rate_percent):
        st.error("Enter a finite risk-free rate.")
        return

    # Catch input errors at the loading boundary, not errors in chart rendering.
    try:
        snapshot = load_chain_snapshot(SNAPSHOT_PATH)
        chain = parse_chain_snapshot(snapshot)
    except (OSError, ValueError, TypeError, RuntimeError, AttributeError):
        st.error("Cannot read the frozen Bybit snapshot. Check its file and schema.")
        return

    try:
        result = build_combined_surface_from_chain(chain, r=rate_percent / 100)
    except ValueError:
        st.error(
            "Cannot build IV: the snapshot needs valid, unique contracts and "
            "usable bid/ask quotes."
        )
        return

    expiry_dates = chain.drop_duplicates("tau").set_index("tau")["expiry_dt"]
    frame = surface_to_long_frame(
        result.iv_surface,
        result.expiries,
        result.strikes,
        result.spot_price,
        expiry_datetimes=expiry_dates.reindex(result.expiries).tolist(),
        source_grid=result.source_grid,
    )
    captured_at = datetime.fromisoformat(
        snapshot["captured_at_utc"].replace("Z", "+00:00")
    )
    st.caption(f"Bybit BTC · Frozen snapshot: {captured_at:%Y-%m-%d %H:%M:%S} UTC")
    st.write(
        f"Shared spot: **${result.spot_price:,.2f}** · "
        f"Risk-free rate: **{rate_percent:.2f}% p.a.**, continuously compounded"
    )
    st.subheader("Combined IV surface")
    st.caption(
        "Within 2% of spot, average same-strike Call and Put IV when both are available. "
        "Outside that band, prefer Put below spot and Call above spot."
    )
    source = frame["source"]
    counts = (
        int(source.isin(["call", "put", "near_atm_average"]).sum()),
        int(
            source.isin(
                [
                    "call_fallback",
                    "put_fallback",
                    "near_atm_call_only",
                    "near_atm_put_only",
                ]
            ).sum()
        ),
        int(source.eq("missing").sum()),
    )
    for column, label, count in zip(
        st.columns(3),
        ["Preferred / averaged", "Fallback / one-sided", "Missing"],
        counts,
    ):
        column.metric(label, f"{count:,}")
    st.caption(f"Counts cover all {len(frame):,} expiry × strike grid cells.")
    if not np.isfinite(frame["iv"]).any():
        st.warning("No IV solves succeeded at this rate. Try another rate assumption.")

    range_name = st.radio(
        "Moneyness range",
        list(MONEYNESS_RANGES),
        horizontal=True,
        help=(
            "Core keeps the denser center of the surface. Extended shows more "
            "of the wings, where missing cells are expected."
        ),
    )
    min_moneyness, max_moneyness = MONEYNESS_RANGES[range_name]
    surface_frame = interpolate_smiles_1d(
        frame,
        min_moneyness=min_moneyness,
        max_moneyness=max_moneyness,
    )
    observed_frame = surface_frame.copy()
    observed_frame["iv"] = observed_frame["observed_iv"]
    observed_frame["surface_value_type"] = np.where(
        np.isfinite(observed_frame["observed_iv"]), "Observed", "Missing"
    )
    surface_counts = surface_frame["surface_value_type"].value_counts()
    st.caption(
        "Visible range: "
        f"{surface_counts.get('Observed', 0):,} observed · "
        f"{surface_counts.get('Interpolated', 0):,} interpolated · "
        f"{surface_counts.get('Missing', 0):,} blank. "
        f"Displayed K / S: {min_moneyness:.1f}–{max_moneyness:.1f}. "
        "Interpolation is linear within each expiry only; wings are not "
        "extrapolated."
    )

    heatmap_tab, surface_tab = st.tabs(["Heatmap", "3D surface"])
    with heatmap_tab:
        heatmap_view = st.radio(
            "Heatmap values",
            ["Surface", "Observed"],
            horizontal=True,
            help=(
                "Surface fills only interior gaps within each expiry. Observed "
                "shows solved market IV without interpolation."
            ),
        )
        heatmap_frame = (
            surface_frame if heatmap_view == "Surface" else observed_frame
        )
        st.plotly_chart(
            make_iv_heatmap(heatmap_frame, equal_expiry_spacing=True),
            use_container_width=True,
        )
    with surface_tab:
        st.caption(
            "Uses the same interpolated values as the Surface heatmap; tau remains "
            "the numerical maturity axis. Observed markers are lifted slightly "
            "above the mesh for visibility; hover reports their original IV."
        )
        st.plotly_chart(
            make_iv_surface_figure(surface_frame), use_container_width=True
        )


if __name__ == "__main__":
    main()
