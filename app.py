"""Offline BTC volatility dashboard. Run with streamlit run app.py."""

from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st

from iv_surface.fetcher import load_chain_snapshot, parse_chain_snapshot
from iv_surface.deribit import load_deribit_snapshot, parse_deribit_snapshot
from iv_surface.surface_builder import build_combined_surface_from_chain
from iv_surface.surface_transforms import (
    atm_term_structure_frame,
    interpolate_surface_2d,
    surface_to_long_frame,
)
from iv_surface.visualization import (
    make_atm_term_structure_figure,
    make_iv_heatmap,
    make_iv_surface_figure,
    make_smile_figure,
)


DATA_PATH = Path(__file__).resolve().parent / "data"
SNAPSHOT_SOURCES = {
    "Deribit": ("deribit_btc_options_snapshot.json", load_deribit_snapshot, parse_deribit_snapshot),
    "Bybit": ("bybit_btc_options_snapshot.json", load_chain_snapshot, parse_chain_snapshot),
}
MONEYNESS_RANGES = {
    "Core (0.8–1.2)": (0.8, 1.2),
    "Extended (0.7–1.3)": (0.7, 1.3),
}


def main():
    st.set_page_config(page_title="BTC volatility", layout="wide")
    st.markdown(
        """<style>
        .stMainBlockContainer {padding-top: 2rem; padding-bottom: 1rem;}
        .st-key-surface-responsive {container-type: inline-size;}
        .st-key-surface-narrow {display: none;}
        @container (max-width: 699px) {
            .st-key-surface-wide {display: none;}
            .st-key-surface-narrow {display: block;}
        }
        </style>""",
        unsafe_allow_html=True,
    )
    st.title("BTC volatility")
    source_name = st.sidebar.selectbox("Data source", list(SNAPSHOT_SOURCES), key="data_source")
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
        filename, load_snapshot, parse_snapshot = SNAPSHOT_SOURCES[source_name]
        snapshot = load_snapshot(DATA_PATH / filename)
        chain = parse_snapshot(snapshot)
    except (OSError, ValueError, TypeError, RuntimeError, AttributeError):
        st.error(f"Cannot read the frozen {source_name} snapshot. Check its file and schema.")
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
    st.caption(f"{source_name} BTC · Frozen snapshot: {captured_at:%Y-%m-%d %H:%M:%S} UTC")
    st.write(
        f"Shared spot: **${result.spot_price:,.2f}** · "
        f"Risk-free rate: **{rate_percent:.2f}% p.a.**, continuously compounded"
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
    with st.sidebar.expander("Combined IV coverage"):
        for label, count in zip(
            ["Preferred / averaged", "Fallback / one-sided", "Missing"], counts
        ):
            st.metric(label, f"{count:,}")
        st.caption(f"All {len(frame):,} expiry × strike grid cells.")
        st.caption(
            "Within 2% of spot, average same-strike Call and Put IV when both "
            "are available. Outside that band, prefer Put below spot and Call "
            "above spot."
        )
    if source_name == "Deribit":
        with st.sidebar.expander("Deribit pricing"):
            st.caption(
                "BTC bid/ask premiums are converted to USD using the snapshot's "
                "BTC/USD index. IV uses our spot Black-Scholes model and the "
                "selected rate; Deribit mark IV uses its own pricing inputs."
            )
    if not np.isfinite(frame["iv"]).any():
        st.warning("No IV solves succeeded at this rate. Try another rate assumption.")

    st.sidebar.subheader("Surface views")
    range_name = st.sidebar.radio(
        "Moneyness range",
        list(MONEYNESS_RANGES),
        horizontal=True,
        help=(
            "Applies to Heatmap and 3D surface. IV smiles show all observed strikes."
        ),
    )
    min_moneyness, max_moneyness = MONEYNESS_RANGES[range_name]
    surface_frame = interpolate_surface_2d(
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
    surface_summary = (
        "Visible range: "
        f"{surface_counts.get('Observed', 0):,} observed · "
        f"{surface_counts.get('Interpolated', 0):,} interpolated · "
        f"{surface_counts.get('Missing', 0):,} blank. "
        f"K / S: {min_moneyness:.1f}–{max_moneyness:.1f}."
    )
    with st.sidebar.expander("Surface interpolation"):
        st.write(
            "Convert observed IV to total variance (IV² × tau), linearly "
            "interpolate it between observations, then convert back to IV. "
            "K / S and tau are each scaled to their observed range for "
            "triangulation. Original IV points stay unchanged."
        )
        st.write(
            "The existing grid is filled only inside the convex hull of "
            "observations in the selected range. A cell can be filled across "
            "expiries, but cells outside that boundary stay blank. "
            "A denser grid would add display points, not market observations."
        )
        st.write(
            "In 3D, observed markers are lifted slightly above the mesh for "
            "visibility; hover reports their original IV."
        )

    heatmap_tab, surface_tab, smile_tab, atm_tab = st.tabs(
        ["Heatmap", "3D surface", "IV smiles", "ATM term structure"]
    )
    with heatmap_tab:
        heatmap_view = st.radio(
            "Heatmap values",
            ["Surface", "Observed"],
            horizontal=True,
            help=(
                "Surface uses 2D total-variance interpolation. Observed "
                "shows solved market IV without interpolation."
            ),
        )
        heatmap_frame = (
            surface_frame if heatmap_view == "Surface" else observed_frame
        )
        heatmap_counts = heatmap_frame["surface_value_type"].value_counts()
        st.caption(
            "Visible range: "
            f"{heatmap_counts.get('Observed', 0):,} observed · "
            f"{heatmap_counts.get('Interpolated', 0):,} interpolated · "
            f"{heatmap_counts.get('Missing', 0):,} blank. "
            f"K / S: {min_moneyness:.1f}–{max_moneyness:.1f}."
        )
        st.plotly_chart(
            make_iv_heatmap(heatmap_frame, equal_expiry_spacing=True),
            use_container_width=True,
        )
    with surface_tab:
        st.caption(surface_summary)
        st.caption("2D total-variance interpolation · observed IV overlaid as points.")
        with st.container(key="surface-responsive"):
            for compact, key in [(False, "surface-wide"), (True, "surface-narrow")]:
                with st.container(key=key):
                    st.plotly_chart(
                        make_iv_surface_figure(surface_frame, compact_layout=compact),
                        key=f"plot-{key}",
                        use_container_width=True,
                        config={"modeBarButtonsToRemove": ["resetCameraDefault3d"]},
                    )
        st.markdown(
            '<div aria-hidden="true" style="height: 32px"></div>',
            unsafe_allow_html=True,
        )
    with smile_tab:
        expiry_labels = {
            row.tau: row.expiry_datetime.strftime("%Y-%m-%d")
            for row in frame[["tau", "expiry_datetime"]]
            .dropna()
            .drop_duplicates("tau")
            .itertuples(index=False)
        }
        available_expiries = sorted(frame["tau"].unique())
        finite_expiries = sorted(
            frame.loc[np.isfinite(frame["iv"]), "tau"].unique()
        )
        with st.popover("Select expiries"):
            st.caption("Expiry (UTC)")
            selected_expiries = [
                tau for tau in available_expiries
                if st.checkbox(
                    expiry_labels.get(tau, f"tau={tau:g}"),
                    value=tau in finite_expiries[:3],
                    key=f"smile_expiry_{source_name}_{snapshot['captured_at_utc']}_{tau}",
                )
            ]
        st.caption("Observed Combined IV · all strikes · gaps preserved")
        if selected_expiries:
            st.plotly_chart(
                make_smile_figure(frame, selected_expiries=selected_expiries),
                use_container_width=True,
            )
        else:
            st.info("Select an expiry to show its IV smile.")
    with atm_tab:
        atm_frame = atm_term_structure_frame(frame)
        st.caption("Observed Combined IV nearest to spot within 2% · all expiries")
        st.plotly_chart(
            make_atm_term_structure_figure(atm_frame),
            use_container_width=True,
        )
        rejected_count = int(atm_frame["iv"].isna().sum())
        if rejected_count:
            st.info(
                f"{rejected_count} expiries have no usable ATM proxy and remain gaps. "
                "See selection details below."
            )
        with st.expander("ATM selection details"):
            st.caption(
                "Choose the nearest finite Combined IV within 2% of spot. "
                "Equally near strikes are averaged. Maturity spacing uses tau in years."
            )
            st.dataframe(
                atm_frame.rename(columns={"expiry_datetime": "Expiry (UTC)"}),
                hide_index=True,
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
