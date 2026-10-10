def render_live_tab(pipe: Pipeline, symbol: str) -> None:
    """The **Live** tab: "Where this usually goes next" for 1-minute data.

    This tab identifies historical patterns matching recent price action
    and projects the most likely subsequent movement. It is only available
    for 1-minute resolution data.
    """
    # Requirement 1: Resolution Check & Guardrails
    if _tf().key != "1m":
        st.info("Live projection is only available for 1-minute resolution data.")
        return

    # Requirement 2: User Configuration (UI Controls)
    # Use a sidebar for controls to keep the main area for the chart.
    with st.sidebar:
        st.subheader("Live Projection Settings")
        # Requirement 2.1: Recent Bars
        # Range: 100 to 600, Step: 100, Default: 300
        recent_bars = st.slider(
            "Recent bars",
            min_value=100, max_value=600,
            step=100, value=300,
            help="How many of the most recent bars to use as the pattern to match."
        )
        # Requirement 2.2: Projection Bars
        # Range: 100 to 600, Step: 100, Default: 300
        projection_bars = st.slider(
            "Projection bars",
            min_value=100, max_value=600,
            step=100, value=300,
            help="How many bars into the future to project the historical match."
        )

    # Requirement 3: Analysis Logic (Backend)
    # Requirement 3.1: Extract most recent N bars.
    # forecast_path_for handles window=None by using the most recent 'length' bars.
    # Requirement 3.2 & 3.3: Invoke forecast_path_for.
    # We use symbol as the cache_key.
    path = forecast_path_for(
        cache_key=f"live_{symbol}_{recent_bars}_{projection_bars}",
        pipe=pipe,
        length=recent_bars,
        horizon=projection_bars,
        k=FORECAST_PATH_MATCHES,
        amplitude_weight=M.DEFAULT_AMPLITUDE_WEIGHT,
        window=None,
    )

    if path is None:
        st.warning("Could not find a matching historical pattern for the current price action.")
        return

    # Requirement 4: Visualization (Frontend)
    # Requirement 4.1 & 4.2: Build forecast path figure.
    # build_forecast_path_figure takes (pipe, path, history_bars=..., window=...)
    # When window=None, it uses archive's last 'history_bars'.
    # We want history_bars to match 'recent_bars'.
    fig = build_forecast_path_figure(
        pipe,
        path,
        history_bars=recent_bars,
        title=f"Live Projection for {symbol}"
    )

    # Requirement 4.3: Render using st.plotly_chart.
    st.plotly_chart(fig, use_container_width=True)
