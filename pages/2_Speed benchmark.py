import os
from pathlib import Path
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import cast
import random as rn
import plotly.express as px
import pandas as pd
from PIL import Image, ImageDraw
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
import time
import supervision as sv
import onnxruntime as ort
import numpy as np
import torchvision.transforms.functional as F
import plotly.graph_objects as go




working_path = Path.cwd()
content_path = working_path / "content"

st.set_page_config(
    page_title="Speed Benchmark",
    page_icon="🏃🏽‍♀️‍➡️",
    layout="wide",
)

st.title("Speed Benchmarks")
st.caption(
    "Comparison of the best validation result for each training run."
)


########################################
# Load benchmark data
#########################################

# Load benchmark results
benchmark_path = (
    working_path
    / "maize_dataset"
    / "runs"
    / "inference_benchmark_summary.csv"
)

benchmark_df = pd.read_csv(
    benchmark_path
)

# Load evaluation results
results_path = (
    working_path
    / "results_full.parquet"
)

results_df = pd.read_parquet(
    results_path
)


# ------------------------------------------------------------
# Normalize data for merging
# ------------------------------------------------------------

# Normalize framework names in evaluation results
results_df["framework"] = (
    results_df["framework"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({
        "yolo26": "yolo",
        "rf-detr": "detr",
        "rfdetr": "detr",
    })
)

# Normalize framework names in benchmark results
benchmark_df["framework"] = (
    benchmark_df["framework"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Extract model size from run name
results_df["model_size"] = (
    results_df["run"]
    .astype(str)
    .str.strip()
    .str[0]
    .str.lower()
)

benchmark_df["model_size"] = (
    benchmark_df["model_size"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ------------------------------------------------------------
# Select best evaluation result
# ------------------------------------------------------------

best_results = (
    results_df
    .dropna(subset=["mAP50_95"])
    .sort_values(
        "mAP50_95",
        ascending=False,
    )
    .groupby(
        [
            "framework",
            "model_size",
        ],
        as_index=False,
    )
    .first()
)

best_results = best_results[
    [
        "framework",
        "model_size",
        "mAP50_95",
        "mAP50",
        "precision",
        "recall",
    ]
]


# ------------------------------------------------------------
# Merge latency and evaluation results
# ------------------------------------------------------------

plot_df = benchmark_df.merge(
    best_results,
    on=[
        "framework",
        "model_size",
    ],
    how="inner",
)


# ------------------------------------------------------------
# Create labels
# ------------------------------------------------------------

plot_df["Backend"] = (
    plot_df["backend"]
    .map({
        "torch": "PyTorch",
        "onnx": "ONNX",
    })
)

plot_df["Framework"] = (
    plot_df["framework"]
    .map({
        "yolo": "YOLO",
        "detr": "RF-DETR",
    })
)

plot_df["Model"] = (
    plot_df["Framework"]
    + " "
    + plot_df["model_size"].str.upper()
    + " · "
    + plot_df["Backend"]
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

# ------------------------------------------------------------
# Visual encoding
# ------------------------------------------------------------

plot_df["model_group"] = (
    plot_df["Framework"]
    + "_"
    + plot_df["model_size"]
)

# Framework = color hue
# Model size = color intensity
color_map = {
    "YOLO_n": "#FAD7B5",
    "YOLO_s": "#F39C4A",
    "YOLO_m": "#D95F02",

    "RF-DETR_n": "#BFD7EA",
    "RF-DETR_s": "#5B9BD5",
    "RF-DETR_m": "#1565C0",
}

# Backend = marker shape
symbol_map = {
    "PyTorch": "circle",
    "ONNX": "x",
}


# ------------------------------------------------------------
# Scatter plot
# ------------------------------------------------------------

fig = px.scatter(
    plot_df,
    x="mean_ms",
    y="mAP50_95",

    color="model_group",
    color_discrete_map=color_map,

    symbol="Backend",
    symbol_map=symbol_map,

    #text="Model",

    hover_data={
        "model_group": False,
        "Framework": True,
        "model_size": True,
        "Backend": True,
        "mean_ms": ":.1f",
        "median_ms": ":.1f",
        "p95_ms": ":.1f",
        "fps": ":.1f",
        "mAP50_95": ":.3f",
        "mAP50": ":.3f",
        "precision": ":.3f",
        "recall": ":.3f",
    },

    labels={
        "mean_ms": "Mean inference latency (ms)",
        "mAP50_95": "mAP50-95",
        "model_size": "Model size",
    },

    title="Inference latency vs. mAP50-95",
)


fig.update_traces(
    marker=dict(
        size=13,
    ),
    textposition="top center",
    showlegend=False,
)


# ------------------------------------------------------------
# Legend 1: Framework = color hue
# ------------------------------------------------------------

framework_colors = {
    "RF-DETR": "#1565C0",
    "YOLO": "#D95F02",
}

for framework_name, color in framework_colors.items():
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=framework_name,
            marker=dict(
                color=color,
                size=11,
                symbol="circle",
            ),
            showlegend=True,
            legend="legend",
        )
    )


# ------------------------------------------------------------
# Legend 2: Model size = color intensity
# ------------------------------------------------------------

size_colors = {
    "Medium": "#4D4D4D",
    "Small": "#999999",
    "Nano": "#D9D9D9",
}

for size_name, color in size_colors.items():
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=size_name,
            marker=dict(
                color=color,
                size=11,
                symbol="circle",
            ),
            showlegend=True,
            legend="legend2",
        )
    )


# ------------------------------------------------------------
# Legend 3: Backend = marker shape
# ------------------------------------------------------------

backend_symbols = {
    "PyTorch": "circle",
    "ONNX": "x",
}

for backend_name, symbol in backend_symbols.items():
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=backend_name,
            marker=dict(
                color="#404040",
                size=11,
                symbol=symbol,
            ),
            showlegend=True,
            legend="legend3",
        )
    )


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

fig.update_layout(
    height=650,

    xaxis_title="Mean inference latency (ms)",
    yaxis_title="mAP50-95",

    margin=dict(
        l=20,
        r=180,
        t=60,
        b=20,
    ),

    legend=dict(
        title="Framework",
        x=1.02,
        y=1.0,
    ),

    legend2=dict(
        title="Model size",
        x=1.02,
        y=0.65,
    ),

    legend3=dict(
        title="Backend",
        x=1.02,
        y=0.30,
    ),

    legend_itemclick=False,
    legend_itemdoubleclick=False,
)


fig.update_xaxes(
    showgrid=True,
    title_font=dict(size=16),
    tickfont=dict(size=14),
)


fig.update_yaxes(
    showgrid=True,
    title_font=dict(size=16),
    tickfont=dict(size=14),
)




# ------------------------------------------------------------
# Connect model sizes: N -> S -> M
# ------------------------------------------------------------
# ------------------------------------------------------------
# Connect model sizes with color gradients: N -> S -> M
# ------------------------------------------------------------

def hex_to_rgb(hex_color):
    # Convert hex color to RGB tuple
    hex_color = hex_color.lstrip("#")
    return tuple(
        int(hex_color[i:i + 2], 16)
        for i in (0, 2, 4)
    )


def interpolate_color(color_start, color_end, fraction):
    # Interpolate between two hex colors
    rgb_start = hex_to_rgb(color_start)
    rgb_end = hex_to_rgb(color_end)

    rgb = tuple(
        int(
            start + (end - start) * fraction
        )
        for start, end in zip(
            rgb_start,
            rgb_end,
        )
    )

    return f"rgb{rgb}"


size_order = {
    "n": 0,
    "s": 1,
    "m": 2,
}

line_dashes = {
    "PyTorch": "solid",
    "ONNX": "dash",
}

n_segments = 30


for (framework, backend), group in plot_df.groupby(
    ["Framework", "Backend"]
):

    group = (
        group
        .assign(
            size_order=group["model_size"].map(size_order)
        )
        .sort_values("size_order")
        .reset_index(drop=True)
    )

    # Draw N -> S and S -> M separately
    for i in range(len(group) - 1):

        row_start = group.iloc[i]
        row_end = group.iloc[i + 1]

        color_start = color_map[
            f"{framework}_{row_start['model_size']}"
        ]

        color_end = color_map[
            f"{framework}_{row_end['model_size']}"
        ]

        # Split connection into small colored segments
        for segment in range(n_segments):

            t1 = segment / n_segments
            t2 = (segment + 1) / n_segments
            t_mid = (t1 + t2) / 2

            x1 = (
                row_start["mean_ms"]
                + (
                    row_end["mean_ms"]
                    - row_start["mean_ms"]
                ) * t1
            )

            x2 = (
                row_start["mean_ms"]
                + (
                    row_end["mean_ms"]
                    - row_start["mean_ms"]
                ) * t2
            )

            y1 = (
                row_start["mAP50_95"]
                + (
                    row_end["mAP50_95"]
                    - row_start["mAP50_95"]
                ) * t1
            )

            y2 = (
                row_start["mAP50_95"]
                + (
                    row_end["mAP50_95"]
                    - row_start["mAP50_95"]
                ) * t2
            )

            segment_color = interpolate_color(
                color_start,
                color_end,
                t_mid,
            )

            fig.add_trace(
                go.Scatter(
                    x=[x1, x2],
                    y=[y1, y2],
                    mode="lines",
                    line=dict(
                        color=segment_color,
                        width=2.5,
                        dash=line_dashes[backend],
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

st.space("medium")

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": True,
        "scrollZoom": True,
    },
)




