from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import base64

PROJECT_DIR = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Impact experiments",
    page_icon="🧪",
    layout="wide",
)

st.title("Impact experiments")
st.caption("Impact of augmentation and input resolution when training **YOLO nano**.")

st.markdown("""
<style>

/* Top control cards */
div[data-testid="stHorizontalBlock"]:has(
    div[data-testid="stSlider"]
    div[data-testid="stRadio"]
) > div[data-testid="stColumn"] {
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 8px;
    padding: 0.8rem 1rem;
}

/* Metric cards */
div[data-testid="stMetric"] {
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 8px;
    padding: 0.8rem 1rem;
}

/* Radio cards */
div[data-testid="stRadio"] {
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 8px;
    padding: 0.8rem 1rem;
}

<style>
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: rgba(128, 228, 128, 0.04);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

df = pd.read_parquet(Path.cwd() / "results_full.parquet")


df["framework"] = df["framework"].replace({
    "YOLO26": "YOLO",
})

df = df[df["framework"] == "YOLO"].copy()


# -------------------------------------------------------------------
# Best epoch per run
# Same criterion as Benchmarks.py: highest mAP50-95
# -------------------------------------------------------------------

best_idx = (
    df.groupby("run")["mAP50_95"]
    .idxmax()
)

df_best = (
    df.loc[best_idx]
    .reset_index(drop=True)
)


# -------------------------------------------------------------------
# Metric selector
# -------------------------------------------------------------------

metric_labels = {
    "mAP50_95": "mAP50-95",
    "mAP50": "mAP50",
    "precision": "Precision",
    "recall": "Recall",
}

metric = st.radio(
    "Metric",
    options=list(metric_labels),
    format_func=lambda x: metric_labels[x],
    index=0,
    horizontal=True,
)

# -------------------------------------------------------------------
# Augmentation: all experiments in one plot
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Select experiments
# -------------------------------------------------------------------

std_params = {
    "mosaic": 1.0,
    "flipud": 0.0,
    "hsv_v": 0.4,
    "imgsz": 640,
}

# Default
mask_default = (
    df_best["run"].eq("n")
    & df_best["mosaic"].eq(std_params["mosaic"])
    & df_best["flipud"].eq(std_params["flipud"])
    & df_best["hsv_v"].eq(std_params["hsv_v"])
    & df_best["imgsz"].eq(std_params["imgsz"])
)

# Mosaic: only mosaic differs from the default
mask_mosaic = (
    df_best["mosaic"].ne(std_params["mosaic"])
    & df_best["flipud"].eq(std_params["flipud"])
    & df_best["hsv_v"].eq(std_params["hsv_v"])
    & df_best["imgsz"].eq(std_params["imgsz"])
)

# FlipUD: only flipud differs from the default
mask_flipud = (
    df_best["mosaic"].eq(std_params["mosaic"])
    & df_best["flipud"].ne(std_params["flipud"])
    & df_best["hsv_v"].eq(std_params["hsv_v"])
    & df_best["imgsz"].eq(std_params["imgsz"])
)

# Brightness: only hsv_v differs from the default
mask_hsvv = (
    df_best["mosaic"].eq(std_params["mosaic"])
    & df_best["flipud"].eq(std_params["flipud"])
    & df_best["hsv_v"].ne(std_params["hsv_v"])
    & df_best["imgsz"].eq(std_params["imgsz"])
)

# Combined augmentation: at least two parameters differ from the default
n_changed = (
    df_best["mosaic"].ne(std_params["mosaic"]).astype(int)
    + df_best["flipud"].ne(std_params["flipud"]).astype(int)
    + df_best["hsv_v"].ne(std_params["hsv_v"]).astype(int)
)

mask_aug_combined = (
    df_best["imgsz"].eq(std_params["imgsz"])
    & n_changed.ge(2)
)

# Resolution: default augmentations, but different input resolution
mask_resolution = (
    df_best["mosaic"].eq(std_params["mosaic"])
    & df_best["flipud"].eq(std_params["flipud"])
    & df_best["hsv_v"].eq(std_params["hsv_v"])
    & df_best["imgsz"].ne(std_params["imgsz"])
)

# Collect experiment masks using the final plot labels
masks = {
    "Default": mask_default,
    "Mosaic": mask_mosaic,
    "Vertical Flip": mask_flipud,
    "Brightness": mask_hsvv,
    "Combined": mask_aug_combined,
    "1280 x 720": mask_resolution,
}

# Get the best run for each experiment
best_idx = {
    name: df_best.loc[mask, "mAP50_95"].idxmax()
    for name, mask in masks.items()
}

# Select the best runs
plot_df = df_best.loc[list(best_idx.values())].copy()
plot_df["experiment"] = list(best_idx.keys())

# Define plot order
order = [
    "Default",
    "Mosaic",
    "Vertical Flip",
    "Brightness",
    "Combined",
    "1280 x 720",
]

plot_df["experiment"] = pd.Categorical(
    plot_df["experiment"],
    categories=order,
    ordered=True,
)

plot_df = plot_df.sort_values("experiment")

# Normalize all experiments relative to the default run
base_value = plot_df.loc[
    plot_df["experiment"] == "Default",
    metric,
].iloc[0]

plot_df["relative"] = plot_df[metric] / base_value * 100

# omit Default bar
plot_df.drop(plot_df.index[plot_df["experiment"] == "Default"], inplace=True)

# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------

#st.subheader("Augmentation & resolution")

fig = go.Figure(
    go.Bar(marker_color="#FAD7B5",
        x=plot_df["experiment"],
        y=plot_df["relative"],
        text=plot_df["relative"].map(lambda x: f"{x:.1f}%"),
        textposition="outside",
        customdata=plot_df[["run", metric]],
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Relative: %{y:.1f}%<br>"
            + metric_labels[metric]
            + ": %{customdata[1]:.3f}<br>"
            "Run: %{customdata[0]}"
            "<extra></extra>"
        ),
    )
)

fig.add_hline(
    y=100,
    line_dash="dash",
    annotation_text="Default = 100%",
)

fig.update_layout(
    height=450,
    showlegend=False,
    bargap=0.35,
    margin=dict(l=20, r=20, t=20, b=20),
)

fig.update_yaxes(
    title="Metric relative to YOLO26 nano",
    title_font=dict(size=16),
    range=[
        plot_df["relative"].min() - 2,
        plot_df["relative"].max() + 2,
    ],
    tickfont=dict(size=14),
    ticksuffix="%",
)

fig.update_xaxes(title=None,     
                 tickfont=dict(size=16))

fig.update_traces(
    textfont=dict(size=16)
    )



# Map each experiment to its icon

# Load experiment icons
def load_svg(path):
    # Convert SVG file to a data URI for Plotly
    svg = Path(path).read_bytes()
    encoded = base64.b64encode(svg).decode()
    return f"data:image/svg+xml;base64,{encoded}"


icon_map = {
    "Mosaic": load_svg(PROJECT_DIR / "assets" / "mosaic.svg"),
    "Vertical Flip": load_svg(PROJECT_DIR / "assets" / "flip.svg"),
    "Brightness": load_svg(PROJECT_DIR / "assets" / "brightness.svg"),
    "Combined": load_svg(PROJECT_DIR / "assets" / "combined.svg"),
    "1280 x 720": load_svg(PROJECT_DIR / "assets" / "resolution.svg"),
}

# Add experiment icons below the x-axis labels
experiments = plot_df["experiment"].astype(str).tolist()

for i, experiment in enumerate(experiments):
    if experiment == "Default":
        continue

    fig.add_layout_image(
        dict(
            source=icon_map[experiment],
            x=(i + 0.5) / len(experiments),
            y=-0.12,
            xref="paper",
            yref="paper",
            sizex=0.10,
            sizey=0.10,
            xanchor="center",
            yanchor="top",
            sizing="contain",
            layer="above",
        )
    )

fig.update_layout(
    height=520,
    showlegend=False,
    bargap=0.35,
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=100,
    ),
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"displayModeBar": False},
)
