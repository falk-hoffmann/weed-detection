# pages/2_Model_Comparison.py

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go


st.set_page_config(
    page_title="Model Quality",
    page_icon="📊",
    layout="wide",
)

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


st.title("Model Quality")
st.caption(
    "Comparison of the best validation results for each training run."
)


# -------------------------------------------------------------------
# Load combined dataframe 
# -------------------------------------------------------------------


df_results_full = pd.read_parquet(Path.cwd() /"results_full.parquet")

df_plot = df_results_full.copy()


# -------------------------------------------------------------------
# Normalize framework
# -------------------------------------------------------------------

df_plot["framework"] = df_plot["framework"].replace({
    "YOLO26": "YOLO",
})


# -------------------------------------------------------------------
# Unified batch size
# -------------------------------------------------------------------

df_plot["batch_size"] = np.where(
    df_plot["framework"].eq("YOLO"),
    df_plot["batch"],
    df_plot["train_config.batch_size"],
)


# -------------------------------------------------------------------
# Model size
# -------------------------------------------------------------------

def get_model_size(row):

    if row["framework"] == "YOLO":
        model = str(row.get("model", "")).lower()

        if "26n" in model:
            return "n"
        if "26s" in model:
            return "s"
        if "26m" in model:
            return "m"

    elif row["framework"] == "RF-DETR":
        model = str(row.get("model_config.model_name", "")).lower()

        if "nano" in model:
            return "n"
        if "small" in model:
            return "s"
        if "medium" in model:
            return "m"

    return "other"


df_plot["model_size"] = df_plot.apply(get_model_size, axis=1)


# Only n / s / m (no base, since it is redundant because results are comparable to medium)
df_plot = df_plot[
    df_plot["model_size"].isin(["n", "s", "m"])
].copy()


# -------------------------------------------------------------------
# Omit the augmented YOLO runs (that will be displayed on another page)
# -------------------------------------------------------------------

df_plot = df_plot[
    ~df_plot["run"].str.startswith("n_aug_", na=False)
].copy()


# -------------------------------------------------------------------
# Omit the DETR runs with batchsize other than 16
# differences to 4 and 16 will be displayed on another page (train speed)
# in terms of model performance, there is no difference
# -------------------------------------------------------------------

df_plot = df_plot[
    df_plot["batch_size"] == 16
].copy()



# -------------------------------------------------------------------
# Select best epoch per run
# Criterion: highest mAP50-95
# -------------------------------------------------------------------

best_idx = (
    df_plot
    .groupby(["framework", "run"])["mAP50_95"]
    .idxmax()
)

df_best = (
    df_plot
    .loc[best_idx]
    .reset_index(drop=True)
)


# -------------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------------

st.sidebar.header("Filters")

frameworks = st.sidebar.multiselect(
    "Framework",
    options=sorted(df_best["framework"].dropna().unique()),
    default=sorted(df_best["framework"].dropna().unique()),
)

model_sizes = st.sidebar.multiselect(
    "Model size",
    options=["n", "s", "m"],
    default=["n", "s", "m"],
)

batch_sizes = st.sidebar.multiselect(
    "Batch size",
    options=sorted(df_best["batch_size"].dropna().unique()),
    default=sorted(df_best["batch_size"].dropna().unique()),
)


df_filtered = df_best[
    df_best["framework"].isin(frameworks)
    & df_best["model_size"].isin(model_sizes)
    & df_best["batch_size"].isin(batch_sizes)
].copy()


df_filtered["model_group"] = (
    df_filtered["framework"]
    + "_"
    + df_filtered["model_size"]
)

# model resolution is stored in two different columns for YOLO / DETR. Combine them into column "resolution"
df_filtered["resolution"] = df_filtered["imgsz"].fillna(
    df_filtered["model_config.resolution"]
)
df_filtered["resolution"] = df_filtered["resolution"].astype("Int64")

df_filtered["model_group"] = (
    df_filtered["framework"]
    + "_"
    + df_filtered["model_size"]
)

# axis-Labels: like "YOLO small"
size_labels = {"n": "nano", "s": "small", "m": "medium"}

df_filtered["model_group_resolution_label"] = (
    df_filtered["framework"]
    + " "
    + df_filtered["model_size"].map(size_labels)
    + " ("
    + df_filtered["resolution"].astype(str)
    + ")"
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

st.space("medium")

metric = st.radio(
    "Metric",
    options=list(metric_labels),
    format_func=lambda x: metric_labels[x],
    index=0,
    horizontal=True,
)

df_filtered = df_filtered.sort_values(metric, ascending=False)
order = df_filtered["model_group_resolution_label"].tolist()

# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------

color_map = {
    # YOLO: Blau
    "YOLO_n": "#FAD7B5",
    "YOLO_s": "#F39C4A",
    "YOLO_m": "#D95F02",

    # RF-DETR: Orange
    "RF-DETR_n": "#BFD7EA",
    "RF-DETR_s": "#5B9BD5",
    "RF-DETR_m": "#1565C0"
}
# color_map = {
#     # YOLO: Blau
#     "YOLO nano (640)": "#BFD7EA",
#     "YOLO nano (1280)": "#29DC32",
#     "YOLO small (640)": "#5B9BD5",
#     "YOLO medium (640)": "#1565C0",

#     # RF-DETR: Orange
#     "RF-DETR nano 384": "#FAD7B5",
#     "RF-DETR small 512": "#F39C4A",
#     "RF-DETR medium 576": "#D95F02",
# }

fig = px.bar(
    df_filtered,
    y="model_group_resolution_label",
    x=metric,

    #color="framework",
    color="model_group",
    color_discrete_map=color_map,

    #barmode="group",
    #pattern_shape="model_size",
    hover_name="run",
    hover_data={
        "framework": True,
        "model_size": True,
        "batch_size": True,
        "epoch": False,
        "precision": ":.2f",
        "recall": ":.2f",
        "mAP50": ":.2f",
        "mAP50_95": ":.2f",
        "model_group_resolution_label": False
    },
    labels={
        "run": "Training Run",
        metric: metric_labels[metric],
        "framework": "Framework",
        "model_size": "Model Size",
    },
)

# Legende der eigentlichen Traces ausblenden
fig.update_traces(showlegend=False)


# --------------------------------------------------
# Legend 1: Framework = Farbton
# --------------------------------------------------

framework_colors = {
    "RF-DETR": "#1565C0",
    "YOLO26": "#D95F02",
    
}

for framework, color in framework_colors.items():
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            name=framework,
            marker_color=color,
            showlegend=True,
            legend="legend",
        )
    )


# --------------------------------------------------
# Legend 2: Model size = Farbsättigung
# neutrale Graustufen
# --------------------------------------------------

size_colors = {
   
    "medium": "#4D4D4D",
    "small": "#999999",
    "nano": "#D9D9D9",
}

for size, color in size_colors.items():
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            name=size,
            marker_color=color,
            showlegend=True,
            legend="legend2",
        )
    )


fig.update_layout(
    legend=dict(
        title="Framework",
        x=1.02,
        y=1.0,
    ),
    legend2=dict(
        title="Model size",
        x=1.02,
        y=0.5,
    ),
)

fig.update_layout(
    xaxis_tickangle=-45,
    legend_title_text="Framework",
    height=max(400, len(df_filtered) * 55),
    #height=650,
    bargap=0.2,
    legend_itemclick=False,
    legend_itemdoubleclick=False,
)

fig.update_xaxes(
    tickfont=dict(size=14),
    title_font=dict(size=16),
    showgrid=True,
    autorange=True,
    fixedrange=False,
)

fig.update_yaxes(
    title_text="Model (train. resolution)",
    autorange=True,
    fixedrange=False,
    title_font=dict(size=16),
    tickfont=dict(size=14),
    categoryorder="array",
    categoryarray=order[::-1],
)



# st.plotly_chart(
#     fig,
#     use_container_width=True,
#     config={
#     "displayModeBar": True,
#     "scrollZoom": True,
#     },
# )

event = st.plotly_chart(
    fig,
    use_container_width=True,
    on_select="rerun",
    selection_mode="points",
    key="model_comparison_chart",
    config={
        "displayModeBar": True,
        "scrollZoom": True,
    },
)

# -------------------------------------------------------------------
# Result table
# -------------------------------------------------------------------

st.subheader("Best results")

result_columns = [
    "framework",
    "model_size",
    #"batch_size",
    "run",
    "resolution",
    "precision",
    "recall",
    "mAP50",
    "mAP50_95",
]

result_table = (
    df_filtered[result_columns]
    .sort_values("mAP50_95", ascending=False)
    .reset_index(drop=True)
)

selected_run = None

if event.selection.points:
    selected_run = event.selection.points[0].get("hovertext")


def style_table(row):
    styles = []

    for col in row.index:
        style = ""

        # ausgewählte Zeile hervorheben
        if row["run"] == selected_run:
            style += "background-color: rgba(255, 100, 100, 0.15);"

        # Framework + Model Size + resolution grau
        elif col in ["framework", "model_size", "resolution"]:
            style += "background-color: rgba(170, 170,170, 0.15);"

        styles.append(style)

    return styles


styled_table = result_table.style.apply(style_table, axis=1)

st.dataframe(
    styled_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "run": None,
        "precision": st.column_config.NumberColumn(
            "Precision",
            format="%.3f",
        ),
        "recall": st.column_config.NumberColumn(
            "Recall",
            format="%.3f",
        ),
        "mAP50": st.column_config.NumberColumn(
            "mAP50",
            format="%.3f",
        ),
        "mAP50_95": st.column_config.NumberColumn(
            "mAP50-95",
            format="%.3f",
        ),
    },
)