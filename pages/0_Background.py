import streamlit as st
from textwrap import dedent

st.set_page_config(
    page_title="Precision Farming · Weed Detection",
    page_icon="🌱",
    layout="wide",
)


# -------------------------------------------------------------------
# Image sources
# -------------------------------------------------------------------

ROBOT_IMAGE = (
    "robot_image.jpg"
)

WORKFLOW_IMAGE = (
    "Process-flow.png"
)


# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1400px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        text-align: center;
        padding: 1rem 1rem 2rem 1rem;
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        font-size: 1.35rem;
        opacity: 0.68;
        margin-top: 0;
    }

    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        opacity: 0.5;
        margin-bottom: 0.25rem;
    }

    .image-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
    }

    .image-description {
        opacity: 0.72;
        line-height: 1.55;
        min-height: 4.5rem;
    }

    .workflow {
        text-align: center;
        padding: 1.2rem 1rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        border-radius: 10px;
        background-color: rgba(128, 128, 128, 0.06);
        font-size: 1.1rem;
        font-weight: 600;
    }

    .project-scope {
        text-align: center;
        padding: 1.5rem;
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 12px;
        background-color: rgba(128, 228, 128, 0.04);
    }

    .project-scope h3 {
        margin-top: 0;
        margin-bottom: 0.5rem;
    }

    .scope-main {
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 0.8rem;
    }

    .arrow {
        text-align: center;
        font-size: 1.8rem;
        opacity: 0.5;
        padding: 0.2rem;
    }

    .source-text {
        font-size: 0.78rem;
        opacity: 0.55;
        line-height: 1.4;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Hero
# -------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🌱 Precision Farming · Weed Detection</h1>
        <div class="hero-subtitle">
            From field images to targeted weed removal
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Motivation
# -------------------------------------------------------------------

robot_col, workflow_col = st.columns(
    2,
    gap="large",
)


with robot_col:

    with st.container(border=True):

        st.markdown(
            """
            <div class="section-label">
                The vision
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.image(
            ROBOT_IMAGE,
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="image-title">
                🤖 Autonomous precision weeding
            </div>

            <div class="image-description">
                Camera-based perception can enable agricultural robots
                to identify individual weeds and apply highly targeted
                treatment instead of treating the entire field.    
            </div>
            """,
            unsafe_allow_html=True,
        )
        #st.space("xxsmall")
        st.markdown(
            """
            <div class="source-text">
                Image: D. G. Pai, R. Kamath and M. Balachandra, Deep Learning Techniques for Weed Detection in Agricultural Environments: A Comprehensive Review
            </div>
            """,
            unsafe_allow_html=True,
        )


with workflow_col:

    with st.container(border=True):

        st.markdown(
            """
            <div class="section-label">
                The system
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.image(
            WORKFLOW_IMAGE,
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="image-title">
                ⚙️ Perception to actuation
            </div>

            <div class="image-description">
                A complete weeding system must detect crop and weeds,
                determine a safe target location and finally control
                the physical actuator.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="source-text">
                Figure: Fatima, H. S., ul Hassan, I., Hasan, S., Khurram, M., Stricker, D., & Afzal, M. Z. (2023).
                Formation of a Lightweight, Deep Learning-Based Weed Detection System for a Commercial Autonomous Laser Weeding Robot.
            </div>
            """,
            unsafe_allow_html=True,
        )


# -------------------------------------------------------------------
# Full application workflow
# -------------------------------------------------------------------

st.space("medium")

st.markdown(
    """
    <div class="section-label" style="text-align:center;">
        Long-term application
    </div>

    <div class="workflow">
        📷 Field Image
        &nbsp;&nbsp; → &nbsp;&nbsp;
        🧠 Detect Crop & Weed
        &nbsp;&nbsp; → &nbsp;&nbsp;
        📍 Localize Target
        &nbsp;&nbsp; → &nbsp;&nbsp;
        🛡️ Verify Safety
        &nbsp;&nbsp; → &nbsp;&nbsp;
        🎯 Actuate
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Project scope
# -------------------------------------------------------------------

st.markdown(
    """
<div class="arrow">↓</div>

<div class="project-scope">
<div class="section-label">Scope of this project</div>

<h3>🧠 Object Detection</h3>

<div class="scope-main">
🌽 Maize &nbsp;&nbsp; + &nbsp;&nbsp; 🌿 Weeds
</div>

<p>
Development and evaluation of a robust detection pipeline
under realistic field conditions.
</p>

<p>
<strong>
YOLO26 ↔ RF-DETR
&nbsp;&nbsp; · &nbsp;&nbsp;
Nano ↔ Small ↔ Medium
&nbsp;&nbsp; · &nbsp;&nbsp;
PyTorch ↔ ONNX
</strong>
</p>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Research questions
# -------------------------------------------------------------------

st.space("medium")

st.markdown("### What does the project investigate?")

question_1, question_2, question_3 = st.columns(
    3,
    gap="medium",
)


with question_1:

    with st.container(border=True):

        st.markdown(
            """
            ### 🎯 Accuracy

            How reliably can the models distinguish
            **maize from weeds** under realistic field conditions?

            **Metrics**

            Precision · Recall · mAP50 · mAP50-95
            """
        )


with question_2:

    with st.container(border=True):

        st.markdown(
            """
            ### 🌤️ Robustness

            How do training strategies affect detection under
            **shadows, illumination changes, small plants and
            complex backgrounds?**

            **Experiments**

            Mosaic · Flipping · HSV variation
            """
        )


with question_3:

    with st.container(border=True):

        st.markdown(
            """
            ### ⚡ Inference

            What is the practical trade-off between
            **detection quality and inference latency?**

            **Comparison**

            Framework · Model size · Backend
            """
        )


# -------------------------------------------------------------------
# Safety motivation
# -------------------------------------------------------------------

st.space("medium")

with st.container(border=True):

    st.markdown(
        """
        ### 🛡️ Why detection errors are not equally expensive

        In a future laser-based system, standard detection accuracy
        would not be sufficient as the only optimization target.
        Application-specific errors have very different consequences.
        """
    )

    safety_1, safety_2, safety_3 = st.columns(3)

    with safety_1:
        st.markdown(
            """
            **🌽 → 🌿 Crop misclassification**

            Maize classified as weed could become a direct target.
            """
        )

    with safety_2:
        st.markdown(
            """
            **👻 Missed maize**

            An undetected crop plant cannot be included in a
            protected safety region.
            """
        )

    with safety_3:
        st.markdown(
            """
            **🌿 ↔ 🌽 Spatial proximity**

            A correctly detected weed may still be too close to
            crop tissue for safe treatment.
            """
        )


# -------------------------------------------------------------------
# Boundary of current project
# -------------------------------------------------------------------

st.info(
    "This project stops at bounding-box object detection. "
    "A real targeting system would require additional spatial "
    "localization, such as segmentation and/or plant or stem "
    "keypoints, together with explicit crop safety zones."
)


# -------------------------------------------------------------------
# Sources
# -------------------------------------------------------------------

st.divider()

st.markdown("### Sources")

source_col1, source_col2 = st.columns(2)


with source_col1:

    st.markdown(
        """
        **ETH Zurich · Laser-based weeding robot**

        Example of a camera- and laser-based precision-weeding system.
        """
    )

    st.link_button(
        "ETH Zurich project",
        "https://partnerships.ethz.ch/news-and-events/news/data/"
        "2022/03/laser-based-weeding-robot.html",
    )


with source_col2:

    st.markdown(
        """
        **Balabantaray et al. · Robotic weed management**

        Example workflow from visual detection to targeted actuation.
        """
    )

    st.link_button(
        "Scientific publication",
        "https://www.frontiersin.org/journals/robotics-and-ai/"
        "articles/10.3389/frobt.2024.1441371/full",
    )