import json
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from beyx_physics_sim import (
    ARENA_R,
    BIT,
    BLADE,
    POCKET_ANGLES,
    POCKET_HW,
    RATCHET,
    load_parts,
    monte_carlo,
    part_dicts,
    simulate,
)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

COLOR_A = "tab:blue"
COLOR_B = "tab:red"


def set_combo(prefix: str, blade: str, ratchet: str, bit: str) -> None:
    st.session_state[f"{prefix}_blade"] = blade
    st.session_state[f"{prefix}_ratchet"] = ratchet
    st.session_state[f"{prefix}_bit"] = bit


def plot_arena(histA, histB):
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    circle = plt.Circle((0, 0), ARENA_R, color="black", fill=False)
    ax.add_patch(circle)
    for ang in POCKET_ANGLES:
        theta1 = np.degrees(ang - POCKET_HW)
        theta2 = np.degrees(ang + POCKET_HW)
        wedge = plt.matplotlib.patches.Wedge((0, 0), ARENA_R, theta1, theta2, width=0.05, color="lightgrey")
        ax.add_patch(wedge)
    if histA:
        ax.plot([p[0] for p in histA], [p[1] for p in histA], color=COLOR_A, label="A")
    if histB:
        ax.plot([p[0] for p in histB], [p[1] for p in histB], color=COLOR_B, label="B")
    ax.legend()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    st.pyplot(fig)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "A_blade" not in st.session_state:
    st.session_state.update({
        "A_blade": list(BLADE.keys())[0],
        "A_ratchet": list(RATCHET.keys())[0],
        "A_bit": list(BIT.keys())[0],
        "B_blade": list(BLADE.keys())[1],
        "B_ratchet": list(RATCHET.keys())[0],
        "B_bit": list(BIT.keys())[0],
    })

# ---------------------------------------------------------------------------
# Sidebar: combo pickers and controls
# ---------------------------------------------------------------------------

st.sidebar.header("Combo A")
A_blade = st.sidebar.selectbox("Blade", options=list(BLADE.keys()), key="A_blade")
A_ratchet = st.sidebar.selectbox("Ratchet", options=list(RATCHET.keys()), key="A_ratchet")
A_bit = st.sidebar.selectbox("Bit", options=list(BIT.keys()), key="A_bit")

st.sidebar.header("Combo B")
B_blade = st.sidebar.selectbox("Blade", options=list(BLADE.keys()), key="B_blade")
B_ratchet = st.sidebar.selectbox("Ratchet", options=list(RATCHET.keys()), key="B_ratchet")
B_bit = st.sidebar.selectbox("Bit", options=list(BIT.keys()), key="B_bit")

seed = st.sidebar.number_input("Seed", value=0, step=1)
N = st.sidebar.number_input("Monte-Carlo N", min_value=10, max_value=5000, value=200, step=10)

# Presets
with st.sidebar.expander("Presets"):
    st.markdown("**Deck A – Fortress**")
    presets_A = {
        "Knight Shield/3-60/Gear Needle": ("Knight Shield", "3-60", "Gear Needle"),
        "Knight Shield/3-60/Point": ("Knight Shield", "3-60", "Point"),
        "Cobalt Ifrit/4-55/Wall Ball": ("Cobalt Ifrit", "4-55", "Wall Ball"),
        "Cobalt Ifrit/4-55/Disc Ball": ("Cobalt Ifrit", "4-55", "Disc Ball"),
        "Silver Wolf/9-60/Gear Ball": ("Silver Wolf", "9-60", "Gear Ball"),
        "Silver Wolf/9-60/Bound Spike": ("Silver Wolf", "9-60", "Bound Spike"),
    }
    for name, combo in presets_A.items():
        if st.button(name):
            set_combo("A", *combo)

    st.markdown("**Deck B – Recoil Defense**")
    presets_B = {
        "Phoenix Wing/3-80/Bound Spike": ("Phoenix Wing", "3-80", "Bound Spike"),
        "Dran Sword/1-60/Dot": ("Dran Sword", "1-60", "Dot"),
        "Dran Sword/1-60/Gear Needle": ("Dran Sword", "1-60", "Gear Needle"),
        "Cowle Dragon/9-60/Spike": ("Cowle Dragon", "9-60", "Spike"),
        "Cowle Dragon/9-60/Gear Point": ("Cowle Dragon", "9-60", "Gear Point"),
    }
    for name, combo in presets_B.items():
        if st.button(name):
            set_combo("B", *combo)


# ---------------------------------------------------------------------------
# Main panels
# ---------------------------------------------------------------------------

st.title("Beyblade X Physics Coach")

st.subheader("Single Battle")
if st.button("Run 1 Battle"):
    comboA = {"blade": A_blade, "ratchet": A_ratchet, "bit": A_bit}
    comboB = {"blade": B_blade, "ratchet": B_ratchet, "bit": B_bit}
    result = simulate(comboA, comboB, int(seed))
    win, reason = result["result"]
    st.markdown(f"**Outcome: {win} by {reason}**")
    plot_arena(result["histA"], result["histB"])

st.subheader("Monte-Carlo")
if st.button("Run Monte-Carlo"):
    comboA = {"blade": A_blade, "ratchet": A_ratchet, "bit": A_bit}
    comboB = {"blade": B_blade, "ratchet": B_ratchet, "bit": B_bit}
    counts = monte_carlo(comboA, comboB, N=int(N), seed=int(seed))
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=[COLOR_A, COLOR_B] * 3)
    ax.set_ylabel("count")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    st.pyplot(fig)
    st.download_button(
        "Download JSON",
        data=json.dumps({"A": comboA, "B": comboB, "counts": counts}, indent=2),
        file_name="mc_results.json",
    )

# ---------------------------------------------------------------------------
# Advanced section for editing dictionaries
# ---------------------------------------------------------------------------

with st.expander("Advanced"):
    parts = part_dicts()
    col1, col2, col3 = st.columns(3)
    with col1:
        blade_text = st.text_area("Blades", json.dumps(parts["blade"], indent=2), height=300)
        if st.button("Apply Blades"):
            try:
                load_parts(blade_json=blade_text)
                st.success("Blades updated")
            except Exception as e:
                st.error(f"Bad JSON: {e}")
        st.download_button("Download Blades", data=blade_text, file_name="blades.json")
    with col2:
        ratchet_text = st.text_area("Ratchets", json.dumps(parts["ratchet"], indent=2), height=300)
        if st.button("Apply Ratchets"):
            try:
                load_parts(ratchet_json=ratchet_text)
                st.success("Ratchets updated")
            except Exception as e:
                st.error(f"Bad JSON: {e}")
        st.download_button("Download Ratchets", data=ratchet_text, file_name="ratchets.json")
    with col3:
        bit_text = st.text_area("Bits", json.dumps(parts["bit"], indent=2), height=300)
        if st.button("Apply Bits"):
            try:
                load_parts(bit_json=bit_text)
                st.success("Bits updated")
            except Exception as e:
                st.error(f"Bad JSON: {e}")
        st.download_button("Download Bits", data=bit_text, file_name="bits.json")

    st.markdown("Upload JSON to replace all dicts")
    uploaded = st.file_uploader("JSON file", type="json")
    if uploaded and st.button("Load Uploaded"):
        try:
            data = json.load(uploaded)
            load_parts(
                blade_json=json.dumps(data.get("blade", {})),
                ratchet_json=json.dumps(data.get("ratchet", {})),
                bit_json=json.dumps(data.get("bit", {})),
            )
            st.success("Loaded from file")
        except Exception as e:
            st.error(f"Failed to load: {e}")

    # Optional video analysis panel
    with st.expander("Video Analysis (beta)"):
        upload = st.file_uploader("Upload MP4/MOV", type=["mp4", "mov"], key="video")
        if upload and st.button("Analyze Video"):
            from video_tools import analyze_video

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(upload.read())
                tmp_path = tmp.name
            result = analyze_video(tmp_path)
            st.write(result)

st.caption("Beyblade X Physics Coach – demo simulator")
