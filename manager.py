import streamlit as st
import pandas as pd
import os
import numpy as np
import networkx as nx

from core.face import FaceEngine

os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")

from db import delete_person, delete_person, get_known_faces, get_people_count, get_people_info, get_setting, merge_identities, set_approval, set_setting, update_name

st.set_page_config(page_title="Vision Manager", layout="wide")  

@st.fragment(run_every="3s")
def refresh_people_list():
    data = get_people_info()
    df = pd.DataFrame(data, columns=["id", "name", "thumbnail_path", "is_approved"])
    
    if df.empty:
        st.info("No people detected yet. Run your main vision script first!")
    else:
        st.subheader("📡 Live Detected Identities")
        
        for index, row in df.iterrows():
            # Create 4 columns to fit Image, ID, Name Input, and Actions
            col_img, col_id, col_name, col_act = st.columns([1.5, 0.5, 2, 1])
            
            with col_img:
                # Show the saved face crop
                if row['thumbnail_path'] and os.path.exists(row['thumbnail_path']):
                    st.image(row['thumbnail_path'], width=120)
                else:
                    st.caption("No thumbnail")
            
            with col_id:
                st.write(f"**ID: {row['id']}**")
            
            with col_name:
                new_name = st.text_input(f"Label", value=row['name'], key=f"in_{row['id']}")
                if st.button("Save", key=f"save_{row['id']}"):
                    update_name(row['id'], new_name)
                    st.success(f"ID {row['id']} Named!")
            
            with col_act:
                # Add a delete button to clean up the DB
                if st.button("🗑️", key=f"del_{row['id']}", help="Delete this identity"):
                    delete_person(row['id'], row['thumbnail_path'])
                    st.rerun()
                # Checkbox to approve/cloak
                is_approved = row['is_approved'] == 1
                if st.checkbox("✅ Authorize", value=is_approved, key=f"auth_{row['id']}"):
                    set_approval(row['id'], 1)
                else:
                    set_approval(row['id'], 0)
            
            st.divider()
            
def show_smart_merge():
    st.subheader("🤖 AI Smart Grouping & Bulk Merge")
    st.info("Adjust the threshold. Faces grouped together are highly likely to be the same person.")

    # 1. Load Data and Engine
    engine = FaceEngine(model_name='buffalo_s')
    raw_known = get_known_faces()
    if len(raw_known) < 2:
        st.info("Not enough data to cluster.")
        return

    # Group by ID and get centroids
    grouped_vectors = {}
    for pid, vec in raw_known:
        grouped_vectors.setdefault(pid, []).append(vec)

    identity_centroids = []
    for pid, vecs in grouped_vectors.items():
        avg_vec = np.mean(vecs, axis=0)
        avg_vec /= np.linalg.norm(avg_vec)
        identity_centroids.append((pid, avg_vec))

    # 2. Setup Graph and Threshold
    threshold = st.slider("Similarity Sensitivity", 0.1, 0.9, 0.45, key="smart_threshold")
    people_info = get_people_info()
    info_map = {row[0]: (row[1], row[2]) for row in people_info}

    # Create a graph where nodes are IDs and edges are "similarity > threshold"
    G = nx.Graph()
    G.add_nodes_from([item[0] for item in identity_centroids])

    for i in range(len(identity_centroids)):
        for j in range(i + 1, len(identity_centroids)):
            id1, v1 = identity_centroids[i]
            id2, v2 = identity_centroids[j]
            if engine.compute_similarity(v1, v2) > threshold:
                G.add_edge(id1, id2)

    # Find clusters (Connected Components)
    clusters = [list(c) for c in nx.connected_components(G) if len(c) > 1]

    if not clusters:
        st.success("✨ No duplicate clusters found at this threshold.")
        return

    st.write(f"Found **{len(clusters)}** potential duplicate groups.")

    # 3. Render Clusters
    for idx, cluster in enumerate(clusters):
        with st.container(border=True):
            st.markdown(f"#### 📦 Cluster Group {idx + 1}")
            
            # Display all faces in this cluster side-by-side
            cols = st.columns(len(cluster))
            selected_to_merge = []
            
            for col, pid in zip(cols, cluster):
                name, path = info_map.get(pid, ("Unknown", None))
                with col:
                    if path and os.path.exists(path):
                        st.image(path, width=100)
                    st.write(f"**ID {pid}**")
                    st.caption(name)
                    # Checkbox to include in merge
                    is_selected = st.checkbox("Select", value=True, key=f"sel_{idx}_{pid}")
                    if is_selected:
                        selected_to_merge.append(pid)

            if len(selected_to_merge) > 1:
                st.divider()
                # User selects which one survives
                primary_id = st.selectbox(
                    "Select Primary ID (the one to keep)", 
                    options=selected_to_merge,
                    key=f"primary_{idx}"
                )
                
                if st.button(f"🚀 Bulk Merge {len(selected_to_merge)-1} IDs into {primary_id}", key=f"btn_bulk_{idx}"):
                    # Execute all merges
                    for source_id in selected_to_merge:
                        if source_id != primary_id:
                            merge_identities(primary_id, source_id)
                            # Clean up file
                            _, source_path = info_map.get(source_id, (None, None))
                            if source_path and os.path.exists(source_path):
                                os.remove(source_path)
                    
                    st.success(f"Merged successfully into ID {primary_id}!")
                    st.rerun()
            else:
                st.warning("Select at least 2 IDs to perform a merge.")

st.title("👥 Advanced Identity Manager")

total_count = get_people_count()
st.sidebar.metric("Total Identities", total_count)

# --- SIDEBAR SETTINGS SECTION ---
st.sidebar.title("⚙️ System Settings")

# 1. Privacy Cloak Toggle
# We fetch the current status from DB (default to 'off' if not set)
cloak_status = get_setting("enable_privacy_cloak") == "True"
if st.sidebar.toggle("🔒 Enable Privacy Cloak", value=cloak_status):
    set_setting("enable_privacy_cloak", "True")
else:
    set_setting("enable_privacy_cloak", "False")

# 2. Cyberpunk HUD Toggle
hud_status = get_setting("enable_hud") == "True"
if st.sidebar.toggle("🕶️ Cyberpunk HUD Overlay", value=hud_status):
    set_setting("enable_hud", "True")
else:
    set_setting("enable_hud", "False")

#3. Show Landmarks Toggle
landmark_status = get_setting("show_landmarks") == "True"
if st.sidebar.toggle("📍 Show Facial Landmarks", value=landmark_status):
    set_setting("show_landmarks", "True")
else:
    set_setting("show_landmarks", "False")

st.sidebar.divider()
# -------------------------------

tab_main, tab_smart_merge, tab_merge = st.tabs(["Identify & Name", "Smart Deduplication", "Merge Maintenance"])

with tab_main:
    refresh_people_list()

with tab_merge:
    st.subheader("🔗 Merge Duplicate Identities")
    st.info("This will move all face data from the 'Source' to the 'Destination' and delete the source entry.")

    data = get_people_info()
    people_map = {row[0]: {"name": row[1], "path": row[2]} for row in data}

    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.markdown("### 🎯 Destination")
        keep_id = st.number_input("ID to KEEP (Correct ID)", min_value=1, step=1, key="keep_id")
        
        if keep_id in people_map:
            person = people_map[keep_id]
            st.success(f"Found: **{person['name']}**")
            if person['path'] and os.path.exists(person['path']):
                st.image(person['path'], width=150)
        else:
            st.warning(f"ID {keep_id} does not exist.")

    with m_col2:
        st.markdown("### 🗑️ Source")
        delete_id = st.number_input("ID to REMOVE (Duplicate)", min_value=1, step=1, key="delete_id")
        
        if delete_id in people_map:
            person = people_map[delete_id]
            st.error(f"Found: **{person['name']}**")
            if person['path'] and os.path.exists(person['path']):
                st.image(person['path'], width=150)
        else:
            st.warning(f"ID {delete_id} does not exist.")

    st.divider()

    if keep_id == delete_id:
        st.error("Select two different IDs to merge.")
    elif keep_id not in people_map or delete_id not in people_map:
        st.button("Execute Merge", disabled=True, help="Both IDs must exist to merge.")
    else:
        st.warning(f"Warning: Confirming will permanently delete ID {delete_id} and associate its face data with {people_map[keep_id]['name']}.")
        
        if st.button("🚀 Confirm and Execute Merge"):
            merge_identities(keep_id, delete_id)
            
            source_path = people_map[delete_id]['path']
            if source_path and os.path.exists(source_path):
                os.remove(source_path)
                
            st.success(f"Done! ID {delete_id} merged into {keep_id}.")
            st.balloons()
            st.rerun()
            
with tab_smart_merge:
    show_smart_merge()