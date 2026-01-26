import streamlit as st
import pandas as pd
import os

from db import delete_person, delete_person, get_people_count, get_people_info, merge_identities, update_name

st.set_page_config(page_title="Vision Manager", layout="wide")  

@st.fragment(run_every="3s")
def refresh_people_list():
    data = get_people_info()
    df = pd.DataFrame(data, columns=["id", "name", "thumbnail_path"])
    
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
            
            st.divider()

st.title("👥 Advanced Identity Manager")

total_count = get_people_count()
st.sidebar.metric("Total Identities", total_count)

tab_main, tab_merge = st.tabs(["Identify & Name", "Merge Maintenance"])

with tab_main:
    refresh_people_list()

with tab_merge:
    st.subheader("🔗 Merge Duplicate Identities")
    st.info("This will move all face data from the 'Source' to the 'Destination' and delete the source entry.")

    # 1. Fetch current data for validation
    data = get_people_info()
    # Create a lookup dictionary: {id: (name, path)}
    people_map = {row[0]: {"name": row[1], "path": row[2]} for row in data}

    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.markdown("### 🎯 Destination")
        keep_id = st.number_input("ID to KEEP (Correct ID)", min_value=1, step=1, key="keep_id")
        
        # Validation Preview for Keep ID
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
        
        # Validation Preview for Delete ID
        if delete_id in people_map:
            person = people_map[delete_id]
            st.error(f"Found: **{person['name']}**")
            if person['path'] and os.path.exists(person['path']):
                st.image(person['path'], width=150)
        else:
            st.warning(f"ID {delete_id} does not exist.")

    st.divider()

    # 2. Pre-Merge Safety Checks
    if keep_id == delete_id:
        st.error("Select two different IDs to merge.")
    elif keep_id not in people_map or delete_id not in people_map:
        st.button("Execute Merge", disabled=True, help="Both IDs must exist to merge.")
    else:
        # If all checks pass, show the active button
        st.warning(f"Warning: Confirming will permanently delete ID {delete_id} and associate its face data with {people_map[keep_id]['name']}.")
        
        if st.button("🚀 Confirm and Execute Merge"):
            # Call your DB function
            merge_identities(keep_id, delete_id)
            
            # Clean up the duplicate's thumbnail
            source_path = people_map[delete_id]['path']
            if source_path and os.path.exists(source_path):
                os.remove(source_path)
                
            st.success(f"Done! ID {delete_id} merged into {keep_id}.")
            st.balloons()
            st.rerun()