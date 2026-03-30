import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle

# ══════════════════════════════════════════════════════════
# FLIP THIS FLAG AFTER FIRST RUN:
# True = load from disk (2 seconds)
# False = rebuild from scratch (~5 minutes)
LOAD_SAVED = False
# ══════════════════════════════════════════════════════════

PATH      = r'C:\Users\Public\synthea\output\csv\\'
SAVE_PATH = r'C:\Users\Public\synthea\\'

if LOAD_SAVED:
    # ── FAST PATH ──────────────────────────────────────────
    print("Loading from disk...")
    with open(SAVE_PATH + 'graph.pkl', 'rb') as f:
        G = pickle.load(f)
    nodes = pd.read_csv(SAVE_PATH + 'nodes.csv')
    edges = pd.read_csv(SAVE_PATH + 'edges.csv')
    print(f"✅ Graph loaded:  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"✅ Nodes loaded:  {len(nodes):,} rows")
    print(f"✅ Edges loaded:  {len(edges):,} rows")

else:
    # ── STEP 1: LOAD CSVs ──────────────────────────────────
    print("Loading CSVs...")
    patients    = pd.read_csv(PATH + 'patients.csv')
    conditions  = pd.read_csv(PATH + 'conditions.csv')
    encounters  = pd.read_csv(PATH + 'encounters.csv')
    medications = pd.read_csv(PATH + 'medications.csv')
    print(f"✅ Patients:    {len(patients):,}")
    print(f"✅ Conditions:  {len(conditions):,}")
    print(f"✅ Encounters:  {len(encounters):,}")
    print(f"✅ Medications: {len(medications):,}")

    # ── STEP 2: NODE FEATURES ──────────────────────────────
    print("\nBuilding node features...")
    condition_counts  = conditions.groupby('PATIENT').size().reset_index(name='condition_count')
    encounter_counts  = encounters.groupby('PATIENT').size().reset_index(name='encounter_count')
    medication_counts = medications.groupby('PATIENT').size().reset_index(name='medication_count')

    nodes = patients[['Id']].copy()
    nodes = nodes.merge(condition_counts,  left_on='Id', right_on='PATIENT', how='left').drop(columns='PATIENT')
    nodes = nodes.merge(encounter_counts,  left_on='Id', right_on='PATIENT', how='left').drop(columns='PATIENT')
    nodes = nodes.merge(medication_counts, left_on='Id', right_on='PATIENT', how='left').drop(columns='PATIENT')
    nodes = nodes.fillna(0)
    print(f"✅ Node features built: {len(nodes):,} patients")

    # ── STEP 3: BUILD EDGES ────────────────────────────────
    print("\nBuilding edges...")
    provider_patients = encounters[['PATIENT', 'PROVIDER']].drop_duplicates()
    edges = provider_patients.merge(provider_patients, on='PROVIDER')\
                             .query('PATIENT_x != PATIENT_y')\
                             [['PATIENT_x', 'PATIENT_y']]\
                             .drop_duplicates()
    print(f"✅ Edges built: {len(edges):,} connections")

    # ── STEP 4: BUILD GRAPH (FAST) ─────────────────────────
    print("\nBuilding graph (fast mode)...")
    G = nx.Graph()

    # Fast node adding — no iterrows!
    node_attrs = nodes.set_index('Id')[['condition_count',
                                        'encounter_count',
                                        'medication_count']].to_dict('index')
    G.add_nodes_from(node_attrs.items())
    print(f"✅ Nodes added: {G.number_of_nodes():,}")

    # Fast edge adding — no iterrows!
    G.add_edges_from(zip(edges['PATIENT_x'], edges['PATIENT_y']))
    print(f"✅ Edges added: {G.number_of_edges():,}")

    print(f"\n✅ Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # ── STEP 5: SAVE EVERYTHING ────────────────────────────
    print("\nSaving to disk...")
    with open(SAVE_PATH + 'graph.pkl', 'wb') as f:
        pickle.dump(G, f)
    nodes.to_csv(SAVE_PATH + 'nodes.csv', index=False)
    edges.to_csv(SAVE_PATH + 'edges.csv', index=False)
    print("✅ graph.pkl saved!")
    print("✅ nodes.csv saved!")
    print("✅ edges.csv saved!")
    print("⭐ Set LOAD_SAVED = True for all future runs!")

