"""
FCDT-TPFF – IEEE Conference Ready Training Script
✔ Deterministic
✔ Temporal validation
✔ Robust clustering evaluation
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
import os
import random
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import kruskal
import warnings
warnings.filterwarnings("ignore")

from fcdt_tpff_model import FCDT_TPFF, DiabetesDataset, collate_fn

# ------------------------------------------------------------------
# REPRODUCIBILITY (IEEE MUST)
# ------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
NUM_CLUSTERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nFCDT-TPFF Training | Device: {DEVICE}")

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
with open("./processed_data/patient_features.pkl", "rb") as f:
    temporal = pickle.load(f)

static = pd.read_csv("./processed_data/static_features.csv", index_col=0)
demo = pd.read_csv("./processed_data/demographics.csv")

common_ids = list(set(temporal.keys()) & set(static.index))
temporal = {pid: temporal[pid] for pid in common_ids}
static = static.loc[common_ids]
demo = demo[demo.subject_id.isin(common_ids)]

# ------------------------------------------------------------------
# TEMPORAL VALIDATION SPLIT (CONFERENCE-SAFE)
# ------------------------------------------------------------------
demo["year"] = pd.to_datetime(demo["admittime"]).dt.year
cutoff = demo["year"].quantile(0.7)

train_ids = demo[demo.year <= cutoff].subject_id.tolist()
test_ids  = demo[demo.year > cutoff].subject_id.tolist()

print(f"Train patients: {len(train_ids)} | Test patients: {len(test_ids)}")

train_temporal = {pid: temporal[pid] for pid in train_ids}
test_temporal  = {pid: temporal[pid] for pid in test_ids}

train_static = static.loc[train_ids]
test_static  = static.loc[test_ids]

# ------------------------------------------------------------------
# DATASETS
# ------------------------------------------------------------------
train_loader = torch.utils.data.DataLoader(
    DiabetesDataset(train_temporal, train_static),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

test_loader = torch.utils.data.DataLoader(
    DiabetesDataset(test_temporal, test_static),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# ------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------
sample = next(iter(train_temporal.values()))
temporal_dim = sample["micro"].shape[1]
static_dim = train_static.shape[1]

model = FCDT_TPFF(
    temporal_input_dim=temporal_dim,
    static_input_dim=static_dim,
    num_clusters=NUM_CLUSTERS
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)

# ------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------
best_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    losses = []

    for batch in train_loader:
        optimizer.zero_grad()

        z, adj = model(batch["temporal"], batch["static"])

        loss_compact = torch.mean(torch.norm(z, dim=1))
        loss_graph = torch.mean(adj * torch.cdist(z, z))
        loss = loss_compact + 0.15 * loss_graph

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

    avg_loss = np.mean(losses)
    print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "./processed_data/best_model.pt")

# ------------------------------------------------------------------
# EMBEDDING EXTRACTION (TEST SET)
# ------------------------------------------------------------------
model.load_state_dict(torch.load("./processed_data/best_model.pt"))
model.eval()

embeddings = []

with torch.no_grad():
    for batch in test_loader:
        z = model(batch["temporal"], batch["static"], return_pregraph=True)

        embeddings.append(z.cpu().numpy())

embeddings = np.vstack(embeddings)

# ------------------------------------------------------------------
# CLUSTERING + METRICS
# ------------------------------------------------------------------
kmeans = KMeans(n_clusters=NUM_CLUSTERS, n_init=30, random_state=SEED)
labels = kmeans.fit_predict(embeddings)

sil = silhouette_score(embeddings, labels)
dbi = davies_bouldin_score(embeddings, labels)

print("\nIEEE-READY RESULTS (TEMPORAL VALIDATION)")
print(f"Silhouette Score: {sil:.4f}")
print(f"Davies-Bouldin Index: {dbi:.4f}")

np.save("./processed_data/test_embeddings.npy", embeddings)
np.save("./processed_data/test_labels.npy", labels)
