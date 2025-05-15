
import pandas as pd
import joblib
from sklearn.cluster import KMeans

data = pd.DataFrame({
    "Harga_Maks": [7000000, 8500000, 6000000, 9000000, 10000000],
    "RAM_Min": [8, 16, 8, 12, 16],
    "Prosesor_Min": [5, 7, 6, 7, 8],
    "Penyimpanan_Min": [256, 512, 256, 512, 1024]
})

model = KMeans(n_clusters=3, random_state=42)
model.fit(data)

joblib.dump(model, "model_cluster.pkl")

cluster_bobot = {
    0: [0.3, 0.25, 0.25, 0.2],
    1: [0.2, 0.3, 0.3, 0.2],
    2: [0.25, 0.25, 0.3, 0.2]
}
joblib.dump(cluster_bobot, "cluster_bobot.pkl")
