import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


os.makedirs("output", exist_ok=True)

df = pd.read_csv("data/Wholesale customers data.csv")

print("=" * 60)
print("Wholesale Customers Dataset Loaded Successfully")
print("=" * 60)


print("\nFirst Five Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum())


X = df.drop(columns=["Channel", "Region"])


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


wcss = []

for i in range(1,11):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)


plt.figure(figsize=(8,6))

plt.plot(range(1,11), wcss, marker="o")

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.grid(True)

plt.savefig("output/elbow_method.png")

plt.show()


kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters


plt.figure(figsize=(8,6))

plt.scatter(
    df["Fresh"],
    df["Milk"],
    c=df["Cluster"]
)

plt.xlabel("Fresh")

plt.ylabel("Milk")

plt.title("Customer Segments using K-Means")

plt.savefig("output/customer_clusters.png")

plt.show()


print("\nCluster Counts")

print(df["Cluster"].value_counts())

print("\nCluster Centers")

print(kmeans.cluster_centers_)

print("\nAssignment Completed Successfully")