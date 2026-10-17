# step 1 load data

import pandas as pd
df=pd.read_csv("Mall_customer_Dataset.csv")

print("\n Data set rows and columbs")
print(df.head())

print("\n Dataset info")
print(df.info())

print("\n summary statics")
print(df.describe())

print("\ndata shape")
print(df.shape)

print("\n missing values")
print(df.isnull().sum())



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# STEp =2
# Feature selsection

x=df[["Annual Income (k$)","Spending Score (1-100)"]]

print("\n selected feature")
print(x.columns)

# step=3
# feature scalling

scaler=StandardScaler()

X_scaled=scaler.fit_transform(x)

# step 4 Elbow method

wcss=[]

for i in range (1,11):
    kmeans=KMeans(n_clusters=i,random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11),wcss,marker="o")
plt.title("Elbow method")
plt.xlabel("Number of cluster")
plt.ylabel("Wcss")
plt.grid(True)
plt.savefig("number of cluster vs Wccs.png")
plt.show()

# step 5 Model training

kmeans=KMeans(n_clusters=5,random_state=42)
kmeans.fit(X_scaled)

# step 6

clusters=kmeans.predict(X_scaled)
df["clusters"]=clusters

# step 7
print(df.head())

# step 8 cluster visulization

plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=clusters,cmap="viridis",s=60)

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    color="red",
    s=250,
    marker="X",
    label="centroids"
)
plt.title("Customers segmetion using K_means")
plt.xlabel("Annual income Scaled")
plt.ylabel("spending scaore scaled")
plt.legend()
plt.grid(True)
plt.savefig("annual income scaled vs spending score scaled.png")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["clusters"],
    cmap="viridis",
    s=60
    )
plt.title("Customer segments")
plt.xlabel("Annual income ")
plt.ylabel("Spending income")
plt.grid(True)
plt.savefig("annual income vs spending income.png")
plt.show()