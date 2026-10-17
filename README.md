# mall-customer-segmentation-kmeans
Mall Customer Segmentation using K-Means Clustering to group customers based on annual income and spending score, with feature scaling, the Elbow Method, and cluster visualization using Python.

# Mall Customer Segmentation using K-Means Clustering

# Project Overview

This project uses **K-Means Clustering**, an unsupervised machine learning algorithm, to segment mall customers based on their annual income and spending score.

The objective is to identify customer groups with similar characteristics and visualize their distribution to understand different customer segments.

# Objectives

* Analyze the mall customer dataset.
* Select relevant features for customer segmentation.
* Standardize numerical features using `StandardScaler`.
* Use the Elbow Method to help determine a suitable number of clusters.
* Train a K-Means clustering model.
* Assign cluster labels to customers.
* Visualize customer segments and cluster centroids.

## Dataset

The dataset used in this project was obtained from Kaggle.

It contains customer information, including annual income and spending score, which are used to perform customer segmentation with K-Means clustering.

**Features Used:**

* `Annual Income (k$)`
* `Spending Score (1-100)`

**Dataset Source:** Kaggle

**Note:** Please refer to the original Kaggle dataset page for the dataset creator, license, and usage conditions.


**Selected Features:**

* `Annual Income (k$)`
* `Spending Score (1-100)`

# Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

# Machine Learning Workflow

# 1. Data Loading and Exploration

Loaded the dataset using Pandas and inspected its structure, summary statistics, dimensions, and missing values.

# 2. Feature Selection

Selected annual income and spending score as the input features for clustering.

# 3. Feature Scaling

Applied `StandardScaler` to standardize the selected features.

Scaling is important because K-Means uses distances to assign observations to clusters. Features with larger numerical scales can otherwise have a disproportionate influence on those distances.

# 4. Elbow Method

Used the Elbow Method to evaluate different cluster counts.

The method plots the Within-Cluster Sum of Squares (WCSS) against the number of clusters. The resulting curve helps identify a potential balance between cluster compactness and model complexity.

# 5. Model Training

Trained a K-Means clustering model with five clusters:

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)
```

Five clusters are used in this implementation. The Elbow Method can help assess whether this is a suitable choice for the dataset.

# 6. Customer Cluster Assignment

Assigned a cluster label to each customer and added the labels to the original DataFrame.

```python
clusters = kmeans.predict(X_scaled)
df["clusters"] = clusters
```

# 7. Data Visualization

Created visualizations to examine the customer segments:

* Elbow Method plot for evaluating cluster counts.
* Scaled-feature scatter plot with cluster centroids.
* Customer segmentation scatter plot using the original feature values.

## Project Visualizations

The code generates the following plots:

1. `number of cluster vs Wccs.png`
2. `annual income scaled vs spending score scaled.png`
3. `annual income vs spending income.png`

These visualizations help examine cluster separation, centroid locations, and customer distribution.

# How to Run the Project

# Prerequisites

Install Python and the required libraries.

# 1. Clone the Repository

```bash
git clone https://github.com/deepanshurajpoot777-cpu/mall-customer-segmentation-kmeans.git
```

# 2. Navigate to the Project Directory

```bash
cd mall-customer-segmentation-kmeans
```

# 3. Install Dependencies

```bash
pip install pandas matplotlib seaborn scikit-learn
```

# 4. Add the Dataset

Place `Mall_customer_Dataset.csv` in the project directory.

Ensure the CSV file contains the required columns:

* `Annual Income (k$)`
* `Spending Score (1-100)`

# 5. Run the Python Script

Save the project code in a Python file, such as `mall_customer_segmentation.py`, and run:

```bash
python mall_customer_segmentation.py
```

The script will display the visualizations and save the generated plots to the project directory.

# Key Learnings

* Understanding unsupervised machine learning.
* Implementing K-Means clustering.
* Applying feature scaling before distance-based clustering.
* Using the Elbow Method to evaluate cluster counts.
* Interpreting cluster centroids.
* Visualizing customer segments using Matplotlib.

# Limitations

* The model uses only annual income and spending score.
* The choice of five clusters should be evaluated using the dataset and clustering results.
* Cluster labels represent group assignments, not predefined customer categories.
* The analysis identifies patterns in the available data and does not establish customer behavior or business outcomes by itself.

# Future Improvements

* Compare different cluster counts using additional evaluation metrics, such as the Silhouette Score.
* Analyze customer demographics if relevant data is available.
* Develop an interactive customer segmentation dashboard.
* Explore alternative clustering algorithms.

# Author

Deepanshu

GitHub: [deepanshurajpoot777-cpu](https://github.com/deepanshurajpoot777-cpu)

---

This project was developed as part of my journey in learning and applying Machine Learning concepts through practical projects.

