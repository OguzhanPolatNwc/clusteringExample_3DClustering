import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
# from matplotlib.colors import ListedColormap
# from sklearn import metrics
import warnings
import sys

# To ignore error
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)

# Loading the dataset
data = pd.read_csv("ClassificationExample/data/marketing_campaign.csv", sep="\t")
data.head()

# Information on features
data.info()

# To remove the NA values
data = data.dropna()

# Dates of the newest and oldest recorded customer
data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"])
dates = []
for i in data["Dt_Customer"]:
    i = i.date()
    dates.append(i)

# Created a feature "Customer_For"
days = []
d1 = max(dates)  # taking it to be the newest customer
for i in dates:
    delta = d1 - i
    days.append(delta)
data["Customer_For"] = days
data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")

data["Marital_Status"].value_counts()
data["Education"].value_counts()

# Age of customer today
data["Age"] = 2022 - data["Year_Birth"]

# Total spendings on various items
data["Spent"] = data["MntWines"] + data["MntFruits"] + data["MntMeatProducts"] + data["MntFishProducts"] + data[
    "MntSweetProducts"] + data["MntGoldProds"]

# Deriving living situation by marital status"Alone"
data["Living_With"] = data["Marital_Status"].replace(
    {"Married": "Partner", "Together": "Partner", "Absurd": "Alone", "Widow": "Alone", "YOLO": "Alone",
     "Divorced": "Alone", "Single": "Alone", })

# Feature indicating total children living in the household
data["Children"] = data["Kidhome"] + data["Teenhome"]

# Feature for total members in the household
data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner": 2}) + data["Children"]

# Feature pertaining parenthood
data["Is_Parent"] = np.where(data.Children > 0, 1, 0)

# Segmenting education levels in three groups
data["Education"] = data["Education"].replace(
    {"Basic": "Undergraduate", "2n Cycle": "Undergraduate", "Graduation": "Graduate", "Master": "Postgraduate",
     "PhD": "Postgraduate"})

# Dropping some of the redundant features
data.drop(["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID",
           'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response']
          , axis=1, inplace=True)

data.describe()

sns.boxplot(y="Age", data=data)
plt.show()

sns.boxplot(y="Income", data=data)
plt.show()

# Dropping the outliers by setting a cap on Age and income.
data = data[(data["Age"] < 90)]
data = data[(data["Income"] < 150000)]

# correlation matrix
corrmat = data.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corrmat, annot=True, center=0)
plt.show()

# Get list of categorical variables
categorical_col = [col for col in data.columns if data[col].dtype == "object"]

# Label Encoding the object dtypes.
# *** LabelEncoder can be used to normalize labels. It can also be used to transform
# non-numerical labels (as long as they are hashable and comparable) to numerical labels. Fit label encoder. ***
LE = LabelEncoder()
for i in categorical_col:
    data[i] = data[[i]].apply(LE.fit_transform)


# Scaling
scaler = StandardScaler()
scaler.fit(data)
scaled_ds = pd.DataFrame(scaler.transform(data), columns=data.columns)

# Scaled data to be used for reducing the dimensionality
scaled_ds.head()

# Initiating PCA to reduce dimensions aka features to 3
pca = PCA(n_components=3)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["col1", "col2", "col3"]))
PCA_ds.describe().T

# A 3D Projection Of Data In The Reduced Dimension
x = PCA_ds["col1"]
y = PCA_ds["col2"]
z = PCA_ds["col3"]
# To plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, c="maroon", marker="o")
ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
plt.show()

# Quick examination of elbow method to find numbers of clusters to make.
print('Elbow Method to determine the number of clusters to be formed:')
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(PCA_ds)
Elbow_M.show()

""" -----------------------------------------------------------------------------------------------------------  """

# Agglomerative Clustering is a type of hierarchical clustering algorithm. It is an unsupervised machine learning
# technique that divides the population into several clusters
# Initiating the Agglomerative Clustering model
AC = AgglomerativeClustering(n_clusters=4)
# fit model and predict clusters
yhat_AC = AC.fit_predict(PCA_ds)
PCA_ds["Clusters"] = yhat_AC
# Adding the Clusters feature to the original dataframe.
data["Clusters"] = yhat_AC

# Plotting the clusters
fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o')
ax.set_title("The Plot Of The Clusters")
plt.show()

# Plotting countplot of clusters
pal = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60"]
pl = sns.countplot(x=data["Clusters"], palette=pal)
pl.set_title("Distribution Of The Clusters")
plt.show()

pl = sns.scatterplot(data=data, x=data["Spent"], y=data["Income"], hue=data["Clusters"], palette=pal)
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()

# Plotting the number of deals purchased
plt.figure()
pl = sns.boxenplot(y=data["NumDealsPurchases"], x=data["Clusters"], palette=pal)
pl.set_title("Number of Deals Purchased")
plt.show()

# Print the relation between spent and selected columns
Personal = ["Kidhome", "Teenhome", "Customer_For", "Age", "Children", "Family_Size", "Is_Parent", "Education",
            "Living_With"]

for i in Personal:
    plt.figure()
    sns.jointplot(x=data[i], y=data["Spent"], hue=data["Clusters"], kind="kde", palette=pal)
    plt.show()
