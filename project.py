import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

# ---------------- LOAD DATA ----------------
file_path = r"C:\Users\NITHIN\Downloads\world_happiness_report.csv"
df = pd.read_csv(file_path)

target_regression = 'Happiness Score'

df = df.dropna(subset=[target_regression])
df = df.drop(columns=[df.columns[0], 'Standard Error', 'Dystopia Residual'], errors='ignore')

median_score = df[target_regression].median()
df['Happiness_Level'] = df[target_regression].apply(
    lambda x: 'High' if x >= median_score else 'Low'
)
target_classification = 'Happiness_Level'

numerical_features = [
    'Economy (GDP per Capita)', 'Family',
    'Health (Life Expectancy)', 'Freedom',
    'Trust (Government Corruption)', 'Generosity'
]

categorical_features = ['Region', 'year']
all_features = numerical_features + categorical_features

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, numerical_features),
    ('cat', cat_pipe, categorical_features)
])

# ---------------- ML MODELS ----------------
X = df[all_features]
y = df[target_regression]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

reg = LinearRegression()
reg.fit(X_train_p, y_train)
y_pred = reg.predict(X_test_p)

RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
R2 = r2_score(y_test, y_pred)

# Classification
y_cls = df[target_classification]
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_cls, test_size=0.3, random_state=42
)

X_train_cp = preprocessor.fit_transform(X_train_c)
X_test_cp = preprocessor.transform(X_test_c)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_cp, y_train_c)
y_pred_c = clf.predict(X_test_cp)

ACC = accuracy_score(y_test_c, y_pred_c)

# Clustering
factors = df[numerical_features]
factors_scaled = StandardScaler().fit_transform(factors)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(factors_scaled)

# ---------------- PLOTS ----------------
def heatmap():
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df[numerical_features + [target_regression]].corr(),
        annot=True, fmt=".2f",
        cmap="viridis", linewidths=0.5
    )
    plt.title("Correlation Heatmap of Happiness Factors and Score")
    plt.tight_layout()
    plt.show()

def distribution():
    plt.figure(figsize=(8, 5))
    sns.histplot(df[target_regression], kde=True, bins=20, color='teal')
    plt.title("Distribution of World Happiness Scores")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    plt.show()

def region_box():
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Region", y=target_regression, data=df, palette="Set3")
    plt.xticks(rotation=45, ha='right')
    plt.title("Happiness Score Distribution by World Region")
    plt.tight_layout()
    plt.show()

def scatter():
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        x="Economy (GDP per Capita)",
        y=target_regression,
        data=df,
        hue=target_classification,
        palette="RdYlGn",
        s=70
    )
    plt.title("GDP per Capita vs. Happiness Score")
    plt.xlabel("Economy (GDP per Capita)")
    plt.ylabel("Happiness Score")
    plt.legend(title="Happiness Level")
    plt.show()

def clusters():
    cluster_avg = df.groupby("Cluster")[target_regression].mean().reset_index()
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Cluster", y=target_regression, data=cluster_avg, palette="plasma")
    plt.title("Average Happiness Score by Factor Cluster")
    plt.xlabel("Factor Cluster ID (K=3)")
    plt.ylabel("Average Happiness Score")
    plt.show()

def confusion_mat():
    cm = confusion_matrix(y_test_c, y_pred_c)
    labels = np.unique(y_cls)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title("Confusion Matrix – Decision Tree Classifier")
    plt.xlabel("Predicted Happiness Level")
    plt.ylabel("Actual Happiness Level")
    plt.show()
# ---------------- GUI ----------------
root = tk.Tk()
root.title("🌍 World Happiness Analysis Dashboard")
root.geometry("1200x700")
root.configure(bg="#f8fafc")

# ---------- Header ----------
header = tk.Frame(root, bg="#4f46e5", height=70)
header.pack(fill="x")

tk.Label(
    header,
    text="🌍 World Happiness Analysis Dashboard",
    bg="#4f46e5",
    fg="white",
    font=("Segoe UI", 22, "bold")
).pack(pady=15)

# ---------- Main Layout ----------
main = tk.Frame(root, bg="#f8fafc")
main.pack(expand=True, fill="both")

sidebar = tk.Frame(main, bg="#312e81", width=240)
sidebar.pack(side="left", fill="y")

content = tk.Frame(main, bg="#f8fafc")
content.pack(side="right", expand=True, fill="both", padx=20, pady=20)

def clear():
    for w in content.winfo_children():
        w.destroy()

# ---------- Sidebar Button Style (same shape, new colors) ----------
def nav_button(parent, text, cmd):
    btn = tk.Button(
        parent,
        text=text,
        command=cmd,
        bg="#6366f1",
        fg="white",
        activebackground="#818cf8",
        activeforeground="white",
        font=("Segoe UI", 12, "bold"),
        bd=0,
        relief="flat",
        padx=15,
        pady=12,
        anchor="w"
    )
    btn.pack(fill="x", pady=8, padx=10)

    # Hover effect
    btn.bind("<Enter>", lambda e: btn.config(bg="#818cf8"))
    btn.bind("<Leave>", lambda e: btn.config(bg="#6366f1"))
    return btn

# ---------- Pages ----------
def dashboard():
    clear()
    card = tk.Frame(content, bg="white", bd=0)
    card.place(relx=0.5, rely=0.5, anchor="center", width=440, height=290)

    tk.Label(card, text="📊 Model Performance",
             bg="white", fg="#4f46e5",
             font=("Segoe UI", 18, "bold")).pack(pady=18)

    for label, value in [
        ("R² Score", f"{R2:.3f}"),
        ("RMSE", f"{RMSE:.3f}"),
        ("Accuracy", f"{ACC:.3f}")
    ]:
        row = tk.Frame(card, bg="white")
        row.pack(pady=8)
        tk.Label(row, text=f"{label} : ",
                 bg="white", fg="#334155",
                 font=("Segoe UI", 13)).pack(side="left")
        tk.Label(row, text=value,
                 bg="white", fg="#6366f1",
                 font=("Segoe UI", 13, "bold")).pack(side="left")


def eda():
    clear()
    card = tk.Frame(content, bg="white")
    card.place(relx=0.5, rely=0.5, anchor="center", width=540, height=390)

    tk.Label(card, text="📈 Exploratory Data Analysis",
             bg="white", fg="#4f46e5",
             font=("Segoe UI", 18, "bold")).pack(pady=18)

    def plot_btn(txt, cmd):
        b = tk.Button(
            card, text=txt, command=cmd,
            bg="#6366f1", fg="white",
            activebackground="#818cf8",
            activeforeground="white",
            font=("Segoe UI", 11, "bold"),
            bd=0, relief="flat",
            padx=28, pady=10
        )
        b.pack(pady=7)

    plot_btn("Correlation Heatmap", heatmap)
    plot_btn("Happiness Distribution", distribution)
    plot_btn("Region Box Plot", region_box)
    plot_btn("GDP vs Happiness Scatter", scatter)
    plot_btn("Confusion Matrix (Decision Tree)", confusion_mat)


def clustering_page():
    clear()
    card = tk.Frame(content, bg="white")
    card.place(relx=0.5, rely=0.5, anchor="center", width=440, height=230)

    tk.Label(card, text="🔍 Clustering Analysis",
             bg="white", fg="#4f46e5",
             font=("Segoe UI", 18, "bold")).pack(pady=22)

    btn = tk.Button(
        card, text="Show Factor Clusters",
        command=clusters,
        bg="#6366f1", fg="white",
        activebackground="#818cf8",
        activeforeground="white",
        font=("Segoe UI", 12, "bold"),
        bd=0, relief="flat",
        padx=32, pady=12
    )
    btn.pack()

# ---------- Sidebar Buttons ----------
nav_button(sidebar, "🏠  Dashboard", dashboard)
nav_button(sidebar, "📊  EDA & Models", eda)
nav_button(sidebar, "🔍  Clustering", clustering_page)

dashboard()
root.mainloop()
