import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_silhouette(X, labels) -> float | None:
    """Silhouette defined only if there are >=2 clusters and not all points are noise."""
    labels = np.asarray(labels)
    uniq = set(labels.tolist())
    if len(uniq) < 2:
        return None

    # DBSCAN: optionally exclude noise (-1)
    if -1 in uniq:
        mask = labels != -1
        if mask.sum() < 3:
            return None
        if len(set(labels[mask].tolist())) < 2:
            return None
        return silhouette_score(X[mask], labels[mask])

    return silhouette_score(X, labels)


def plot_elbow(ks, inertias, outpath, title):
    plt.figure()
    plt.plot(ks, inertias, marker="o")
    plt.title(title)
    plt.xlabel("k (n_clusters)")
    plt.ylabel("Inertia (SSE)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_silhouette_curve(xs, sils, outpath, title, xlabel):
    plt.figure()
    plt.plot(xs, sils, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Silhouette")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_2d_scatter(X2, labels, outpath, title):
    X2 = np.asarray(X2)
    if X2.ndim != 2 or X2.shape[1] < 2:
        raise ValueError(f"Expected 2D array with >=2 dims, got {X2.shape}")

    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=12)
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def boxplots_by_cluster(df, numeric_cols, cluster_col, outdir):
    for col in numeric_cols:
        groups = []
        labels = []
        for cl in sorted(df[cluster_col].dropna().unique()):
            vals = df.loc[df[cluster_col] == cl, col].dropna().values
            if len(vals) > 0:
                groups.append(vals)
                labels.append(str(cl))

        if len(groups) < 2:
            continue

        plt.figure()
        plt.boxplot(groups, labels=labels, showfliers=False)
        plt.title(f"{col} by cluster")
        plt.xlabel("cluster")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"boxplot_{col}.png"), dpi=150)
        plt.close()


def categorical_distribution(df, cat_cols, cluster_col, outdir, top_n=10):
    """
    Для интерпретации категорий: доля значений по кластерам.
    Рисуем bar-chart для самых частых категорий.
    """
    for col in cat_cols:
        # частые категории, чтобы графики были читабельными
        top_cats = df[col].value_counts(dropna=False).head(top_n).index.tolist()
        sub = df[df[col].isin(top_cats)].copy()

        # доли внутри кластера
        pivot = (
            sub.pivot_table(index=cluster_col, columns=col, aggfunc="size", fill_value=0)
        )
        pivot = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

        if pivot.shape[0] < 2:
            continue

        plt.figure()
        pivot.plot(kind="bar", ax=plt.gca())
        plt.title(f"{col}: category share by cluster (top {top_n})")
        plt.xlabel("cluster")
        plt.ylabel("share")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"catshare_{col}.png"), dpi=150)
        plt.close()


# -----------------------------
# Data load / EDA
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Убираем лишние пробелы в названиях колонок (часто в Kaggle файле есть)
    df.columns = df.columns.str.strip()

    # Приводим NA/unknown маркеры к NaN
    df = df.replace({"NA": np.nan, "N/A": np.nan, "nan": np.nan})
    return df


def basic_eda(df: pd.DataFrame, outdir: str) -> None:
    ensure_dir(outdir)

    # 1) общая инфа
    info_path = os.path.join(outdir, "eda_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("Shape:\n")
        f.write(str(df.shape) + "\n\n")
        f.write("DTypes:\n")
        f.write(str(df.dtypes) + "\n\n")
        f.write("Missing values (count):\n")
        f.write(str(df.isna().sum().sort_values(ascending=False)) + "\n\n")
        f.write("Describe (numeric):\n")
        f.write(str(df.select_dtypes(include=[np.number]).describe()) + "\n")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        plt.figure()
        df[col].dropna().hist(bins=30)
        plt.title(f"Histogram: {col}")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"hist_{col}.png"), dpi=150)
        plt.close()

    cat_cols = [c for c in df.columns if c not in num_cols]
    for col in cat_cols:
        vc = df[col].astype("string").fillna("MISSING").value_counts().head(15)
        plt.figure()
        vc.plot(kind="bar", ax=plt.gca())
        plt.title(f"Top categories: {col}")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"bar_{col}.png"), dpi=150)
        plt.close()


# -----------------------------
# Preprocessing (numeric + categorical -> scaled matrix)
# -----------------------------
def build_preprocessor(cat_cols, num_cols) -> Pipeline:
    """
    1) OneHot for categories (handle_unknown ignore)
    2) StandardScaler for numeric
    3) MaxAbsScaler on the final matrix (safe for sparse)
    """
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )

    prep_and_scale = Pipeline(steps=[
        ("prep", preprocess),
        ("scale_all", MaxAbsScaler()),
    ])
    return prep_and_scale


# -----------------------------
# Model selection: KMeans / Agglo / DBSCAN
# -----------------------------
def tune_kmeans(X_50, outdir, k_min=2, k_max=12, random_state=42):
    ks = list(range(k_min, k_max + 1))
    inertias = []
    sils = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X_50)
        inertias.append(km.inertia_)
        sils.append(safe_silhouette(X_50, labels))

    plot_elbow(ks, inertias, os.path.join(outdir, "kmeans_elbow.png"), "KMeans Elbow")
    plot_silhouette_curve(
        ks,
        [s if s is not None else np.nan for s in sils],
        os.path.join(outdir, "kmeans_silhouette.png"),
        "KMeans Silhouette vs k",
        "k",
    )

    # best k by silhouette (ignore nan)
    best = max([(k, s) for k, s in zip(ks, sils) if s is not None], key=lambda x: x[1], default=None)
    return best  # (k, silhouette)


def tune_agglo(X_50, outdir, k_min=2, k_max=12):
    ks = list(range(k_min, k_max + 1))
    sils = []

    for k in ks:
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(X_50)
        sils.append(safe_silhouette(X_50, labels))

    plot_silhouette_curve(
        ks,
        [s if s is not None else np.nan for s in sils],
        os.path.join(outdir, "agglo_silhouette.png"),
        "Agglomerative (Ward) Silhouette vs k",
        "k",
    )

    # Elbow method в классическом смысле (inertia) для Agglo не определён как у KMeans.
    # Для "аналогии" используют dendrogram / distance jumps, но это отдельная процедура.
    # В этом решении: оптимальный k выбираем по silhouette.

    best = max([(k, s) for k, s in zip(ks, sils) if s is not None], key=lambda x: x[1], default=None)
    return best


def tune_dbscan(X_50, outdir, eps_grid=None, min_samples_grid=None):
    if eps_grid is None:
        eps_grid = np.linspace(0.6, 3.0, 13)  # под 50D это более реалистично, чем 0.2..1.0
    if min_samples_grid is None:
        min_samples_grid = [5, 10, 15]

    rows = []
    best = None

    for ms in min_samples_grid:
        sils = []
        epss = []
        clusters = []
        noise_frac = []

        for eps in eps_grid:
            db = DBSCAN(eps=float(eps), min_samples=int(ms))
            labels = db.fit_predict(X_50)

            n_clusters = len(set(labels.tolist())) - (1 if -1 in labels else 0)
            frac_noise = float(np.mean(labels == -1))

            s = safe_silhouette(X_50, labels)

            rows.append({
                "eps": float(eps),
                "min_samples": int(ms),
                "n_clusters": int(n_clusters),
                "noise_frac": frac_noise,
                "silhouette": s if s is not None else np.nan,
            })

            epss.append(float(eps))
            sils.append(s if s is not None else np.nan)
            clusters.append(n_clusters)
            noise_frac.append(frac_noise)

            if s is not None:
                cand = (s, eps, ms, n_clusters, frac_noise)
                if best is None or cand[0] > best[0]:
                    best = cand

        # silhouette curve for each min_samples
        plot_silhouette_curve(
            epss,
            sils,
            os.path.join(outdir, f"dbscan_silhouette_ms{ms}.png"),
            f"DBSCAN Silhouette vs eps (min_samples={ms})",
            "eps",
        )

    pd.DataFrame(rows).to_csv(os.path.join(outdir, "dbscan_grid.csv"), index=False)
    return best  # (silhouette, eps, min_samples, n_clusters, noise_frac)


# -----------------------------
# Dim reduction (2D): PCA-like (SVD), UMAP, t-SNE
# -----------------------------
def reduce_2d(X_enc, X_50, method: str, random_state=42):
    method = method.lower()

    if method == "pca":
        svd2 = TruncatedSVD(n_components=2, random_state=random_state)
        return svd2.fit_transform(X_enc)

    if method == "umap":
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=random_state,
        )
        return reducer.fit_transform(X_50)

    if method == "tsne":
        from sklearn.manifold import TSNE
        n = X_50.shape[0]
        perplexity = min(30, max(2, (n - 1) // 3))
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=random_state,
        )
        return reducer.fit_transform(X_50)

    raise ValueError("method must be one of: pca, umap, tsne")


# -----------------------------
# Interpretation
# -----------------------------
def interpret_clusters(df_original, labels, outdir, method_name: str, cat_cols, num_cols):
    ensure_dir(outdir)
    df = df_original.copy()
    df[f"cluster_{method_name}"] = labels

    # 1) Числовые средние по кластерам
    means = df.groupby(f"cluster_{method_name}")[num_cols].mean(numeric_only=True)
    means.to_csv(os.path.join(outdir, f"{method_name}_numeric_means.csv"))

    # 2) Категории: мода (top1) и доли (top categories)
    cat_summary = []
    for col in cat_cols:
        grp = df.groupby(f"cluster_{method_name}")[col]
        for cl, series in grp:
            vc = series.astype("string").fillna("MISSING").value_counts(normalize=True).head(5)
            row = {"cluster": cl, "feature": col}
            for i, (k, v) in enumerate(vc.items(), start=1):
                row[f"top{i}_cat"] = k
                row[f"top{i}_share"] = float(v)
            cat_summary.append(row)

    pd.DataFrame(cat_summary).to_csv(os.path.join(outdir, f"{method_name}_categorical_top.csv"), index=False)

    # 3) Boxplots по числовым признакам
    boxplots_by_cluster(df, num_cols, f"cluster_{method_name}", outdir)

    # 4) Распределения категорий (bar)
    categorical_distribution(df, cat_cols, f"cluster_{method_name}", outdir, top_n=10)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="german_credit_data.csv", help="Path to german credit CSV")
    parser.add_argument("--out", default="out", help="Output directory")
    parser.add_argument("--kmax", type=int, default=12, help="Max k for KMeans/Agglo")
    args = parser.parse_args()

    ensure_dir(args.out)

    df = load_data(args.csv)

    # --- Define columns ---
    # Числовые (в твоём датасете это обычно:)
    # Age, Job, Credit amount, Duration
    # Остальные: категориальные (Sex, Housing, Saving accounts, Checking account, Purpose)
    # (Если Job у тебя категориальный — перенеси в cat_cols, но в Kaggle это обычно 0..3)
    num_cols = ["Age", "Job", "Credit amount", "Duration"]
    cat_cols = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]

    # Проверки на наличие колонок
    missing = set(num_cols + cat_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # --- EDA ---
    basic_eda(df[num_cols + cat_cols], os.path.join(args.out, "eda"))

    # --- Preprocess + scale ---
    prep = build_preprocessor(cat_cols, num_cols)
    X_raw = df[num_cols + cat_cols].copy()

    X_enc = prep.fit_transform(X_raw)  # sparse matrix (likely)
    # Reduce to 50 dims for clustering stability/speed
    svd50 = TruncatedSVD(n_components=50, random_state=42)
    X_50 = svd50.fit_transform(X_enc)  # dense (n,50)

    # --- Modeling / tuning ---
    model_dir = os.path.join(args.out, "modeling")
    ensure_dir(model_dir)

    best_kmeans = tune_kmeans(X_50, model_dir, k_min=2, k_max=args.kmax)
    best_agglo = tune_agglo(X_50, model_dir, k_min=2, k_max=args.kmax)
    best_dbscan = tune_dbscan(X_50, model_dir)

    with open(os.path.join(model_dir, "best_params.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best KMeans (k, silhouette): {best_kmeans}\n")
        f.write(f"Best Agglo  (k, silhouette): {best_agglo}\n")
        f.write(f"Best DBSCAN (sil, eps, min_samples, n_clusters, noise_frac): {best_dbscan}\n")

    # --- Fit final models using best params ---
    final_labels = {}

    if best_kmeans:
        k = best_kmeans[0]
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        final_labels["kmeans"] = km.fit_predict(X_50)

    if best_agglo:
        k = best_agglo[0]
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        final_labels["agglo"] = agg.fit_predict(X_50)

    if best_dbscan:
        _, eps, ms, _, _ = best_dbscan
        db = DBSCAN(eps=float(eps), min_samples=int(ms))
        final_labels["dbscan"] = db.fit_predict(X_50)

    # --- Dim reduction + visualization ---
    viz_dir = os.path.join(args.out, "viz")
    ensure_dir(viz_dir)

    for red in ["pca", "umap", "tsne"]:
        X2 = reduce_2d(X_enc, X_50, red, random_state=42)
        for name, labels in final_labels.items():
            plot_2d_scatter(
                X2,
                labels,
                os.path.join(viz_dir, f"{name}_{red}.png"),
                f"{name.upper()} on {red.upper()} 2D",
            )

    interp_dir = os.path.join(args.out, "interpretation")
    ensure_dir(interp_dir)

    for name, labels in final_labels.items():
        interpret_clusters(
            df_original=df[num_cols + cat_cols],
            labels=labels,
            outdir=os.path.join(interp_dir, name),
            method_name=name,
            cat_cols=cat_cols,
            num_cols=num_cols,
        )

    print(f"Done. Results saved to: {args.out}")


if __name__ == "__main__":
    main()