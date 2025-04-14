import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import plotly.express as px
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def select_csv_file():
    print("📁 Seleziona un file CSV...")
    Tk().withdraw()  # Nasconde la finestra principale di Tk
    file_path = askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        print("❌ Nessun file selezionato. Uscita.")
        sys.exit(0)
    return file_path

def load_dataset(csv_path):
    try:
        data = pd.read_csv(csv_path)
        print(f"\n✅ Dataset caricato: {data.shape[0]} righe, {data.shape[1]} colonne\n")
        return data
    except Exception as e:
        print(f"❌ Errore nel caricamento del file: {e}")
        sys.exit(1)

def get_categorical_columns(data):
    return data.select_dtypes(include=['object', 'category']).columns.tolist()

def select_categorical_column(categorical_cols):
    print("🔹 Colonne categoriali disponibili:")
    for idx, col in enumerate(categorical_cols):
        print(f"  [{idx}] {col}")
    
    choice = input("\n👉 Seleziona l'indice della colonna da usare per la colorazione (o premi invio per saltare): ")
    if choice.strip() == "":
        return None

    try:
        col_index = int(choice)
        if 0 <= col_index < len(categorical_cols):
            return categorical_cols[col_index]
        else:
            print("❌ Indice non valido.")
            return None
    except ValueError:
        print("❌ Input non valido.")
        return None

def choose_n_components(max_components):
    print(f"\n🔢 Puoi selezionare un numero di componenti da 2 a {max_components}")
    while True:
        choice = input(f"Inserisci il numero di componenti PCA (default=2): ").strip()
        if choice == "":
            return 2
        try:
            n = int(choice)
            if 2 <= n <= max_components:
                return n
            else:
                print(f"⚠️ Inserisci un valore compreso tra 2 e {max_components}")
        except ValueError:
            print("❌ Input non valido.")

def perform_pca(data, n_components):
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if numeric_data.empty:
        print("❌ Nessuna colonna numerica trovata per la PCA.")
        sys.exit(1)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_

    pc_df = pd.DataFrame(data=principal_components,
                         columns=[f'PC{i+1}' for i in range(n_components)])

    print(f"\n📊 Varianza spiegata dalle prime {n_components} componenti:")
    for i, var in enumerate(explained_variance):
        print(f"   - PC{i+1}: {var*100:.2f}%")

    return pc_df, explained_variance

def plot_pca(pc_df, explained_variance, color_labels=None, color_column_name=None):
    n_components = pc_df.shape[1]

    if n_components == 2:
        plt.figure(figsize=(8,6))
        sns.scatterplot(x='PC1', y='PC2', data=pc_df, hue=color_labels, palette='Set2', s=60)
        plt.title("PCA - Componenti 1 e 2")
        plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)')
        plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)')
        if color_labels is not None:
            plt.legend(title=color_column_name, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif n_components == 3:
        pc_df[color_column_name] = color_labels if color_labels is not None else "N/A"
        fig = px.scatter_3d(
            pc_df, x='PC1', y='PC2', z='PC3',
            color=color_column_name,
            title='PCA 3D',
            labels={f'PC{i+1}': f'PC{i+1} ({explained_variance[i]*100:.2f}%)' for i in range(3)}
        )
        fig.show()

    else:
        plt.figure(figsize=(10, 6))
        pc_subset = pc_df.iloc[:, :min(10, n_components)]
        sns.heatmap(pc_subset.T, cmap='coolwarm', cbar=True, xticklabels=False)
        plt.title("Heatmap delle componenti principali")
        plt.ylabel("Componenti")
        plt.xlabel("Osservazioni")
        plt.tight_layout()
        plt.show()

def main():
    csv_path = select_csv_file()
    data = load_dataset(csv_path)
    categorical_cols = get_categorical_columns(data)

    max_components = min(data.select_dtypes(include=['float64', 'int64']).shape[1], 20)
    n_components = choose_n_components(max_components)
    pc_df, explained_variance = perform_pca(data, n_components)

    if not categorical_cols:
        print("⚠️ Nessuna colonna categoriale trovata. Il grafico non sarà colorato.")
        plot_pca(pc_df, explained_variance)
        return

    while True:
        color_column = select_categorical_column(categorical_cols)
        if color_column is None:
            print("\n👋 Uscita dal programma.")
            break

        color_labels = data[color_column]
        plot_pca(pc_df, explained_variance, color_labels, color_column)

        again = input("\n🔁 Vuoi selezionare un'altra colonna o cambiare numero di componenti? (s/n): ").strip().lower()
        if again == "s":
            n_components = choose_n_components(max_components)
            pc_df, explained_variance = perform_pca(data, n_components)
        else:
            print("\n👋 Uscita dal programma.")
            break

if __name__ == "__main__":
    main()
