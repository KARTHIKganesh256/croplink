import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gpd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import requests
import joblib
import json

sns.set_style("whitegrid")

# Base directory
BASE_DIR = "K:/html project/cropk"
os.makedirs(BASE_DIR, exist_ok=True)

# Helper for repeating colors
def color_list_for_n(colors, n):
    if isinstance(colors, str):
        return [colors] * n
    if len(colors) >= n:
        return colors[:n]
    repeats = (n // len(colors)) + 1
    return (colors * repeats)[:n]

# Color sets
class colorss:
    yellows = ['#ffffd4','#fee391','#fec44f','#fe9929','#d95f0e','#993404','#a70000','#ff5252','#ff7b7b','#ffbaba']
    greens = ['#ffffd4','#fee391','#fec44f','#fe9929','#d9f0a3','#addd8e','#78c679','#41ab5d','#238443','#005a32']

# Download dataset if not present
def download_dataset_if_needed(file_name=os.path.join(BASE_DIR, "yield_df.csv"), 
                             url="https://raw.githubusercontent.com/researchpy/Data-sets/master/crop_yield.csv"):
    if os.path.exists(file_name):
        print(f"{file_name} exists.")
        return True
    try:
        print(f"Downloading {file_name}...")
        response = requests.get(url)
        response.raise_for_status()
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {file_name} successfully.")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        print("Manual download: Go to https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset, "
              "download yield_df.csv, and place it in K:/html project/cropk")
        return False

# Load dataset
def load_dataset(file_path=os.path.join(BASE_DIR, "yield_df.csv")):
    if not download_dataset_if_needed(file_path):
        raise FileNotFoundError("Dataset not found and download failed.")
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully:")
        print(df.head())
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Failed to load '{file_path}': {e}")
        raise

# Load dataset
try:
    df = load_dataset()
except Exception as e:
    print(f"Error: {e}")
    exit()

# Global sample for plots
df_sample = df.sample(n=min(2000, len(df)), random_state=42)

# Check for required columns
required_columns = ['hg/ha_yield', 'Area', 'Item', 'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Warning: Missing columns {missing_columns}. Some plots may be skipped.")
    print("Rename columns if needed, e.g.: df.rename(columns={'old_name': 'hg/ha_yield'}, inplace=True)")

# Remove unnamed columns
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Remove countries with less than 100 records
if 'Area' in df.columns:
    country_counts = df['Area'].value_counts()
    countries_to_drop = country_counts[country_counts < 100].index.tolist()
    df = df[~df['Area'].isin(countries_to_drop)].reset_index(drop=True)

# Label encoding
datacorr = df.copy()
categorical_columns = datacorr.select_dtypes(include=['object']).columns.tolist()
encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    datacorr[column] = le.fit_transform(datacorr[column].astype(str))
    encoders[column] = le
    joblib.dump(le, os.path.join(BASE_DIR, f"{column}_encoder.pkl"))
    print(f"Saved {column}_encoder.pkl")

# List of generated files
generated_files = []

# Correlation heatmap
try:
    plt.figure(figsize=(12, 9))
    sns.heatmap(datacorr.corr(), annot=True, cmap='PuOr', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.tight_layout()
    file_path = os.path.join(BASE_DIR, "correlation_heatmap.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append("correlation_heatmap.png")
    print(f"Saved {file_path}")
except Exception as e:
    print(f"Correlation heatmap skipped: {e}")

# Histograms
num_cols = df.select_dtypes(include=[np.number]).columns
try:
    df[num_cols].hist(figsize=(12, 10))
    plt.tight_layout()
    file_path = os.path.join(BASE_DIR, "histograms.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append("histograms.png")
    print(f"Saved {file_path}")
except Exception as e:
    print(f"Histograms skipped: {e}")

# Pairplot
try:
    sample_size = min(1000, len(df))
    pairplot_sample = df.sample(n=sample_size, random_state=42)
    pairplot_cols = pairplot_sample.select_dtypes(include=[np.number]).columns.tolist()[:6]
    pairplot_df = pairplot_sample[pairplot_cols + ['Item']] if 'Item' in pairplot_sample.columns else pairplot_sample[pairplot_cols]
    sns.pairplot(pairplot_df, hue='Item' if 'Item' in pairplot_df.columns else None, kind='scatter', palette='BrBG', diag_kind='hist')
    file_path = os.path.join(BASE_DIR, "pairplot.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    generated_files.append("pairplot.png")
    print(f"Saved {file_path}")
except Exception as e:
    print(f"Pairplot skipped: {e}")

# Yams yield over years
if all(col in df.columns for col in ['Item', 'hg/ha_yield', 'Year']):
    try:
        df2 = df[df['Item'] == 'Yams']
        if not df2.empty:
            df2.groupby('Year')['hg/ha_yield'].mean().plot(color='brown', figsize=(10, 5))
            plt.title('Yams Yield Over Years')
            plt.tight_layout()
            file_path = os.path.join(BASE_DIR, "yams_yield.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files.append("yams_yield.png")
            print(f"Saved {file_path}")
    except Exception as e:
        print(f"Yams yield plot skipped: {e}")

# Geospatial plot
geojson_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
if all(col in df.columns for col in ['hg/ha_yield', 'Area']):
    try:
        data = gpd.read_file(geojson_url)
        data['NAME_norm'] = data['NAME'].str.strip().str.lower()
        df['Area_norm'] = df['Area'].astype(str).str.strip().str.lower()
        merged_data = data.merge(df, left_on='NAME_norm', right_on='Area_norm', how='left')
        if not merged_data.empty and 'hg/ha_yield' in merged_data.columns:
            fig, ax = plt.subplots(figsize=(15, 10))
            merged_data.plot(column='hg/ha_yield', cmap='Greens_r', linewidth=0.4, edgecolor='0.8', ax=ax, legend=True)
            plt.title("Crop Yield by Country")
            plt.tight_layout()
            file_path = os.path.join(BASE_DIR, "geospatial_plot.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files.append("geospatial_plot.png")
            print(f"Saved {file_path}")
        else:
            print("Geospatial plot skipped: No matching data after merge.")
        del merged_data, data
    except Exception as e:
        print(f"Geospatial plot skipped: {e}")
else:
    print("Geospatial plot skipped: Required columns missing.")

# Area-based histograms
if 'Area' in df.columns and 'hg/ha_yield' in df.columns:
    unique_areas = sorted(df['Area'].unique())
    areas_per_plot = 10
    area_chunks = [unique_areas[i:i+areas_per_plot] for i in range(0, len(unique_areas), areas_per_plot)]
    num_plots = len(area_chunks)
    try:
        fig, axs = plt.subplots(ncols=min(7, num_plots), figsize=(30, 10))
        axs = np.atleast_1d(axs)
        for i, ax in enumerate(axs[:num_plots]):
            plot_areas = area_chunks[i]
            plot_df = df[df['Area'].isin(plot_areas)]
            areas_in_plot = plot_df['Area'].unique()
            colors = color_list_for_n(colorss.greens, len(areas_in_plot))
            for j, area in enumerate(areas_in_plot):
                data_ = plot_df[plot_df['Area'] == area]['hg/ha_yield'].dropna()
                if not data_.empty:
                    ax.hist(data_, bins=15, alpha=0.7, label=area, color=colors[j])
            ax.legend(fontsize='small', loc='upper right')
            ax.set_title(f'Yield Distribution - Group {i+1}')
        plt.tight_layout()
        file_path = os.path.join(BASE_DIR, "area_histograms.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append("area_histograms.png")
        print(f"Saved {file_path}")
    except Exception as e:
        print(f"Area-based histograms skipped: {e}")

# Yield statistics by area and item
yield_stats = []
if all(col in df.columns for col in ['Area', 'Item', 'hg/ha_yield']):
    for i in range(min(7, num_plots)):
        plot_df = df[df['Area'].isin(area_chunks[i])]
        if not plot_df.empty:
            try:
                dk = plot_df.groupby(['Area','Item'])['hg/ha_yield'].mean().to_frame()
                dg = dk.sort_values(by=['hg/ha_yield'], ascending=False)
                print(f"\nTop 5 Yields for Group {i+1}:")
                try:
                    from IPython.display import display
                    display(dg.head())
                except (ImportError, NameError):
                    print(dg.head())
                yield_stats.append(dg.head().reset_index().to_html(index=False, classes='table-auto w-full border-collapse'))
            except Exception as e:
                print(f"Yield statistics for Group {i+1} skipped: {e}")

# Rainfall and pesticides plots
if all(col in df.columns for col in ['Area', 'average_rain_fall_mm_per_year', 'pesticides_tonnes']):
    for i in range(min(3, num_plots)):
        plot_df = df[df['Area'].isin(area_chunks[i])]
        if plot_df.empty:
            continue
        try:
            if 'average_rain_fall_mm_per_year' in plot_df.columns:
                avg_rain = plot_df.groupby('Area')['average_rain_fall_mm_per_year'].mean().sort_values(ascending=False)
                colors_for_bars = color_list_for_n(colorss.greens, len(avg_rain))
                plt.figure(figsize=(10, 6))
                avg_rain.plot(kind='bar', color=colors_for_bars)
                plt.xticks(rotation=90)
                plt.title(f'Average Rainfall by Area - Group {i+1}')
                plt.tight_layout()
                file_path = os.path.join(BASE_DIR, f"rainfall_group_{i+1}.png")
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                generated_files.append(f"rainfall_group_{i+1}.png")
                print(f"Saved {file_path}")
        except Exception as e:
            print(f"Average rainfall plot for Group {i+1} skipped: {e}")
        
        try:
            if 'pesticides_tonnes' in plot_df.columns:
                avg_pest = plot_df.groupby('Area')['pesticides_tonnes'].mean().sort_values(ascending=False)
                colors_for_bars = color_list_for_n(colorss.yellows, len(avg_pest))
                plt.figure(figsize=(10, 6))
                avg_pest.plot(kind='bar', color=colors_for_bars)
                plt.xticks(rotation=90)
                plt.title(f'Average Pesticides by Area - Group {i+1}')
                plt.tight_layout()
                file_path = os.path.join(BASE_DIR, f"pesticides_group_{i+1}.png")
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                generated_files.append(f"pesticides_group_{i+1}.png")
                print(f"Saved {file_path}")
        except Exception as e:
            print(f"Average pesticides plot for Group {i+1} skipped: {e}")

# Combined pesticides and yield plot
if all(col in df.columns for col in ['Area', 'pesticides_tonnes', 'hg/ha_yield']):
    for i in range(min(3, num_plots)):
        plot_df = df[df['Area'].isin(area_chunks[i])]
        if plot_df.empty:
            continue
        try:
            agg = plot_df.groupby('Area')[['pesticides_tonnes', 'hg/ha_yield']].mean().dropna()
            if not agg.empty:
                plt.figure(figsize=(12, 6))
                agg.plot(kind='bar', color=color_list_for_n(colorss.yellows, agg.shape[0]*2))
                plt.xticks(rotation=90)
                plt.title(f'Pesticides and Yield by Area - Group {i+1}')
                plt.tight_layout()
                file_path = os.path.join(BASE_DIR, f"combined_group_{i+1}.png")
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                generated_files.append(f"combined_group_{i+1}.png")
                print(f"Saved {file_path}")
        except Exception as e:
            print(f"Combined plot for Group {i+1} skipped: {e}")

# Scatter plot for pesticides vs yield
if all(col in df.columns for col in ['hg/ha_yield', 'pesticides_tonnes', 'Area']):
    try:
        fig = px.scatter(df_sample, x='hg/ha_yield', y='pesticides_tonnes', color="Area", title="Pesticides vs Yield", width=900, height=500)
        file_path = os.path.join(BASE_DIR, "pesticides_scatter.png")
        fig.write_image(file_path, format="png")
        generated_files.append("pesticides_scatter.png")
        print(f"Saved {file_path}")
    except Exception as e:
        print(f"Pesticides vs Yield scatter plot skipped: {e}")
else:
    print("Pesticides vs Yield scatter plot skipped: Required columns missing.")

# Item-based pesticides bar plot
if all(col in df.columns for col in ['Item', 'pesticides_tonnes']):
    try:
        plt.figure(figsize=(14, 6))
        item_pest = df.groupby('Item')['pesticides_tonnes'].mean().sort_values(ascending=False)
        sns.barplot(x=item_pest.index, y=item_pest.values, palette='BrBG')
        plt.xticks(rotation=90)
        plt.title('Pesticides Usage by Crop Item')
        plt.tight_layout()
        file_path = os.path.join(BASE_DIR, "pesticides_by_item.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append("pesticides_by_item.png")
        print(f"Saved {file_path}")
    except Exception as e:
        print(f"Item-based pesticides plot skipped: {e}")

# Box plot + temp scatter overlay
if all(col in df.columns for col in ['Item', 'hg/ha_yield', 'avg_temp']):
    try:
        fig, ax = plt.subplots(figsize=(16.7, 8.27))
        sns.boxplot(x="Item", y="hg/ha_yield", palette="BrBG", data=df, ax=ax)
        scatter_sample = df.sample(n=min(1000, len(df)), random_state=42)
        sns.scatterplot(x='Item', y='avg_temp', data=scatter_sample, size=10, color='y', ax=ax, legend=False)
        plt.xticks(rotation=90)
        plt.title('Yield Distribution and Average Temperature by Item')
        plt.tight_layout()
        file_path = os.path.join(BASE_DIR, "box_temp_overlay.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append("box_temp_overlay.png")
        print(f"Saved {file_path}")
    except Exception as e:
        print(f"Box plot with temperature overlay skipped: {e}")

# Best areas for each item
if all(col in df.columns for col in ['Item', 'Area', 'hg/ha_yield']):
    try:
        best_areas = []
        grouped = df.groupby('Item')
        for item, group in grouped:
            max_idx = group['hg/ha_yield'].idxmax()
            if pd.isna(max_idx):
                continue
            area = group.loc[max_idx, 'Area']
            production = group.loc[max_idx, 'hg/ha_yield']
            best_areas.append({'Item': item, 'Area': area, 'hg/ha_yield': production})
        best_areas_df = pd.DataFrame(best_areas)
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=best_areas_df, x='hg/ha_yield', y='Area', hue='Item', dodge=False, palette=color_list_for_n(colorss.yellows, len(best_areas_df)))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('Best Areas for Each Crop by Yield')
        plt.tight_layout()
        file_path = os.path.join(BASE_DIR, "best_areas.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append("best_areas.png")
        print(f"Saved {file_path}")
    except Exception as e:
        print(f"Best areas plot skipped: {e}")

# Multiple scatter/line plots
if all(col in df.columns for col in ['pesticides_tonnes', 'average_rain_fall_mm_per_year', 'avg_temp', 'Year', 'hg/ha_yield', 'Item']):
    try:
        fig, axes = plt.subplots(4, 1, figsize=(18, 22))
        plot_items = [
            ('pesticides_tonnes', 'Pesticides vs Yield'),
            ('average_rain_fall_mm_per_year', 'Rainfall vs Yield'),
            ('avg_temp', 'Temperature vs Yield'),
            ('Year', 'Yield Over Time')
        ]
        for ax, (x_col, title) in zip(axes, plot_items):
            if x_col not in df.columns:
                ax.text(0.5, 0.5, f"{x_col} missing", ha='center', va='center')
                continue
            data_use = df_sample if len(df) > 2000 else df
            if x_col == 'Year':
                sns.lineplot(x=x_col, y="hg/ha_yield", hue="Item", data=data_use, ax=ax, palette='Spectral')
            else:
                sns.scatterplot(x=x_col, y="hg/ha_yield", hue="Item", data=data_use, ax=ax, palette='Spectral', alpha=0.6)
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylabel('Average Yield')
            ax.set_title(title)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        file_path = os.path.join(BASE_DIR, "multiple_plots.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append("multiple_plots.png")
        print(f"Saved {file_path}")
    except Exception as e:
        print(f"Multiple scatter/line plots skipped: {e}")

# Change of years function
def change_of_years(data, template='seaborn'):
    try:
        plt.style.use(template)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        for i in numeric_cols:
            if i == 'Year':
                continue
            series = data.groupby('Year')[i].mean().dropna()
            if series.empty:
                continue
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=series.index, y=series.values)
            plt.title(f'Effect of Years on {i}')
            plt.tight_layout()
            file_path = os.path.join(BASE_DIR, f"change_of_years_{i}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files.append(f"change_of_years_{i}.png")
            print(f"Saved {file_path}")
    except Exception as e:
        print(f"Change of years plot skipped: {e}")

# Plot for all data and Egypt
if 'Year' in df.columns:
    change_of_years(df)
    if 'Area' in df.columns and 'Egypt' in df['Area'].unique():
        df_Egypt = df[df['Area'] == 'Egypt']
        change_of_years(df_Egypt)

# Model training and evaluation
best_model = None
best_r2 = -np.inf
model_coefficients = None
if 'hg/ha_yield' in datacorr.columns:
    try:
        X = datacorr.drop(labels='hg/ha_yield', axis=1)
        y = datacorr['hg/ha_yield']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        models = [
            ('Linear Regression', LinearRegression()),
            ('Random Forest', RandomForestRegressor(random_state=42)),
            ('Gradient Boost', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
            ('XGBoost', XGBRegressor(random_state=42, objective='reg:squarederror')),
            ('KNN', KNeighborsRegressor(n_neighbors=5)),
            ('Decision Tree', DecisionTreeRegressor(random_state=42)),
            ('Bagging Regressor', BaggingRegressor(n_estimators=150, random_state=42))
        ]

        results = []
        for name, model in models:
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                train_score = model.score(X_train, y_train) * 100
                test_score = model.score(X_test, y_test) * 100
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results.append((name, test_score, mse, r2))
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
                    if name == 'Linear Regression':
                        model_coefficients = {'coef': model.coef_.tolist(), 'intercept': float(model.intercept_)}
                print(f'\n{name} Model Performance:')
                print(f'Training Accuracy: {train_score:.2f}%')
                print(f'Testing Accuracy: {test_score:.2f}%')
                print(f'MSE: {mse:.2f}')
                print(f'R2 Score: {r2:.2f}')
                file_path = os.path.join(BASE_DIR, f"{name}_actual_vs_pred.png")
                plt.figure(figsize=(8, 6))
                plt.scatter(y_test, y_pred, s=10, color='#9B673C', alpha=0.6)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='green', linewidth=2)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.title(f'{name} - Actual vs Predicted')
                plt.tight_layout()
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                generated_files.append(f"{name}_actual_vs_pred.png")
                print(f"Saved {file_path}")
            except Exception as e:
                print(f"Model {name} failed: {e}")

        dff = pd.DataFrame(results, columns=['Model', 'Accuracy', 'MSE', 'R2_score'])
        try:
            from IPython.display import display
            display(dff.style.highlight_max(subset=['Accuracy', 'R2_score'], color='lightgreen').highlight_min(subset=['MSE'], color='lightgreen'))
        except (ImportError, NameError):
            print(dff)
        file_path = os.path.join(BASE_DIR, "model_results.csv")
        dff.to_csv(file_path, index=False)
        generated_files.append("model_results.csv")
        print(f"Saved {file_path}")

        # Second evaluation pass
        results2 = []
        for name, model in models:
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = model.score(X_test, y_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred) if not np.any(np.isclose(y_test, 0)) else np.nan
                r2 = r2_score(y_test, y_pred)
                results2.append((name, accuracy, mse, mae, mape, r2))
                print(f"\n{name} Model Performance (Second Pass):")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"MSE: {mse:.2f}")
                print(f"MAE: {mae:.2f}")
                print(f"MAPE: {mape:.4f}" if not np.isnan(mape) else "MAPE: N/A")
                print(f"R2 Score: {r2:.4f}")
                num_folds = 5
                kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
                try:
                    scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
                    for fold, score in enumerate(scores):
                        print(f"Fold {fold+1}: {score:.4f}")
                    mean_score = np.nanmean(scores)
                    print(f"Mean CV Score: {mean_score:.4f}")
                except Exception as e:
                    print(f"Cross-validation skipped for {name}: {e}")
                print('-'*30)
            except Exception as e:
                print(f"Model {name} failed in second pass: {e}")

        df_results = pd.DataFrame(results2, columns=['Model', 'Accuracy', 'MSE', 'MAE', 'MAPE', 'R2_score'])
        try:
            from IPython.display import display
            display(df_results.style.highlight_max(subset=['Accuracy', 'R2_score'], color='lightblue').highlight_min(subset=['MSE', 'MAE', 'MAPE'], color='lightblue'))
        except (ImportError, NameError):
            print(df_results)
        file_path = os.path.join(BASE_DIR, "detailed_results.csv")
        df_results.to_csv(file_path, index=False)
        generated_files.append("detailed_results.csv")
        print(f"Saved {file_path}")

        # Save best model
        if best_model:
            file_path = os.path.join(BASE_DIR, "best_model.pkl")
            joblib.dump(best_model, file_path)
            generated_files.append("best_model.pkl")
            print(f"Saved {file_path}")
    except Exception as e:
        print(f"Model training skipped: {e}")
else:
    print("Model training skipped: 'hg/ha_yield' not found.")

# Save model coefficients
if model_coefficients:
    file_path = os.path.join(BASE_DIR, "model_coefficients.json")
    with open(file_path, 'w') as f:
        json.dump(model_coefficients, f)
    generated_files.append("model_coefficients.json")
    print(f"Saved {file_path}")
else:
    file_path = os.path.join(BASE_DIR, "model_coefficients.json")
    with open(file_path, 'w') as f:
        json.dump({'coef': [0]*len(datacorr.columns.drop('hg/ha_yield')), 'intercept': 0}, f)
    generated_files.append("model_coefficients.json")
    print(f"Saved default {file_path}")

# Verify generated files
print("\nGenerated files:")
for file in os.listdir(BASE_DIR):
    print(file)

# Generate HTML
html_content = """
<div style="background-color: #f0f0f0;">Hello</div>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crop Yield Analysis Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .dark-mode {{ background-color: #1f2937; color: #f9fafb; }}
        .dark-mode .bg-white {{ background-color: #374151; }}
        .dark-mode .text-gray-800 {{ color: #f9fafb; }}
        .dark-mode .bg-gray-100 {{ background-color: #4b5563; }}
        .dark-mode table thead {{ background-color: #4b5563; }}
        .dark-mode table tbody tr:nth-child(even) {{ background-color: #4b5563; }}
        .dark-mode table tbody tr:hover {{ background-color: #6b7280; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
        th {{ background-color: #3b82f6; color: white; }}
        tbody tr:nth-child(even) {{ background-color: #f9fafb; }}
        tbody tr:hover {{ background-color: #e0f2fe; }}
        .placeholder-img {{ background-color: #f3f4f6; color: #6b7280; text-align: center; padding: 20px; border-radius: 8px; }}
    </style>
</head>
<body class="min-h-screen bg-gray-100">
    <div class="container mx-auto p-4 md:p-6">
        <div class="flex justify-between items-center mb-6">
            <h1 class="text-3xl font-bold text-gray-800">Crop Yield Analysis Dashboard</h1>
            <button id="darkModeToggle" class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg">Toggle Dark Mode</button>
        </div>
        <div class="bg-white p-6 rounded-lg shadow-lg">
            <p class="text-gray-600 mb-4">Generated on {current_date}. Dataset: {df_shape} rows, columns: {df_columns}</p>

            <h2 class="text-2xl font-semibold text-gray-800 mb-4">Debug Information</h2>
            <p id="debug-info" class="text-red-600 mb-4">Checking for missing files...</p>

            <h2 class="text-2xl font-semibold text-gray-800 mb-4">Key Visualizations</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                {visualizations}
            </div>

            <h2 class="text-2xl font-semibold text-gray-800 mt-8 mb-4">Model Performance</h2>
            {model_table}

            <h2 class="text-2xl font-semibold text-gray-800 mt-8 mb-4">Detailed Model Results</h2>
            {detailed_table}

            <h2 class="text-2xl font-semibold text-gray-800 mt-8 mb-4">Yield Statistics by Area and Item</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                {yield_stats_tables}
            </div>

            <h2 class="text-2xl font-semibold text-gray-800 mt-8 mb-4">Real-Time Prediction</h2>
            <div class="bg-gray-100 p-6 rounded-lg">
                <p class="text-gray-600 mb-4">Enter values to predict crop yield (hg/ha). Uses Linear Regression coefficients for demo purposes.</p>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-gray-700">Area (encoded)</label>
                        <input type="number" id="area" value="0" class="w-full p-2 border rounded-lg">
                    </div>
                    <div>
                        <label class="block text-gray-700">Item (encoded)</label>
                        <input type="number" id="item" value="0" class="w-full p-2 border rounded-lg">
                    </div>
                    <div>
                        <label class="block text-gray-700">Year</label>
                        <input type="number" id="year" value="2000" class="w-full p-2 border rounded-lg">
                    </div>
                    <div>
                        <label class="block text-gray-700">Avg Rain (mm)</label>
                        <input type="number" id="rain" value="1000" class="w-full p-2 border rounded-lg">
                    </div>
                    <div>
                        <label class="block text-gray-700">Pesticides (tonnes)</label>
                        <input type="number" id="pest" value="500" class="w-full p-2 border rounded-lg">
                    </div>
                    <div>
                        <label class="block text-gray-700">Avg Temp (°C)</label>
                        <input type="number" id="temp" value="25" class="w-full p-2 border rounded-lg">
                    </div>
                </div>
                <button onclick="predictYield()" class="mt-4 bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg">Predict Yield</button>
                <p id="real-time-pred" class="mt-4 text-lg font-semibold text-green-600">Prediction will appear here</p>
                <p class="text-gray-600 mt-2">For accurate predictions, deploy with a server (e.g., Flask) using the saved model.</p>
            </div>
        </div>
    </div>
    <script>
        // Dark mode toggle
        document.getElementById('darkModeToggle').addEventListener('click', () => {{
            document.body.classList.toggle('dark-mode');
        }});

        // Check for missing images
        const images = [
            './correlation_heatmap.png',
            './histograms.png',
            './pairplot.png',
            './geospatial_plot.png',
            './area_histograms.png',
            './best_areas.png',
            './pesticides_scatter.png',
            './multiple_plots.png',
            './Linear Regression_actual_vs_pred.png',
            './Random Forest_actual_vs_pred.png',
            './Gradient Boost_actual_vs_pred.png',
            './XGBoost_actual_vs_pred.png',
            './KNN_actual_vs_pred.png',
            './Decision Tree_actual_vs_pred.png',
            './Bagging Regressor_actual_vs_pred.png'
        ];
        let missingFiles = [];
        images.forEach(src => {{
            const img = new Image();
            img.src = src;
            img.onerror = () => missingFiles.push(src);
        }});
        setTimeout(() => {{
            document.getElementById('debug-info').innerText = missingFiles.length ? 
                `Missing files: ${{missingFiles.join(', ')}}` : 
                'All files loaded successfully';
        }}, 1000);

        // Load model coefficients
        fetch('./model_coefficients.json')
            .then(response => response.json())
            .then(coefficients => {{
                window.modelCoefficients = coefficients;
            }})
            .catch(() => {{
                window.modelCoefficients = {{ coef: [0, 0, 0, 0, 0, 0], intercept: 0 }};
                document.getElementById('debug-info').innerText += ' | Failed to load model_coefficients.json (using defaults)';
            }});

        // Prediction function
        function predictYield() {{
            const area = parseFloat(document.getElementById('area').value) || 0;
            const item = parseFloat(document.getElementById('item').value) || 0;
            const year = parseFloat(document.getElementById('year').value) || 2000;
            const rain = parseFloat(document.getElementById('rain').value) || 1000;
            const pest = parseFloat(document.getElementById('pest').value) || 500;
            const temp = parseFloat(document.getElementById('temp').value) || 25;
            const features = [area, item, year, rain, pest, temp];
            let pred = window.modelCoefficients.intercept || 0;
            for (let i = 0; i < features.length; i++) {{
                pred += (window.modelCoefficients.coef[i] || 0) * features[i];
            }}
            document.getElementById('real-time-pred').innerText = `Predicted Yield (hg/ha): ${{pred.toFixed(2)}}`;
        }}
    </script>
</body>
</html>
""".format(
    current_date=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    df_shape=df.shape,
    df_columns=', '.join(df.columns.tolist()),
    visualizations=''.join([
        f"""
        <div>
            <h3 class="text-lg font-medium">{title}</h3>
            <img src="./{file}" alt="{title}" class="w-full rounded-lg" onerror="this.style.display='none';this.nextElementSibling.style.display='block';">
            <div class="placeholder-img" style="display:none;">Image not found: {file}</div>
        </div>
        """ for file, title in [
            ("correlation_heatmap.png", "Correlation Heatmap"),
            ("histograms.png", "Histograms"),
            ("pairplot.png", "Pairplot"),
            ("geospatial_plot.png", "Geospatial Plot"),
            ("area_histograms.png", "Area Histograms"),
            ("best_areas.png", "Best Areas by Crop"),
            ("pesticides_scatter.png", "Pesticides vs Yield"),
            ("multiple_plots.png", "Multiple Plots"),
            ("Linear Regression_actual_vs_pred.png", "Linear Regression: Actual vs Predicted"),
            ("Random Forest_actual_vs_pred.png", "Random Forest: Actual vs Predicted"),
            ("Gradient Boost_actual_vs_pred.png", "Gradient Boost: Actual vs Predicted"),
            ("XGBoost_actual_vs_pred.png", "XGBoost: Actual vs Predicted"),
            ("KNN_actual_vs_pred.png", "KNN: Actual vs Predicted"),
            ("Decision Tree_actual_vs_pred.png", "Decision Tree: Actual vs Predicted"),
            ("Bagging Regressor_actual_vs_pred.png", "Bagging Regressor: Actual vs Predicted")
        ]
    ]),
    model_table=dff.to_html(classes='table-auto w-full border-collapse', index=False) if 'dff' in locals() else '<p>No model results available.</p>',
    detailed_table=df_results.to_html(classes='table-auto w-full border-collapse', index=False) if 'df_results' in locals() else '<p>No detailed results available.</p>',
    yield_stats_tables=''.join([f'<div><h3 class="text-lg font-medium">Group {i+1}</h3>{table}</div>' for i, table in enumerate(yield_stats)] if 'yield_stats' in locals() else ['<p>No yield stats available.</p>'])
)

file_path = os.path.join(BASE_DIR, "index.html")
with open(file_path, 'w') as f:
    f.write(html_content)
generated_files.append("index.html")
print(f"Saved {file_path}")

print("\nAll generated files:")
for file in generated_files:
    print(file)