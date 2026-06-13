"""
Car Price Prediction with Machine Learning
==========================================
Dataset: 301 car records with features like brand, mileage, year, etc.
Target: Selling_Price (in lakhs INR)
Models: Linear Regression, Random Forest, Gradient Boosting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─── Aesthetics ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#2e3347',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9',
    'grid.color':       '#2e3347',
    'grid.linestyle':   '--',
    'grid.alpha':       0.6,
    'font.family':      'DejaVu Sans',
})
PALETTE = ['#58a6ff', '#f78166', '#3fb950', '#d2a8ff', '#ffa657', '#79c0ff']
ACCENT  = '#58a6ff'

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD & EXPLORE
# ═══════════════════════════════════════════════════════════════════════════
df = pd.read_csv('/mnt/user-data/uploads/car_data.csv')
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print("\nFirst 5 rows:")
print(df.head())
print("\nStatistical Summary:")
print(df.describe())

# ═══════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
df = df.copy()

# Age of car instead of year
df['Car_Age'] = 2026 - df['Year']

# Price depreciation: how much value was lost relative to original price
df['Price_Depreciation'] = df['Present_Price'] - df['Selling_Price']
df['Depreciation_Rate']  = df['Price_Depreciation'] / df['Present_Price']

# Brand goodwill: average selling price per brand
brand_goodwill = df.groupby('Car_Name')['Selling_Price'].mean().rename('Brand_Goodwill')
df = df.merge(brand_goodwill, on='Car_Name', how='left')

# Mileage category
df['Mileage_Category'] = pd.cut(
    df['Driven_kms'],
    bins=[0, 20000, 50000, 100000, 600000],
    labels=['Low (<20k)', 'Medium (20-50k)', 'High (50-100k)', 'Very High (>100k)']
)

# Encode categoricals
le = LabelEncoder()
for col in ['Fuel_Type', 'Selling_type', 'Transmission']:
    df[col + '_enc'] = le.fit_transform(df[col])

mileage_map = {'Low (<20k)': 0, 'Medium (20-50k)': 1, 'High (50-100k)': 2, 'Very High (>100k)': 3}
df['Mileage_enc'] = df['Mileage_Category'].map(mileage_map)

print("\nEngineered Features Added: Car_Age, Price_Depreciation, Depreciation_Rate, Brand_Goodwill, Mileage_Category")

# ═══════════════════════════════════════════════════════════════════════════
# 3. PREPARE FEATURES
# ═══════════════════════════════════════════════════════════════════════════
FEATURES = [
    'Car_Age', 'Present_Price', 'Driven_kms',
    'Fuel_Type_enc', 'Selling_type_enc', 'Transmission_enc',
    'Owner', 'Brand_Goodwill', 'Depreciation_Rate', 'Mileage_enc'
]
TARGET = 'Selling_Price'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. TRAIN MODELS
# ═══════════════════════════════════════════════════════════════════════════
models = {
    'Linear Regression':    LinearRegression(),
    'Random Forest':        RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting':    GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42),
}

results = {}
predictions = {}

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

for name, model in models.items():
    Xtr = X_train_s if name == 'Linear Regression' else X_train
    Xte = X_test_s  if name == 'Linear Regression' else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(model, Xtr, y_train, cv=5, scoring='r2')

    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'CV_R2': cv_scores.mean()}
    predictions[name] = y_pred

    print(f"\n{name}")
    print(f"  MAE  : ₹{mae:.2f} L  |  RMSE: ₹{rmse:.2f} L  |  R²: {r2:.4f}  |  CV R²: {cv_scores.mean():.4f}")

# Best model
best_model_name = max(results, key=lambda k: results[k]['R2'])
best_pred = predictions[best_model_name]
print(f"\n✅ Best Model: {best_model_name}  (R² = {results[best_model_name]['R2']:.4f})")

# ═══════════════════════════════════════════════════════════════════════════
# 5. VISUALIZATIONS  (4 pages / figures)
# ═══════════════════════════════════════════════════════════════════════════

# ── Figure 1: EDA ──────────────────────────────────────────────────────────
fig1 = plt.figure(figsize=(18, 12), facecolor='#0f1117')
fig1.suptitle('Car Price Prediction — Exploratory Data Analysis', fontsize=18, fontweight='bold', color='white', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.42, wspace=0.38)

# 1a. Selling Price Distribution
ax = fig1.add_subplot(gs[0, 0])
ax.hist(df['Selling_Price'], bins=30, color=ACCENT, edgecolor='#0f1117', alpha=0.85)
ax.axvline(df['Selling_Price'].median(), color='#f78166', lw=2, linestyle='--', label=f"Median ₹{df['Selling_Price'].median():.1f}L")
ax.set_title('Selling Price Distribution', fontweight='bold')
ax.set_xlabel('Selling Price (₹ Lakhs)')
ax.set_ylabel('Count')
ax.legend()

# 1b. Price vs Car Age
ax = fig1.add_subplot(gs[0, 1])
scatter = ax.scatter(df['Car_Age'], df['Selling_Price'], c=df['Present_Price'],
                     cmap='plasma', alpha=0.7, s=40, edgecolors='none')
plt.colorbar(scatter, ax=ax, label='Present Price')
ax.set_title('Price vs Car Age', fontweight='bold')
ax.set_xlabel('Car Age (years)')
ax.set_ylabel('Selling Price (₹ Lakhs)')

# 1c. Fuel Type Breakdown
ax = fig1.add_subplot(gs[0, 2])
fuel_counts = df['Fuel_Type'].value_counts()
wedges, texts, autotexts = ax.pie(fuel_counts, labels=fuel_counts.index,
    autopct='%1.1f%%', colors=PALETTE[:len(fuel_counts)],
    textprops={'color': 'white'}, startangle=140)
for at in autotexts: at.set_fontsize(9)
ax.set_title('Fuel Type Breakdown', fontweight='bold')

# 1d. Driven KMs vs Selling Price
ax = fig1.add_subplot(gs[1, 0])
ax.scatter(df['Driven_kms'] / 1000, df['Selling_Price'], color='#3fb950', alpha=0.6, s=35)
ax.set_title('Mileage vs Selling Price', fontweight='bold')
ax.set_xlabel('Driven (000 km)')
ax.set_ylabel('Selling Price (₹ Lakhs)')

# 1e. Transmission & Type boxplot
ax = fig1.add_subplot(gs[1, 1])
groups = [df[df['Transmission'] == t]['Selling_Price'].values for t in df['Transmission'].unique()]
bp = ax.boxplot(groups, patch_artist=True, labels=df['Transmission'].unique())
for patch, color in zip(bp['boxes'], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.7)
for element in ['whiskers', 'caps', 'medians']:
    for line in bp[element]: line.set_color('#c9d1d9')
ax.set_title('Selling Price by Transmission', fontweight='bold')
ax.set_ylabel('Selling Price (₹ Lakhs)')

# 1f. Correlation Heatmap
ax = fig1.add_subplot(gs[1, 2])
num_cols = ['Selling_Price', 'Present_Price', 'Car_Age', 'Driven_kms', 'Owner',
            'Brand_Goodwill', 'Depreciation_Rate']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(num_cols))); ax.set_yticks(range(len(num_cols)))
ax.set_xticklabels([c.replace('_', '\n') for c in num_cols], fontsize=7)
ax.set_yticklabels([c.replace('_', '\n') for c in num_cols], fontsize=7)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=6.5, color='black')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title('Correlation Heatmap', fontweight='bold')

plt.savefig('/home/claude/fig1_eda.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
print("\nSaved: fig1_eda.png")

# ── Figure 2: Feature Importance & Brand Goodwill ─────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0f1117')
fig2.suptitle('Feature Importance & Brand Goodwill', fontsize=16, fontweight='bold', color='white')

rf_model = models['Random Forest']
importances = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=True)

axes[0].barh(importances.index, importances.values, color=PALETTE * 2, edgecolor='none')
axes[0].set_title('Random Forest — Feature Importance', fontweight='bold')
axes[0].set_xlabel('Importance Score')

top_brands = brand_goodwill.sort_values(ascending=False).head(15)
axes[1].barh(top_brands.index, top_brands.values, color=ACCENT, alpha=0.8, edgecolor='none')
axes[1].set_title('Top 15 Brands by Average Selling Price\n(Brand Goodwill)', fontweight='bold')
axes[1].set_xlabel('Avg Selling Price (₹ Lakhs)')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/home/claude/fig2_importance.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
print("Saved: fig2_importance.png")

# ── Figure 3: Model Comparison ─────────────────────────────────────────────
fig3, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor='#0f1117')
fig3.suptitle('Model Comparison — Predicted vs Actual', fontsize=16, fontweight='bold', color='white')

for idx, (name, y_pred) in enumerate(predictions.items()):
    row, col = divmod(idx, 3)
    ax = axes[row][col]
    ax.scatter(y_test, y_pred, alpha=0.65, s=45, color=PALETTE[idx], edgecolors='none')
    lo = min(y_test.min(), y_pred.min()) - 1
    hi = max(y_test.max(), y_pred.max()) + 1
    ax.plot([lo, hi], [lo, hi], 'w--', lw=1.5, label='Perfect Prediction')
    r2 = results[name]['R2']; mae = results[name]['MAE']
    ax.set_title(f'{name}\nR² = {r2:.4f}  |  MAE = ₹{mae:.2f}L', fontweight='bold')
    ax.set_xlabel('Actual Price (₹ Lakhs)')
    ax.set_ylabel('Predicted Price (₹ Lakhs)')
    ax.legend(fontsize=8)

# Metrics bar chart in [1][1]
ax = axes[1][1]
metric_names = ['MAE (₹L)', 'RMSE (₹L)', 'R²']
x = np.arange(len(metric_names))
w = 0.25
for i, (name, res) in enumerate(results.items()):
    vals = [res['MAE'], res['RMSE'], res['R2']]
    bars = ax.bar(x + i*w, vals, w, label=name, color=PALETTE[i], alpha=0.85, edgecolor='none')
ax.set_xticks(x + w)
ax.set_xticklabels(metric_names)
ax.set_title('Model Metrics Comparison', fontweight='bold')
ax.legend(fontsize=8)

# Residuals for best model in [1][2]
ax = axes[1][2]
residuals = y_test.values - best_pred
ax.hist(residuals, bins=25, color='#d2a8ff', edgecolor='#0f1117', alpha=0.85)
ax.axvline(0, color='white', lw=1.5, linestyle='--')
ax.set_title(f'{best_model_name}\nResidual Distribution', fontweight='bold')
ax.set_xlabel('Residual (₹ Lakhs)')
ax.set_ylabel('Count')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/home/claude/fig3_models.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
print("Saved: fig3_models.png")

# ── Figure 4: Predictions on Sample Cars ──────────────────────────────────
gb_model = models['Gradient Boosting']

sample = df.sample(12, random_state=7).copy()
X_sample = sample[FEATURES]
sample['Predicted'] = gb_model.predict(X_sample)

fig4, ax = plt.subplots(figsize=(14, 6), facecolor='#0f1117')
idx_range = np.arange(len(sample))
w = 0.35
bars1 = ax.bar(idx_range - w/2, sample['Selling_Price'], w, label='Actual', color=ACCENT, alpha=0.85)
bars2 = ax.bar(idx_range + w/2, sample['Predicted'], w, label='Predicted', color='#f78166', alpha=0.85)
ax.set_xticks(idx_range)
ax.set_xticklabels(
    [f"{r['Car_Name'].title()}\n({r['Year']})" for _, r in sample.iterrows()],
    rotation=30, ha='right', fontsize=8
)
ax.set_title('Gradient Boosting — Actual vs Predicted on 12 Sample Cars', fontsize=14, fontweight='bold')
ax.set_ylabel('Price (₹ Lakhs)')
ax.legend()

plt.tight_layout()
plt.savefig('/home/claude/fig4_sample_predictions.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
print("Saved: fig4_sample_predictions.png")

# ═══════════════════════════════════════════════════════════════════════════
# 6. SAMPLE INFERENCE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SAMPLE INFERENCE — Gradient Boosting Model")
print("=" * 60)

test_cars = [
    {'Car_Name': 'honda city', 'Year': 2020, 'Present_Price': 12.5, 'Driven_kms': 25000,
     'Fuel_Type': 'Petrol', 'Selling_type': 'Dealer', 'Transmission': 'Manual', 'Owner': 0},
    {'Car_Name': 'swift',      'Year': 2018, 'Present_Price': 6.0,  'Driven_kms': 55000,
     'Fuel_Type': 'Petrol', 'Selling_type': 'Individual', 'Transmission': 'Manual', 'Owner': 1},
    {'Car_Name': 'fortuner',   'Year': 2019, 'Present_Price': 32.0, 'Driven_kms': 40000,
     'Fuel_Type': 'Diesel', 'Selling_type': 'Dealer', 'Transmission': 'Automatic', 'Owner': 0},
]

fuel_enc   = {'Petrol': 1, 'Diesel': 0, 'CNG': 2}
sell_enc   = {'Dealer': 0, 'Individual': 1}
trans_enc  = {'Manual': 1, 'Automatic': 0}

for car in test_cars:
    car_age  = 2026 - car['Year']
    bg       = brand_goodwill.get(car['Car_Name'], brand_goodwill.mean())
    dep_rate = 0.30  # assumed 30% depreciation
    row = pd.DataFrame([{
        'Car_Age':          car_age,
        'Present_Price':    car['Present_Price'],
        'Driven_kms':       car['Driven_kms'],
        'Fuel_Type_enc':    fuel_enc[car['Fuel_Type']],
        'Selling_type_enc': sell_enc[car['Selling_type']],
        'Transmission_enc': trans_enc[car['Transmission']],
        'Owner':            car['Owner'],
        'Brand_Goodwill':   bg,
        'Depreciation_Rate': dep_rate,
        'Mileage_enc':      1 if car['Driven_kms'] < 50000 else 2,
    }])
    price = gb_model.predict(row)[0]
    print(f"  {car['Car_Name'].title()} ({car['Year']}) | {car['Driven_kms']:,} km | {car['Fuel_Type']} "
          f"→  Predicted: ₹{price:.2f} Lakhs")

print("\nAll done! 4 figures saved.\n")
