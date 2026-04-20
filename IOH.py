import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH   = 'IOH_data_coord_app.csv'   # change if needed
OUTPUT_DIR  = '.'                         # where to save the 4 PNGs
RANDOM_SEED = 42
TEST_SIZE   = 0.2

# ── PALETTE ───────────────────────────────────────────────────────────────────
DARK  = '#0f172a'
CARD  = '#1e293b'
ACC1  = '#38bdf8'
ACC2  = '#f472b6'
ACC3  = '#a78bfa'
ACC4  = '#34d399'
ACC5  = '#fb923c'
TEXT  = '#f1f5f9'
MUTED = '#94a3b8'
PALETTE = [ACC1, ACC2, ACC3, ACC4, ACC5]

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor':  CARD,
    'axes.edgecolor':   MUTED,'axes.labelcolor': TEXT,
    'text.color':       TEXT, 'xtick.color':     MUTED,
    'ytick.color':      MUTED,'grid.color':      '#334155',
    'grid.linestyle':   '--', 'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
})

# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD & EXPLORE
# ═════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(df.head())
print("\nNull counts:\n", df.isnull().sum())
print("\nisPresent distribution:\n", df['isPresent'].value_counts())

# ═════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
df['log_groupSize'] = np.log1p(df['groupSize'])
df['is_local']      = (df['state'] == 'Tamil Nadu').astype(int)
df['has_college']   = df['collegeName'].notna().astype(int)
df['has_company']   = df['companyName'].notna().astype(int)

le_cat   = LabelEncoder()
le_reg   = LabelEncoder()
le_st    = LabelEncoder()
df['cat_enc']   = le_cat.fit_transform(df['category'])
df['reg_enc']   = le_reg.fit_transform(df['registerType'])
df['state_enc'] = le_st.fit_transform(df['state'])

FEATURES = ['cat_enc','reg_enc','state_enc','groupSize','log_groupSize',
            'is_local','has_college','has_company']
TARGET = 'isPresent'

# ═════════════════════════════════════════════════════════════════════════════
# 3. TRAIN / TEST SPLIT + MODEL
# ═════════════════════════════════════════════════════════════════════════════
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

model = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=10,
    class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
model.fit(X_train, y_train)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"\nAccuracy : {acc:.3f}")
print(f"ROC-AUC  : {auc:.3f}")
print(classification_report(y_test, y_pred))

# ═════════════════════════════════════════════════════════════════════════════
# 4. FIGURE 1 — OVERVIEW DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.patch.set_facecolor(DARK)
fig.suptitle('📊  IOH Registration & Attendance — Overview Dashboard',
             fontsize=18, fontweight='bold', color=TEXT, y=0.98)

# (a) Donut — show-up vs no-show
ax = axes[0, 0]
sizes = df['isPresent'].value_counts().values
wedges, texts, autotexts = ax.pie(
    sizes, colors=[ACC4, ACC2], autopct='%1.1f%%', startangle=90,
    pctdistance=0.75, wedgeprops=dict(width=0.55, edgecolor=DARK, linewidth=2))
for at in autotexts:
    at.set_color(TEXT); at.set_fontweight('bold'); at.set_fontsize(13)
ax.set_facecolor(DARK)
ax.set_title('Did Registrants Actually Show Up?', color=TEXT, fontsize=13, pad=12)
ax.legend([f'✅ Attended ({sizes[0]})', f'❌ No-Show ({sizes[1]})'],
          loc='lower center', framealpha=0, labelcolor=TEXT, fontsize=10)

# (b) Bar + line — category breakdown
ax = axes[0, 1]
cat_counts = df['category'].value_counts()
cat_att    = df.groupby('category')['isPresent'].mean() * 100
x = np.arange(len(cat_counts))
ax.bar(x, cat_counts.values, color=PALETTE[:len(cat_counts)],
       edgecolor=DARK, linewidth=0.8, alpha=0.85, zorder=3)
ax2 = ax.twinx()
ax2.plot(x, cat_att[cat_counts.index].values, 'o--',
         color=ACC5, linewidth=2, markersize=8, zorder=4)
ax2.set_ylabel('Attendance Rate (%)', color=ACC5, fontsize=10)
ax2.tick_params(colors=ACC5)
ax2.set_facecolor(CARD)
ax.set_xticks(x)
ax.set_xticklabels([c.replace('_', '\n') for c in cat_counts.index], fontsize=9)
ax.set_title('Registrations by Category\n(+ Attendance Rate)', color=TEXT, fontsize=13)
ax.set_ylabel('# Registrations', color=TEXT)
ax.yaxis.grid(True, alpha=0.4); ax.set_zorder(1)

# (c) Horizontal bar — state attendance rate
ax = axes[1, 0]
state_att = df.groupby('state')['isPresent'].mean().sort_values(ascending=False)
colors_st = [ACC1 if v >= state_att.mean() else ACC2 for v in state_att.values]
bars_st = ax.barh(state_att.index, state_att.values * 100,
                  color=colors_st, edgecolor=DARK, linewidth=0.6)
ax.axvline(state_att.mean() * 100, color=ACC5, linestyle='--', linewidth=1.5,
           label=f'Avg {state_att.mean()*100:.1f}%')
ax.set_xlabel('Attendance Rate (%)', color=TEXT)
ax.set_title('Attendance Rate by State', color=TEXT, fontsize=13)
ax.legend(framealpha=0, labelcolor=TEXT, fontsize=9)
for bar, val in zip(bars_st, state_att.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val*100:.1f}%', va='center', color=TEXT, fontsize=9)

# (d) Histogram — group size vs attendance
ax = axes[1, 1]
bins = np.arange(0.5, 21.5, 1)
ax.hist(df[df['isPresent'] == 0]['groupSize'].clip(upper=20),
        bins=bins, color=ACC2, alpha=0.75, label='No-Show', edgecolor=DARK)
ax.hist(df[df['isPresent'] == 1]['groupSize'].clip(upper=20),
        bins=bins, color=ACC4, alpha=0.75, label='Attended', edgecolor=DARK)
ax.set_xlabel('Group Size (capped at 20)')
ax.set_ylabel('# Registrations')
ax.set_title('Group Size vs Attendance', color=TEXT, fontsize=13)
ax.legend(framealpha=0, labelcolor=TEXT)
ax.yaxis.grid(True, alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{OUTPUT_DIR}/fig1_overview.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print("Saved fig1_overview.png")

# ═════════════════════════════════════════════════════════════════════════════
# 5. FIGURE 2 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.patch.set_facecolor(DARK)
fig.suptitle(
    f'🤖  Prediction Model Performance  |  Accuracy {acc*100:.1f}%  |  ROC-AUC {auc:.3f}',
    fontsize=17, fontweight='bold', color=TEXT, y=0.98)

# (a) Confusion matrix
ax = axes[0, 0]
cm = confusion_matrix(y_test, y_pred)
ax.imshow(cm, cmap='Blues', aspect='auto')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i,j]}', ha='center', va='center',
                color=TEXT, fontsize=22, fontweight='bold')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Pred: No-Show', 'Pred: Attended'], fontsize=11)
ax.set_yticklabels(['Actual: No-Show', 'Actual: Attended'], fontsize=11)
ax.set_title('Confusion Matrix\n(How many did the model get right?)', color=TEXT, fontsize=13)

# (b) ROC curve
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax.plot(fpr, tpr, color=ACC1, linewidth=2.5, label=f'Model (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], '--', color=MUTED, linewidth=1.5, label='Random Guess (AUC=0.5)')
ax.fill_between(fpr, tpr, alpha=0.15, color=ACC1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve\n(Closer to top-left = better)', color=TEXT, fontsize=13)
ax.legend(framealpha=0, labelcolor=TEXT)
ax.yaxis.grid(True)

# (c) Feature importance
ax = axes[1, 0]
feat_labels = {
    'cat_enc': 'Category (Student/Family…)',
    'reg_enc': 'Registration Type (Indiv/Group)',
    'state_enc': 'State',
    'groupSize': 'Group Size',
    'log_groupSize': 'Group Size (log)',
    'is_local': 'Is from Tamil Nadu?',
    'has_college': 'Has a College Name?',
    'has_company': 'Has a Company Name?',
}
feat_imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
colors_fi = [ACC1 if v >= feat_imp.median() else MUTED for v in feat_imp.values]
ax.barh([feat_labels[f] for f in feat_imp.index], feat_imp.values,
        color=colors_fi, edgecolor=DARK)
ax.set_xlabel('Importance Score')
ax.set_title('What Features Decide Attendance?\n(Longer bar = more important)', color=TEXT, fontsize=13)
ax.xaxis.grid(True)

# (d) Predicted probability distributions
ax = axes[1, 1]
bins = np.linspace(0, 1, 30)
ax.hist(y_proba[y_test == 0], bins=bins, color=ACC2, alpha=0.75, label='Actual No-Show', edgecolor=DARK)
ax.hist(y_proba[y_test == 1], bins=bins, color=ACC4, alpha=0.75, label='Actual Attended', edgecolor=DARK)
ax.axvline(0.5, color=ACC5, linewidth=2, linestyle='--', label='Decision Threshold (0.5)')
ax.set_xlabel("Model's Predicted Probability of Attending")
ax.set_ylabel('# People')
ax.set_title('Predicted Probabilities\n(Are the groups separable?)', color=TEXT, fontsize=13)
ax.legend(framealpha=0, labelcolor=TEXT)
ax.yaxis.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{OUTPUT_DIR}/fig2_model.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print("Saved fig2_model.png")

# ═════════════════════════════════════════════════════════════════════════════
# 6. FIGURE 3 — MARKETING INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.patch.set_facecolor(DARK)
fig.suptitle('🎯  Publicity Team Strategy — Where to Focus Your Efforts?',
             fontsize=17, fontweight='bold', color=TEXT, y=0.98)

# (a) Heatmap — category × state
ax = axes[0, 0]
pivot = df.groupby(['category', 'state'])['isPresent'].mean().unstack(fill_value=0)
cat_order = ['INSTITUTE', 'FAMILY', 'INDUSTRY_PROFESSIONAL', 'COLLEGE_STUDENT', 'OTHERS']
pivot = pivot.reindex([c for c in cat_order if c in pivot.index])
im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=35, ha='right', fontsize=9)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([c.replace('_', ' ').title() for c in pivot.index], fontsize=9)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                color='black' if val > 0.55 else TEXT, fontsize=9, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
ax.set_title('Attendance Rate: Category × State\n(Green = great, Red = needs work)', color=TEXT, fontsize=12)

# (b) Bubble chart — registrations vs attendance per state
ax = axes[0, 1]
state_stats = df.groupby('state').agg(
    total=('isPresent', 'count'),
    attended=('isPresent', 'sum'),
    rate=('isPresent', 'mean')
).reset_index()
scatter = ax.scatter(state_stats['total'], state_stats['attended'],
                     s=state_stats['rate'] * 1500,
                     c=state_stats['rate'], cmap='RdYlGn',
                     alpha=0.85, edgecolors=DARK, linewidth=1.2, vmin=0.2, vmax=0.5)
for _, row in state_stats.iterrows():
    ax.annotate(row['state'], (row['total'], row['attended']),
                textcoords='offset points', xytext=(6, 4), color=TEXT, fontsize=9)
ax.set_xlabel('Total Registrations from State')
ax.set_ylabel('Total Attended')
ax.set_title('Registrations → Attendance per State\n(Bubble size = conversion rate)', color=TEXT, fontsize=12)
plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04, label='Attendance Rate')
ax.yaxis.grid(True)

# (c) Group size buckets
ax = axes[1, 0]
bins_gs   = [0, 1, 2, 4, 8, 200]
labels_gs = ['Solo (1)', 'Pair (2)', 'Small (3-4)', 'Medium (5-8)', 'Large (9+)']
df['gs_bucket'] = pd.cut(df['groupSize'], bins=bins_gs, labels=labels_gs,
                          right=True, include_lowest=True)
gs_stats = df.groupby('gs_bucket', observed=True).agg(
    registrations=('isPresent', 'count'),
    attendance_rate=('isPresent', 'mean'),
    total_attended=('isPresent', 'sum')
)
x = np.arange(len(gs_stats))
ax.bar(x - 0.2, gs_stats['registrations'], 0.35, color=ACC1, alpha=0.8, label='Registrations')
ax.bar(x + 0.2, gs_stats['total_attended'], 0.35, color=ACC4, alpha=0.8, label='Actual Attendees')
ax2 = ax.twinx()
ax2.plot(x, gs_stats['attendance_rate'] * 100, 'D--', color=ACC5, linewidth=2, markersize=9)
ax2.set_ylabel('Attendance Rate (%)', color=ACC5)
ax2.tick_params(colors=ACC5)
ax2.set_facecolor(CARD)
ax.set_xticks(x)
ax.set_xticklabels(gs_stats.index.tolist(), fontsize=10)
ax.set_title('Group Size Buckets — Who Actually Shows Up?', color=TEXT, fontsize=12)
ax.set_ylabel('Count')
ax.legend(loc='upper right', framealpha=0, labelcolor=TEXT)
ax.yaxis.grid(True, alpha=0.4)

# (d) Priority matrix
ax = axes[1, 1]
segs = pd.DataFrame({
    'Segment':  ['Institutes', 'Families', 'Industry\nProfessionals', 'College\nStudents', 'Others'],
    'Volume':   [62, 1070, 132, 1228, 508],
    'AttRate':  [0.726, 0.426, 0.386, 0.382, 0.303],
    'color':    [ACC4, ACC1, ACC3, ACC2, MUTED]
})
for _, row in segs.iterrows():
    ax.scatter(row['Volume'], row['AttRate'] * 100, s=row['AttRate'] * 3000,
               c=row['color'], alpha=0.9, edgecolors=DARK, linewidth=1.5, zorder=4)
    ax.annotate(row['Segment'], (row['Volume'], row['AttRate'] * 100),
                textcoords='offset points', xytext=(8, 4), color=TEXT, fontsize=10, fontweight='bold')
ax.axhline(segs['AttRate'].mean() * 100, color=MUTED, linestyle='--', alpha=0.6)
ax.axvline(400, color=MUTED, linestyle=':', alpha=0.6)
ax.set_xlabel('Number of Registrations (Volume)')
ax.set_ylabel('Attendance Rate (%)')
ax.set_title('Priority Matrix\n(High volume + low rate = target these first)', color=TEXT, fontsize=12)
ax.text(500, 30, '⚠️ HIGH VOL\nLOW CONV\n→ TARGET', color=ACC5, fontsize=9, fontweight='bold')
ax.yaxis.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{OUTPUT_DIR}/fig3_insights.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print("Saved fig3_insights.png")

# ═════════════════════════════════════════════════════════════════════════════
# 7. FIGURE 4 — SUMMARY & RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor(DARK)
fig.suptitle('📌  Key Metrics & Publicity Recommendations',
             fontsize=17, fontweight='bold', color=TEXT, y=0.98)

gs_layout = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
ax_funnel = fig.add_subplot(gs_layout[0, 0])
ax_bar    = fig.add_subplot(gs_layout[1, 0])
ax_cards  = fig.add_subplot(gs_layout[:, 1])

# Funnel
funnel_vals   = [3000, 1968, 1132, 1175]
funnel_stages = ['Total Registrations\n3000', 'From TN\n1968',
                 'Valid Groups\n1132', 'Attended\n1175']
funnel_colors = [ACC1, ACC3, ACC5, ACC4]
for i, (stage, val, col) in enumerate(zip(funnel_stages, funnel_vals, funnel_colors)):
    ax_funnel.barh(i, val / funnel_vals[0], color=col, alpha=0.85, height=0.6, edgecolor=DARK)
    ax_funnel.text(val / funnel_vals[0] + 0.01, i, stage, va='center', color=TEXT, fontsize=9)
ax_funnel.set_xlim(0, 1.45); ax_funnel.set_yticks([]); ax_funnel.set_xticks([])
ax_funnel.set_title('Registration → Attendance Funnel', color=TEXT, fontsize=12)
for spine in ax_funnel.spines.values():
    spine.set_visible(False)

# Category bars
cat_att_sorted = df.groupby('category')['isPresent'].mean().sort_values(ascending=True)
colors_bar = [ACC4 if v >= 0.4 else (ACC5 if v >= 0.35 else ACC2) for v in cat_att_sorted.values]
ax_bar.barh([c.replace('_', ' ').title() for c in cat_att_sorted.index],
             cat_att_sorted.values * 100, color=colors_bar, edgecolor=DARK)
ax_bar.axvline(39.2, color=MUTED, linestyle='--', linewidth=1.5, label='Overall avg 39.2%')
ax_bar.set_xlabel('Attendance Rate (%)')
ax_bar.set_title('Category Attendance Rates', color=TEXT, fontsize=12)
ax_bar.legend(framealpha=0, labelcolor=TEXT, fontsize=9)
ax_bar.xaxis.grid(True)
for i, v in enumerate(cat_att_sorted.values):
    ax_bar.text(v * 100 + 0.5, i, f'{v*100:.1f}%', va='center', color=TEXT, fontsize=10)

# Summary cards
ax_cards.set_facecolor(DARK); ax_cards.set_xlim(0, 1); ax_cards.set_ylim(0, 1)
ax_cards.axis('off')
cards = [
    ('🎯  Overall Attendance Rate',    '39.2%',            'Only 4 in 10 registrants actually show up', ACC2),
    ('🏆  Best Segment to Target',     'INSTITUTES',       'They convert at 72.6% — best ROI for outreach', ACC4),
    ('📍  Best State',                  'Puducherry',       '42.6% attendance → closest to IIT Madras campus', ACC1),
    ('👥  Group Bookings Work Better', '44.3%',            'Group registrants attend more than individuals (36.1%)', ACC3),
    ('⚠️  Biggest Opportunity',         'College Students', '1228 registrations but only 38.2% show — fix this!', ACC5),
    ('🚫  Worst Segment',              'Others (30.3%)',   'Vague category converts least — qualify them', MUTED),
]
y_pos = 0.93
for title, metric, desc, color in cards:
    rect = mpatches.FancyBboxPatch((0.02, y_pos - 0.13), 0.96, 0.12,
        boxstyle="round,pad=0.01", linewidth=1.5,
        edgecolor=color, facecolor=CARD, zorder=2)
    ax_cards.add_patch(rect)
    ax_cards.text(0.08, y_pos - 0.03, title,   color=MUTED, fontsize=9,  va='top')
    ax_cards.text(0.08, y_pos - 0.07, metric,  color=color, fontsize=13, fontweight='bold', va='top')
    ax_cards.text(0.08, y_pos - 0.11, desc,    color=TEXT,  fontsize=8.5, va='top', style='italic')
    y_pos -= 0.155
ax_cards.set_title('Key Findings at a Glance', color=TEXT, fontsize=13, pad=10)

plt.savefig(f'{OUTPUT_DIR}/fig4_summary.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print("Saved fig4_summary.png")
print("\nAll done! 4 plots saved.")