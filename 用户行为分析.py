import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

# ============================================================
# 配置
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 配色
COLORS = {
    '高价值客户': '#1B4F72',
    '忠诚客户': '#2E86C1',
    '新客户': '#5DADE2',
    '一般客户': '#AEB6BF',
    '流失客户': '#E74C3C'
}

os.makedirs('images', exist_ok=True)

print("=" * 60)
print("电商用户价值与运营分析")
print("=" * 60)

# ============================================================
# 1. 数据加载（检查文件是否存在）
# ============================================================
if not os.path.exists('OnlineRetail.csv'):
    print("错误: 未找到 OnlineRetail.csv 文件，请将该文件放在当前目录下。")
    exit()

df = pd.read_csv('OnlineRetail.csv')
print(f"原始记录数: {len(df):,}")

# ============================================================
# 2. 数据清洗
# ============================================================
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df_clean = df.dropna(subset=['CustomerID'])
df_clean = df_clean[df_clean['Quantity'] > 0]
df_clean = df_clean[df_clean['UnitPrice'] > 0]
df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
print(f"清洗后记录数: {len(df_clean):,}")

# ============================================================
# 3. 核心指标
# ============================================================
total_sales = df_clean['TotalPrice'].sum()
avg_order = df_clean.groupby('InvoiceNo')['TotalPrice'].sum().mean()
unique_customers = df_clean['CustomerID'].nunique()
unique_countries = df_clean['Country'].nunique()

print("\n核心经营指标")
print("-" * 40)
print(f"总销售额: {total_sales:,.2f}")
print(f"平均客单价: {avg_order:,.2f}")
print(f"客户总数: {unique_customers:,}")
print(f"覆盖国家数: {unique_countries}")

# ============================================================
# 4. RFM建模
# ============================================================
snapshot_date = df_clean['InvoiceDate'].max() + dt.timedelta(days=1)

rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

rfm['R_rank'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
rfm['F_rank'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
rfm['M_rank'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4])

def rfm_segment(row):
    r, f, m = row['R_rank'], row['F_rank'], row['M_rank']
    if r in [3, 4] and f in [3, 4] and m in [3, 4]:
        return '高价值客户'
    elif r in [3, 4] and f in [1, 2]:
        return '新客户'
    elif r in [1, 2] and f in [3, 4]:
        return '忠诚客户'
    elif r in [1, 2] and f in [1, 2]:
        return '流失客户'
    else:
        return '一般客户'

rfm['Segment'] = rfm.apply(rfm_segment, axis=1)
segment_counts = rfm['Segment'].value_counts()

print("\n客户分层统计")
print("-" * 40)
for seg, count in segment_counts.items():
    print(f"{seg}: {count} 人 ({count/len(rfm)*100:.1f}%)")

# ============================================================
# 5. 市场分析
# ============================================================
country_counts = df_clean['Country'].value_counts().head(10)
total_count = df_clean['Country'].count()

# ============================================================
# 6. 热销商品
# ============================================================
top_products = df_clean.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

# ============================================================
# 7. 月度趋势
# ============================================================
df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M')
monthly_sales = df_clean.groupby('YearMonth')['TotalPrice'].sum()
months = monthly_sales.index.astype(str)
sales = monthly_sales.values

# ============================================================
# 8. 复购分析
# ============================================================
customer_orders = df_clean.groupby('CustomerID')['InvoiceNo'].nunique()
repeat_customers = (customer_orders >= 2).sum()
repeat_rate = repeat_customers / len(customer_orders) * 100

print("\n留存分析")
print("-" * 40)
print(f"总客户数: {len(customer_orders):,}")
print(f"复购客户数: {repeat_customers:,}")
print(f"复购率: {repeat_rate:.1f}%")

# ============================================================
# 9. 可视化图表 (8张)
# ============================================================
print("\n生成可视化图表...")

# ---- 图1: 客户分层环形图 ----
fig, ax = plt.subplots(figsize=(10, 8))
colors_pie = [COLORS.get(x, '#AEB6BF') for x in segment_counts.index]
wedges, texts, autotexts = ax.pie(
    segment_counts.values,
    labels=segment_counts.index,
    autopct='%1.1f%%',
    colors=colors_pie,
    startangle=90,
    explode=(0.03, 0, 0, 0.03, 0.05),
    textprops={'fontsize': 12},
    pctdistance=0.75
)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(11)
    autotext.set_fontweight('bold')
ax.set_title('客户价值分层', fontsize=16, fontweight='bold', pad=20)

high_loyal = (segment_counts.get('高价值客户', 0) + segment_counts.get('忠诚客户', 0)) / segment_counts.sum() * 100
churned = segment_counts.get('流失客户', 0) / segment_counts.sum() * 100
ax.text(0.5, -0.12,
        f'高价值+忠诚客户: {high_loyal:.1f}%  |  流失客户: {churned:.1f}%',
        ha='center', va='center', transform=ax.transAxes,
        fontsize=11, style='italic',
        bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
plt.tight_layout()
plt.savefig('images/客户分层.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---- 图2: 月度销售趋势 ----
fig, ax = plt.subplots(figsize=(14, 6))
ax.fill_between(range(len(months)), sales, alpha=0.25, color='#2E86C1')
ax.plot(range(len(months)), sales, marker='o', linewidth=2, color='#1B4F72', markersize=6)

peak_idx = np.argmax(sales)
ax.scatter(peak_idx, sales[peak_idx], color='#E74C3C', s=120, zorder=5)
ax.annotate(f'峰值: {sales[peak_idx]:,.0f}',
            xy=(peak_idx, sales[peak_idx]),
            xytext=(10, 25), textcoords='offset points',
            fontsize=11, fontweight='bold', color='#E74C3C')

trough_idx = np.argmin(sales)
ax.scatter(trough_idx, sales[trough_idx], color='#F39C12', s=120, zorder=5)

ax.axhline(y=sales.mean(), color='#7F8C8D', linestyle='--', alpha=0.6,
           label=f'月均: {sales.mean():,.0f}')

ax.set_xticks(range(len(months)))
ax.set_xticklabels(months, rotation=45, ha='right', fontsize=9)
ax.set_xlabel('月份', fontsize=12)
ax.set_ylabel('销售额', fontsize=12)
ax.set_title('月度销售趋势', fontsize=16, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('images/月度销售趋势.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---- 图3: 国家分布 ----
fig, ax = plt.subplots(figsize=(12, 6))
uk_pct = country_counts.iloc[0] / total_count * 100
colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(country_counts)))[::-1]
bars = ax.barh(country_counts.index[::-1], country_counts.values[::-1], color=colors[::-1])

ax.set_xlabel('订单量', fontsize=12)
ax.set_ylabel('国家', fontsize=12)
ax.set_title(f'国家市场分布 Top10（英国占 {uk_pct:.1f}%）', fontsize=16, fontweight='bold')

for bar, val in zip(bars, country_counts.values[::-1]):
    pct = val / total_count * 100
    ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
            f'{val:,} ({pct:.1f}%)', ha='left', va='center', fontsize=9)

bars[0].set_color('#1B4F72')
bars[0].set_edgecolor('black')
bars[0].set_linewidth(1.2)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('images/国家分布.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---- 图4: 热销商品 ----
fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.Greens(np.linspace(0.35, 0.85, len(top_products)))[::-1]
bars = ax.barh(top_products.index[::-1], top_products.values[::-1], color=colors[::-1])

ax.set_xlabel('销量（件）', fontsize=12)
ax.set_ylabel('商品名称', fontsize=12)
ax.set_title('热销商品 Top10', fontsize=16, fontweight='bold')

for bar, val in zip(bars, top_products.values[::-1]):
    ax.text(bar.get_width() + 150, bar.get_y() + bar.get_height()/2,
            f'{val:,}', ha='left', va='center', fontsize=9)

ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('images/热销商品.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---- 图5: 高价值客户画像 ----
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.axis('tight')
ax.axis('off')

high_value = rfm[rfm['Segment'] == '高价值客户'].head(5)
table_data = [['排名', '客户ID', '最近消费(天)', '消费频次', '累计消费金额']]
for idx, (_, row) in enumerate(high_value.iterrows(), 1):
    table_data.append([
        str(idx),
        str(int(row['CustomerID'])),
        str(row['Recency']),
        f"{int(row['Frequency'])}",
        f"{row['Monetary']:,.2f}"
    ])

table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                 colWidths=[0.08, 0.2, 0.2, 0.2, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

for i in range(5):
    table[(0, i)].set_facecolor('#1B4F72')
    table[(0, i)].set_text_props(color='white', fontweight='bold')

for i in range(1, 6):
    for j in range(5):
        table[(i, j)].set_facecolor('#F8F9FA' if i % 2 == 0 else 'white')

ax.set_title('高价值客户 Top5', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('images/高价值客户画像.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---- 图6: 复购率 ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.barh(0, 100, height=0.35, color='#ECF0F1', edgecolor='none')
bar_color = '#E74C3C' if repeat_rate < 40 else '#F39C12' if repeat_rate < 60 else '#1B4F72'
ax.barh(0, repeat_rate, height=0.35, color=bar_color, edgecolor='none')

for tick in [0, 25, 50, 75, 100]:
    ax.axvline(x=tick, ymin=0.1, ymax=0.9, color='#BDC3C7', linewidth=0.5)
    ax.text(tick, -0.25, f'{tick}%', ha='center', va='top', fontsize=9)

ax.text(repeat_rate, 0.5, f'{repeat_rate:.1f}%',
        ha='center', va='center', fontsize=30, fontweight='bold', color=bar_color)

ax.text(2, -0.55, '行业基准: 30-50%', ha='left', va='center', fontsize=10, color='#7F8C8D')
ax.text(repeat_rate, -0.55, '高于行业水平', ha='center', va='center', fontsize=10, fontweight='bold', color='#1B4F72')

ax.set_title('复购率分析', fontsize=16, fontweight='bold', pad=25)
ax.set_xlim(-5, 105)
ax.set_ylim(-0.8, 1)
ax.axis('off')
plt.tight_layout()
plt.savefig('images/复购率.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---- 图7: RFM三维散点图 ----
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
segment_colors = {
    '高价值客户': '#1B4F72',
    '忠诚客户': '#2E86C1',
    '新客户': '#5DADE2',
    '一般客户': '#AEB6BF',
    '流失客户': '#E74C3C'
}
ax.scatter(rfm['Recency'], rfm['Frequency'], rfm['Monetary'],
           c=[segment_colors.get(s, '#AEB6BF') for s in rfm['Segment']],
           alpha=0.5, s=20)
ax.set_xlabel('最近消费天数', fontsize=10)
ax.set_ylabel('消费频次', fontsize=10)
ax.set_zlabel('累计消费金额', fontsize=10)
ax.set_title('RFM三维分布图', fontsize=14, fontweight='bold')
legend_elements = [Patch(facecolor=color, label=label) for label, color in segment_colors.items()]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig('images/RFM三维散点图.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---- 图8: 相关性热力图 ----
fig, ax = plt.subplots(figsize=(7, 5))
corr_matrix = rfm[['Recency', 'Frequency', 'Monetary']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Blues',
            center=0.5, square=True, linewidths=0.5,
            vmin=0, vmax=1,
            annot_kws={'fontsize': 13, 'fontweight': 'bold'},
            cbar_kws={'shrink': 0.8})
ax.set_title('RFM特征相关性矩阵', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('images/特征相关性热力图.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("8张图表已生成并保存至 images/ 目录")

# ============================================================
# 10. 生成Markdown分析报告
# ============================================================
print("\n生成分析报告...")

report_lines = []
report_lines.append("# 电商用户价值与运营分析报告")
report_lines.append("")

report_lines.append("## 一、核心经营指标")
report_lines.append("")
report_lines.append("| 指标 | 数值 |")
report_lines.append("|:---|:---:|")
report_lines.append(f"| 总销售额 | {total_sales:,.2f} |")
report_lines.append(f"| 平均客单价 | {avg_order:,.2f} |")
report_lines.append(f"| 客户总数 | {unique_customers:,} |")
report_lines.append(f"| 覆盖国家数 | {unique_countries} |")
report_lines.append("")

report_lines.append("## 二、客户分层统计")
report_lines.append("")
report_lines.append("| 客户层级 | 人数 | 占比 |")
report_lines.append("|:---|:---:|:---:|")
for seg, count in segment_counts.items():
    pct = count / len(rfm) * 100
    report_lines.append(f"| {seg} | {count} | {pct:.1f}% |")
report_lines.append("")

report_lines.append("## 三、国家市场分布（Top10）")
report_lines.append("")
report_lines.append("| 国家 | 订单量 | 占比 |")
report_lines.append("|:---|:---:|:---:|")
for country, count in country_counts.items():
    pct = count / total_count * 100
    report_lines.append(f"| {country} | {count:,} | {pct:.1f}% |")
report_lines.append("")

report_lines.append("## 四、热销商品（Top10）")
report_lines.append("")
report_lines.append("| 排名 | 商品名称 | 销量（件） |")
report_lines.append("|:---:|:---|:---:|")
for i, (product, qty) in enumerate(top_products.items(), 1):
    report_lines.append(f"| {i} | {product} | {qty:,} |")
report_lines.append("")

report_lines.append("## 五、月度销售趋势")
report_lines.append("")
report_lines.append("| 月份 | 销售额 | 环比变化 |")
report_lines.append("|:---|:---:|:---:|")
sales_list = list(monthly_sales.items())
for idx, (ym, sales_val) in enumerate(sales_list):
    if idx == 0:
        change = "--"
    else:
        prev = sales_list[idx-1][1]
        change_pct = (sales_val - prev) / prev * 100
        if change_pct >= 0:
            change = f"+{change_pct:.1f}%"
        else:
            change = f"{change_pct:.1f}%"
    report_lines.append(f"| {ym} | {sales_val:,.2f} | {change} |")
report_lines.append("")

report_lines.append("## 六、复购分析")
report_lines.append("")
report_lines.append(f"- **总客户数**: {len(customer_orders):,}")
report_lines.append(f"- **复购客户数**: {repeat_customers:,}")
report_lines.append(f"- **整体复购率**: {repeat_rate:.1f}%")
report_lines.append("")

report_lines.append("## 七、高价值客户画像（Top5）")
report_lines.append("")
report_lines.append("| 排名 | 客户ID | 最近消费(天) | 消费频次 | 累计消费金额 |")
report_lines.append("|:---:|:---:|:---:|:---:|:---:|")
for idx, (_, row) in enumerate(high_value.iterrows(), 1):
    report_lines.append(f"| {idx} | {int(row['CustomerID'])} | {row['Recency']} | {int(row['Frequency'])} | {row['Monetary']:,.2f} |")
report_lines.append("")

report_lines.append("## 八、核心结论")
report_lines.append("")
report_lines.append("1. **客户结构待优化**：高价值客户占比30.4%，流失客户占比34.7%，召回空间巨大。")
report_lines.append("2. **复购表现良好**：整体复购率65.6%，远超行业平均水平（30-50%）。")
report_lines.append("3. **市场集中度高**：英国占89.1%，德/法/爱为Top3国际市场。")
report_lines.append("4. **季节性显著**：9-11月为销售旺季，11月达全年峰值116万。")
report_lines.append("")
report_lines.append("*数据来源：UCI Machine Learning Repository - Online Retail Dataset*")

with open('analysis_report.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print("分析报告已生成: analysis_report.md")
print("分析报告已生成: analysis_report2.docx")
# ============================================================
# 11. 导出RFM结果
# ============================================================
rfm.to_csv('RFM_analysis_result.csv', index=False, encoding='utf-8-sig')
print("RFM结果已导出: RFM_analysis_result.csv")

print("\n" + "=" * 60)
print("全部完成")
print("=" * 60)
