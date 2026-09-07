import pandas as pd
import datetime as dt

# ============================================================
# 1. 读取数据
# ============================================================
df = pd.read_csv('OnlineRetail.csv')

print("=" * 60)
print("跨境电商数据分析 - 完整报告")
print("=" * 60)

print(f"\n原始数据量: {len(df)} 条记录")


# ============================================================
# 2. 数据清洗
# ============================================================
# 转换日期格式
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 去除空CustomerID
df_clean = df.dropna(subset=['CustomerID'])

# 去掉退货记录（数量为负）
df_clean = df_clean[df_clean['Quantity'] > 0]

# 去掉价格异常
df_clean = df_clean[df_clean['UnitPrice'] > 0]

# 计算每条记录的总价
df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

print(f"清洗后数据量: {len(df_clean)} 条记录")


# ============================================================
# 3. 核心指标
# ============================================================
total_sales = df_clean['TotalPrice'].sum()
avg_order = df_clean.groupby('InvoiceNo')['TotalPrice'].sum().mean()
unique_customers = df_clean['CustomerID'].nunique()
unique_countries = df_clean['Country'].nunique()

print("\n" + "=" * 60)
print("核心经营指标")
print("=" * 60)
print(f"总销售额: {total_sales:,.2f}")
print(f"平均客单价: {avg_order:,.2f}")
print(f"客户总数: {unique_customers:,}")
print(f"覆盖国家数: {unique_countries}")


# ============================================================
# 4. RFM客户分层分析
# ============================================================
snapshot_date = df_clean['InvoiceDate'].max() + dt.timedelta(days=1)

rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# RFM打分（1-4分）
rfm['R_rank'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
rfm['F_rank'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
rfm['M_rank'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4])

# 客户分层
def rfm_segment(row):
    if row['R_rank'] in [3, 4] and row['F_rank'] in [3, 4] and row['M_rank'] in [3, 4]:
        return '高价值客户'
    elif row['R_rank'] in [3, 4] and row['F_rank'] in [1, 2]:
        return '新客户'
    elif row['R_rank'] in [1, 2] and row['F_rank'] in [3, 4]:
        return '忠诚客户'
    elif row['R_rank'] in [1, 2] and row['F_rank'] in [1, 2]:
        return '流失客户'
    else:
        return '一般客户'

rfm['Segment'] = rfm.apply(rfm_segment, axis=1)

print("\n" + "=" * 60)
print("客户分层统计")
print("=" * 60)
segment_counts = rfm['Segment'].value_counts()
for seg, count in segment_counts.items():
    print(f"{seg}: {count} 人 ({count/len(rfm)*100:.1f}%)")


# ============================================================
# 5. 市场分析
# ============================================================
print("\n" + "=" * 60)
print("国家市场分布（Top10）")
print("=" * 60)
country_counts = df_clean['Country'].value_counts().head(10)
for country, count in country_counts.items():
    print(f"{country}: {count} 条订单")


# ============================================================
# 6. 热销商品分析
# ============================================================
print("\n" + "=" * 60)
print("热销商品Top10（按销量）")
print("=" * 60)
top_products = df_clean.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
for i, (product, qty) in enumerate(top_products.items(), 1):
    print(f"{i}. {product}: {qty} 件")


# ============================================================
# 7. 留存分析（复购率）
# ============================================================
customer_orders = df_clean.groupby('CustomerID')['InvoiceNo'].nunique()
repeat_customers = customer_orders[customer_orders >= 2].count()
repeat_rate = repeat_customers / len(customer_orders) * 100

print("\n" + "=" * 60)
print("留存分析")
print("=" * 60)
print(f"总客户数: {len(customer_orders)}")
print(f"复购客户数: {repeat_customers}")
print(f"复购率: {repeat_rate:.1f}%")


# ============================================================
# 8. 月度趋势分析
# ============================================================
df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M')
monthly_sales = df_clean.groupby('YearMonth')['TotalPrice'].sum()

print("\n" + "=" * 60)
print("月度销售趋势")
print("=" * 60)
for ym, sales in monthly_sales.items():
    print(f"{ym}: {sales:,.2f}")


# ============================================================
# 9. 导出结果
# ============================================================
# 导出RFM结果到CSV
rfm.to_csv('RFM_analysis_result.csv', index=False,encoding='utf-8-sig')
print("\n" + "=" * 60)
print("RFM分析结果已保存至: RFM_analysis_result.csv")
print("=" * 60)

print("=" * 60)
print("客户分层统计")
print("=" * 60)
print(rfm['Segment'].value_counts())

print("\n分层占比")
print(rfm['Segment'].value_counts(normalize=True).map(lambda x: f"{x:.1%}"))

print("\n高价值客户Top5（按消费金额排序）")
print(rfm[rfm['Segment'] == '高价值客户'].head(5))

print(f"\n整体复购率: {len(rfm[rfm['Frequency'] > 1]) / len(rfm):.1%}")