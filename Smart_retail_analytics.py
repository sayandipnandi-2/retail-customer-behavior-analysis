import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("outputs", exist_ok=True)

# 1. Load the data (filename updated to match your explorer exactly)
df = pd.read_csv("data/groceries - groceries.csv", encoding="ISO-8859-1")

# 2. Inspect the first few rows 
# (Using print() to display the output in the VS Code terminal)
print("--- First 5 Rows ---")
print(df.head())

# 3. Inspect columns and data types
print("\n--- Dataset Info ---")
print(df.info())

# 4. Check for any missing/null values
print("\n--- Missing Values Count ---")
print(df.isnull().sum())

# Drop the "Item(s)" count column - we don't need it
items_df = df.drop(columns=["Item(s)"])

# Convert each row into a list of items, removing NaN values
transactions = items_df.apply(
    lambda row: [item for item in row if pd.notna(item)],
    axis=1
).tolist()

# Check the result
print(transactions[:5])

from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

basket_df = pd.DataFrame(te_array, columns=te.columns_)
print(basket_df.shape)
basket_df.head()

from mlxtend.frequent_patterns import apriori

frequent_itemsets = apriori(basket_df, min_support=0.01, use_colnames=True)

frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
print(frequent_itemsets.shape)
print(frequent_itemsets.head(10))

from mlxtend.frequent_patterns import association_rules

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values('lift', ascending=False)

print(rules.shape)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

def recommend_products(item, rules_df, top_n=5):
    # Find rules where the antecedent contains this single item
    matches = rules_df[rules_df['antecedents'].apply(lambda x: item in x and len(x) == 1)]
    matches = matches.sort_values('lift', ascending=False)
    
    if matches.empty:
        return f"No strong recommendations found for '{item}'."
    
    recommendations = []
    for _, row in matches.head(top_n).iterrows():
        recommendations.append({
            'recommended_item': list(row['consequents']),
            'confidence': round(row['confidence'], 3),
            'lift': round(row['lift'], 3)
        })
    return recommendations

# Test it
import pprint
pprint.pprint(recommend_products('whole milk', rules))
pprint.pprint(recommend_products('yogurt', rules))

import seaborn as sns

# 1. Top 10 most frequent items (bar chart)
plt.figure(figsize=(10,6))
top_items = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 1)]
top_items = top_items.sort_values('support', ascending=False).head(10)
top_items['item_name'] = top_items['itemsets'].apply(lambda x: list(x)[0])

sns.barplot(data=top_items, x='support', y='item_name', palette='viridis')
plt.title('Top 10 Most Frequently Purchased Items')
plt.xlabel('Support (fraction of transactions)')
plt.ylabel('Item')
plt.tight_layout()
plt.savefig('outputs/top_items.png')
plt.show()

# 2. Scatter plot: Support vs Confidence, colored by Lift
plt.figure(figsize=(10,6))
scatter = plt.scatter(rules['support'], rules['confidence'], 
                       c=rules['lift'], cmap='coolwarm', alpha=0.7)
plt.colorbar(scatter, label='Lift')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules: Support vs Confidence (colored by Lift)')
plt.tight_layout()
plt.savefig('outputs/rules_scatter.png')
plt.show()

# 3. Top 10 rules by lift (horizontal bar chart)
plt.figure(figsize=(10,6))
top_rules = rules.head(10).copy()
top_rules['rule'] = top_rules.apply(
    lambda r: f"{list(r['antecedents'])} → {list(r['consequents'])}", axis=1
)
sns.barplot(data=top_rules, x='lift', y='rule', palette='magma')
plt.title('Top 10 Association Rules by Lift')
plt.xlabel('Lift')
plt.ylabel('Rule')
plt.tight_layout()
plt.savefig('outputs/top_rules.png')
plt.show()