# 🛒 Retail Customer Behavior Analysis (Market Basket Analysis)

Discovers which products customers frequently buy together, using the **Apriori algorithm** on retail transaction data, and turns the findings into product recommendations.

![Python](https://img.shields.io/badge/Python-3.x-blue) ![mlxtend](https://img.shields.io/badge/mlxtend-Apriori-green) ![Pandas](https://img.shields.io/badge/Pandas-data%20processing-150458)

## 📌 Overview
Retailers use purchase patterns for store layout, bundling, promotions and cross-selling. This project processes raw transactions from the Groceries dataset, mines frequent itemsets, generates association rules and provides a simple recommendation function.

## ✨ Highlights
- Raw transaction data processed with **Python and Pandas**
- Baskets converted to a one-hot encoded matrix
- Frequent itemsets mined with **Apriori (mlxtend)**
- Association rules ranked by **support, confidence and lift**
- **Recommendation function:** input a product, get related products
- Visualisations of top items and rules

## 🧰 Tech Stack
Python · Pandas · mlxtend · Matplotlib

## 📂 Project Structure
> Adjust names to match your files.
```
retail-customer-behavior-analysis/
├── data/            # Groceries dataset
├── src/             # preprocessing, apriori, rules, recommender scripts
├── images/          # charts
├── requirements.txt
└── README.md
```

## 🔬 Methodology
1. **Preprocessing:** clean transactions and one-hot encode baskets.
2. **Frequent itemsets:** Apriori with a minimum support threshold.
3. **Association rules:** filter by confidence and lift.
4. **Recommendation:** function that returns products linked to a given item.
5. **Visualisation:** top items and rule plots.

### Metrics
- **Support:** how often items occur together
- **Confidence:** how often B is bought when A is bought
- **Lift:** how much more likely B is bought with A than by chance

## 📊 Sample Output
| Rule | Support | Confidence | Lift |
|---|---|---|---|
| _Item A → Item B_ | _value_ | _value_ | _value_ |

_Add your charts here:_ `![Top Items](images/top_items.png)`

## 🚀 Getting Started
```bash
git clone https://github.com/sayandipnandi-2/retail-customer-behavior-analysis.git
cd retail-customer-behavior-analysis
pip install -r requirements.txt
python src/apriori_analysis.py
```

### Get recommendations
```python
recommend("whole milk")   # returns products frequently bought with it
```

## 🔭 Future Work
- Try FP-Growth for faster mining on larger data
- Add customer segments or time-based patterns
- Deploy the recommender as a small web app

## 📚 References
- Agrawal & Srikant (1994), *Fast Algorithms for Mining Association Rules*
- Raschka (2018), *MLxtend*, Journal of Open Source Software

## 👤 Author
**Sayandip Nandi** — B.Tech ECE, Institute of Engineering and Management, Kolkata
[LinkedIn](https://www.linkedin.com/in/sayandip2004/) · [GitHub](https://github.com/sayandipnandi-2) · sayandipnandi0@gmail.com
