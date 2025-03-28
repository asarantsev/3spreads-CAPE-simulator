# 3spreads-CAPE-simulator
Simulation of stock index annual returns (nominal/real, price/total) based on trailing earnings yield and BAA-AAA, AAA-Long, Long-Short bond spreads. This extends the work in the other repository 3spreads-1yearnyield-returns covered in the blog post https://my-finance.org/2025/03/19/earnings-yield-3-bond-spreads-annual-returns/ As usual, we use annual volatility for linear regression, and we add it as another factor. The current research is covered in the post https://my-finance.org/2025/03/27/sp-returns-vs-bond-spreads-and-trailing-earnings-yield-with-volatility/ 

We add Parts I, II and modify part III of a previous blog post: https://my-finance.org/2025/03/19/earnings-yield-3-bond-spreads-annual-returns/ in our new code. We have only one Python file, as opposed to the three Python files in the repository 3spreads-1yearnyield-returns corresponding to Parts I, II, III of that previous post.

Update: Added rates-only.py for fitting the returns (price/total, nominal/real). Here, we remove earnings yield (classic or trailing version, original or logarithmic value). We model these returns only using the three spreads.
