# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

# Load the data
df = pd.read_excel("SodaSales.xlsx", sheet_name='MMM Data')
df.head()

# Convert "Month" column to datetime and set it as index

df['Month'] = pd.to_datetime(df['Month'])
df.set_index(df['Month'], inplace=True)
df.head()

# Plot monthly "SalesVol" to assess the seasonlity in the data
plt.figure(figsize=(12,8))
plt.plot(df['SalesVol'], marker='o')
plt.title('Monthly Sales Volume')
plt.xlabel('Month')
plt.ylabel('Sales Volume')
plt.grid(True)
plt.savefig('Monthly_Sales_Volume.png')
plt.show()

# Decomposition into seasonal component
decomposition_salesVol = seasonal_decompose(df['SalesVol'], model='additive', period=12)

# Plot the seasonal decomposition
plt.figure(figsize=(20,10))
decomposition_salesVol.plot()
plt.xticks(rotation='vertical')
plt.savefig("Seasonal_decomposition.png")
plt.show()

# Compute correlation matrix between sales and media advertising
df.columns
correlation_media = df[['SalesVol', 'TVGrP', 'InstoreAds', 'OutdoorAds','DigitalAds']].corr()
correlation_media

# Get the correlation between Sales and Media types
df_corr_sales_media = correlation_media['SalesVol'][['TVGrP', 'InstoreAds', 'OutdoorAds','DigitalAds']]
df_corr_sales_media

# Correlation between Sales and Price
sales_price_corr = df['SalesVol'].corr(df['Price'])
sales_price_corr

# Correlations between Sales and Promotion
sales_promo_corr = df['SalesVol'].corr(df['Promotion'])
sales_promo_corr

# Compute partial correlation between Sales and Promotion removing the effect of
# media advt

# Calculate rediual of SalesVol and Media Advt OLS regression
x_media = df[['TVGrP', 'InstoreAds', 'OutdoorAds','DigitalAds']]
y_sales = df.loc[x_media.index, 'SalesVol']
x_media_constant = sm.add_constant(x_media)

model_sales_advt = sm.OLS(y_sales, x_media_constant).fit()
model_sales_advt_resid = model_sales_advt.resid

# Calculate residual of Promotion and Media Advt OLS regression
y_promo = df.loc[x_media.index, 'Promotion']

model_promo_advt = sm.OLS(y_promo, x_media_constant).fit()
model_promo_advt_resid = model_promo_advt.resid

# Calculate correlation between the residuals
partial_corr = model_sales_advt_resid.corr(model_promo_advt_resid)
partial_corr

# Compute the correlation matrix between "Sales" and "Competitior"
df.columns
correlation_comp = df[['SalesVol','Comp1TV', 'Comp1NPapers', 'Comp1OOH',
                       'Comp2NP']].corr()
correlation_comp

df_corr_sales_comp = correlation_comp['SalesVol'][['Comp1TV', 'Comp1NPapers',
                                                  'Comp1OOH','Comp2NP']]
df_corr_sales_comp