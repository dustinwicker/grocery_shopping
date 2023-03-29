import os
import base64
import config
import requests
import json
import pandas as pd
import numpy as np

oz_in_lb = 16
oz_in_gal = 128

client_id = os.environ['kroger_client_id']
client_secret = os.environ['kroger_client_secret']
# Authentication requires base64 encoded id:secret, which is precalculated here
encoded_client_token = base64.b64encode(f"{client_id}:{client_secret}".encode('ascii')).decode('ascii')

# obtain access token
api_url = 'https://api.kroger.com/v1'
url = api_url + '/connect/oauth2/token'
headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': f'Basic {encoded_client_token}',
    }
payload = {
        'grant_type': "client_credentials",
        'scope': ['product.compact'],
    }
# figure out why verify = False works
response = requests.post(url, headers=headers, data=payload, verify=False)
print(response.status_code)
access_token = json.loads(response.text).get('access_token')

# Determine Edgewater King Sooper's location information to search store for products
url = api_url + '/locations'
headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
}
params = {
        'filter.zipCode.near': '80204'
}
response_two = requests.get(url, headers=headers, params=params, verify=False)
print(response_two.status_code)

# Create DataFrame
location_df = pd.DataFrame(json.loads(response_two.text)['data'])[['locationId', 'chain', 'address']]
# Drop address column, split address column (dict) into separate columns
location_df = pd.concat([location_df.drop(['address'], axis=1), location_df['address'].apply(pd.Series)], axis=1)
edgewater_location_id = location_df.loc[( location_df.addressLine1.str.contains('1725 Sheridan') ) &
                                        ( location_df.city=='Edgewater' ), 'locationId'].values[0]

# Search products at the Edgewater King Soopers
url = api_url + '/products'
headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
}
params = {
        'filter.locationId': edgewater_location_id,
        'filter.term': 'coffee', #apples #kale #spinach
        'filter.limit': 50,
        'page':1
}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)

df = pd.DataFrame(json.loads(response_three.text)['data'])
# drop 'images'
df = df[['productId', 'upc', 'aisleLocations', 'brand', 'categories',
       'countryOrigin', 'description', 'items', 'itemInformation',
       'temperature']].sort_values(by=['brand', 'description'])
df = pd.concat([df.drop(['items'], axis=1), df['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
df = pd.concat([df.drop(['price'], axis=1), df['price'].apply(pd.Series)], axis=1)

df.loc[(df['size'].str.contains('lb')) & (df['regularPerUnitEstimate'].isna()), 'regularPerUnitEstimate'] = \
    df.loc[(df['size'].str.contains('lb')) & (df['regularPerUnitEstimate'].isna()),'regular']/\
    df.loc[(df['size'].str.contains('lb')) & (df['regularPerUnitEstimate'].isna()),'size'].str.replace(' lb','').astype(int)
df = df.rename(columns={'regularPerUnitEstimate':'regularPerWeight'})
df.regularPerWeight = df.regularPerWeight.round(2)

df.loc[(df['size'].str.contains('lb')) & (df['promoPerUnitEstimate'].isna()), 'promoPerUnitEstimate'] = \
    df.loc[(df['size'].str.contains('lb')) & (df['promoPerUnitEstimate'].isna()),'promo']/\
    df.loc[(df['size'].str.contains('lb')) & (df['promoPerUnitEstimate'].isna()),'size'].str.replace(' lb','').astype(int)
df = df.rename(columns={'promoPerUnitEstimate':'promoPerWeight'})
df.promoPerWeight = df.promoPerWeight.round(2)
df.sort_values(by=['promoPerWeight', 'regularPerWeight'])[['description', 'regularPerWeight', 'promoPerWeight']]

kale = pd.DataFrame(json.loads(response_three.text)['data'])
# drop productId, upc, aisleLocations, images, itemInformation, temperature
kale = kale.drop(columns=['productId', 'upc', 'aisleLocations', 'images', 'itemInformation', 'temperature'])
kale = pd.concat([kale.drop(['items'], axis=1), kale['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
kale = pd.concat([kale.drop(['price'], axis=1), kale['price'].apply(pd.Series)], axis=1)
kale = pd.concat([kale.drop(['fulfillment'], axis=1), kale['fulfillment'].apply(pd.Series)], axis=1)
kale['size_oz'] = np.nan
kale.loc[kale['size'].str.contains('lb|oz'), 'size_oz'] = kale['size']
kale.loc[kale['size'].str.contains('lb', na=False), 'size_oz'] = \
    kale.loc[kale['size'].str.contains('lb', na=False), 'size'].str.replace('lb','').str.strip().astype(float) * oz_in_lb
kale.loc[kale['size_oz'].str.contains('oz',na=False), 'size_oz'] = \
    kale.loc[kale['size_oz'].str.contains('oz',na=False), 'size_oz'].str.replace('oz','').str.strip().astype(float)
kale['regular_per_size_oz'] = kale['regular']/kale['size_oz']
kale['promo_per_size_oz'] = kale['promo']/kale['size_oz']

# get all pages
spinach = pd.DataFrame(json.loads(response_three.text)['data'])
spinach = spinach.drop(columns=['productId', 'upc', 'aisleLocations', 'images', 'itemInformation', 'temperature'])
spinach = pd.concat([spinach.drop(['items'], axis=1), spinach['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
spinach = pd.concat([spinach.drop(['price'], axis=1), spinach['price'].apply(pd.Series)], axis=1)
spinach = pd.concat([spinach.drop(['fulfillment'], axis=1), spinach['fulfillment'].apply(pd.Series)], axis=1)
spinach['size_oz'] = np.nan
spinach.loc[spinach['size'].str.contains('lb|oz'), 'size_oz'] = spinach['size']

spinach.loc[spinach['size'].str.contains('lb', na=False), 'size_oz'] = \
    spinach.loc[spinach['size'].str.contains('lb', na=False), 'size'].str.replace('lb','').str.strip().astype(float) * oz_in_lb

spinach.loc[spinach['size_oz'].str.contains('oz',na=False), 'size_oz'] = \
    spinach.loc[spinach['size_oz'].str.contains('oz',na=False), 'size_oz'].str.replace('oz','').str.strip().astype(float)

spinach['regular_per_size_oz'] = spinach['regular']/spinach['size_oz']
spinach['promo_per_size_oz'] = spinach['promo']/spinach['size_oz']


# coffee
params = {
        'filter.locationId': edgewater_location_id,
        'filter.fulfillment':'csp',
        'filter.term': 'whole bean coffee', #apples #kale #spinach,
        'filter.limit':50#,
        #'filter.start':50
}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
c = pd.DataFrame(json.loads(response_three.text)['data'])
params = {
        'filter.locationId': edgewater_location_id,
        'filter.fulfillment':'csp',
        'filter.term': 'whole bean coffee', #apples #kale #spinach,
        'filter.limit':50,
        'filter.start':50
}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
c_ = pd.DataFrame(json.loads(response_three.text)['data'])
coffee = pd.concat([c,c_],axis=0)
coffee = coffee.drop(columns=['productId', 'upc', 'images', 'itemInformation', 'temperature'])
coffee = pd.concat([coffee.drop(['items'], axis=1), coffee['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
coffee = pd.concat([coffee.drop(['price'], axis=1), coffee['price'].apply(pd.Series)], axis=1)
coffee['size_oz'] = np.nan
coffee.loc[coffee['size'].str.contains('lb|oz'), 'size_oz'] = coffee['size']
coffee.loc[coffee['size_oz'].str.contains('oz',na=False), 'size_oz'] = \
    coffee.loc[coffee['size_oz'].str.contains('oz',na=False), 'size_oz'].str.replace('oz','').str.strip().astype(float)
coffee['regular_per_size_oz'] = coffee['regular']/coffee['size_oz']
coffee['promo_per_size_oz'] = coffee['promo']/coffee['size_oz']
coffee['pct_change']=((coffee.promo_per_size_oz - coffee.regular_per_size_oz)/coffee.regular_per_size_oz)*100
coffee.loc[coffee['promo_per_size_oz']>0.0].sort_values(by=['promo_per_size_oz', 'regular_per_size_oz'])

# decaf coffee
params = {
        'filter.locationId': edgewater_location_id,
        'filter.fulfillment':'csp',
        'filter.term': 'whole bean decaf coffee', #apples #kale #spinach,
        'filter.limit':50#,
        #'filter.start':50
}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
c = pd.DataFrame(json.loads(response_three.text)['data'])
params = {
        'filter.locationId': edgewater_location_id,
        'filter.fulfillment':'csp',
        'filter.term': 'whole bean decaf coffee', #apples #kale #spinach,
        'filter.limit':50,
        'filter.start':50
}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
c_ = pd.DataFrame(json.loads(response_three.text)['data'])
coffee = pd.concat([c,c_],axis=0)
coffee = coffee.drop(columns=['productId', 'upc', 'images', 'itemInformation', 'temperature'])
coffee = pd.concat([coffee.drop(['items'], axis=1), coffee['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
coffee = pd.concat([coffee.drop(['price'], axis=1), coffee['price'].apply(pd.Series)], axis=1)
coffee['size_oz'] = np.nan
coffee.loc[coffee['size'].str.contains('lb|oz'), 'size_oz'] = coffee['size']
coffee.loc[coffee['size_oz'].str.contains('oz',na=False), 'size_oz'] = \
    coffee.loc[coffee['size_oz'].str.contains('oz',na=False), 'size_oz'].str.replace('oz','').str.strip().astype(float)
coffee['regular_per_size_oz'] = coffee['regular']/coffee['size_oz']
coffee['promo_per_size_oz'] = coffee['promo']/coffee['size_oz']
coffee['pct_change']=((coffee.promo_per_size_oz - coffee.regular_per_size_oz)/coffee.regular_per_size_oz)*100
coffee.loc[coffee['promo_per_size_oz']>0.0].sort_values(by=['promo_per_size_oz', 'regular_per_size_oz'])

# vegetables
url = api_url + '/products'
headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
}
ve = pd.DataFrame()
for s in [1,50,100,150,200,250]:
    print(s)
    params = {'filter.locationId': edgewater_location_id,
          'filter.fulfillment':'csp',
          'filter.term': 'fresh vegetables', #apples #kale #spinach,
          'filter.limit': 50,
          'filter.start': s}
    response_three = requests.get(url, headers=headers, params=params, verify=False)
    print(response_three.status_code)
    v = pd.DataFrame(json.loads(response_three.text)['data'])
    ve = pd.concat([ve, v], axis=0)
ve = ve.drop(columns=['productId', 'upc', 'images', 'itemInformation', 'temperature'])
ve = pd.concat([ve.drop(['items'], axis=1), ve['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
ve = pd.concat([ve.drop(['price'], axis=1), ve['price'].apply(pd.Series)], axis=1)

# clean up misc. sizes
ve.loc[ve['size'] == 'each', 'size'] = '1 each'
ve.loc[ve['size'] == '1 pt / 10 oz', 'size'] = '10 oz'
ve.loc[ve['size'] == '4 ct / 3 oz', 'size'] = '12 oz'
ve.loc[ve['size'] == '4 ct / 15.25 oz', 'size'] = '61 oz'

# create size_a column
ve['size_a'] = ve['size'].apply(lambda x: x.split())

# create size_oz column (can compare oz, lb, fl oz)
ve['size_oz'] = np.nan
# oz, oz., fl oz, fl oz.
# lb, lb.
ve['size_oz'] = ve['size_a'].apply(lambda x: float(x[0]) if ( (x[-1] == 'oz' or x[-1] == 'oz.') and (len(x) == 2 or len(x) == 3) )
                                    else ( float(x[0])*oz_in_lb if ( (x[-1] == 'lb' or x[-1] == 'lb.') and (len(x) == 2) ) else np.nan ) )

# create size_each column (bunch, ct, each)
ve['size_each'] = np.nan
ve['size_each'] = ve['size_a'].apply(lambda x: float(x[0]) if (x[-1] == 'ct' or x[-1] == 'each' or x[-1] == 'bunch' ) and (len(x) == 2) else np.nan )

# check to see if any products remain that need sizing information
print(ve.loc[ (ve['size_oz'].isna() & ve['size_each'].isna()), ['description', 'size_a', 'size']])

# column creation
ve['regular_per_size_oz'] = ve['regular']/ve['size_oz']
ve['promo_per_size_oz'] = ve['promo']/ve['size_oz']
ve['pct_change_regular_to_promo_size_oz']=((ve.promo_per_size_oz - ve.regular_per_size_oz)/ve.regular_per_size_oz)*100

ve['regular_per_size_each'] = ve['regular']/ve['size_each']
ve['promo_per_size_each'] = ve['promo']/ve['size_each']
ve['pct_change_regular_to_promo_size_each']=((ve.promo_per_size_each - ve.regular_per_size_each)/ve.regular_per_size_each)*100

# size_oz price
veg_size_oz = pd.concat([
    ve[['description','size','regular', 'promo', 'regular_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'regular_per_size_oz':'per_size_oz'}),
    ve.loc[ve.promo_per_size_oz>0,['description','size','regular', 'promo', 'promo_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'promo_per_size_oz':'per_size_oz'})
    ]).sort_values(by=['per_size_oz']).drop_duplicates(subset='description', keep='first')
veg_size_oz['per_size_rank'] = veg_size_oz.groupby('per_size_oz')['per_size_oz'].transform('mean').rank(method='dense',ascending=True)

# size_each price
veg_size_each = pd.concat([
    ve[['description','size','regular', 'promo', 'regular_per_size_each', 'pct_change_regular_to_promo_size_each']].dropna().rename(columns={'regular_per_size_each':'per_size_each'}),
    ve.loc[ve.promo_per_size_each>0,['description','size','regular', 'promo', 'promo_per_size_each', 'pct_change_regular_to_promo_size_each']].dropna().rename(columns={'promo_per_size_each':'per_size_each'})
    ]).sort_values(by=['per_size_each']).drop_duplicates(subset='description', keep='first')
veg_size_each['per_size_rank'] = veg_size_each.groupby('per_size_each')['per_size_each'].transform('mean').rank(method='dense',ascending=True)

# decaf and herbal tea
url = api_url + '/products'
headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
}
# decaf
decaf_tea = pd.DataFrame()
for s in [1,50,100]:
    print(s)
    params = {'filter.locationId': edgewater_location_id,
          'filter.fulfillment':'csp',
          'filter.term': 'decaf tea', #apples #kale #spinach,
          'filter.limit': 50}
    response_three = requests.get(url, headers=headers, params=params, verify=False)
    print(response_three.status_code)
    t = pd.DataFrame(json.loads(response_three.text)['data'])
    tea = pd.concat([tea, t], axis=0)
tea = tea.drop(columns=['productId', 'upc', 'images', 'itemInformation', 'temperature'])
tea = pd.concat([tea.drop(['items'], axis=1), tea['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
tea = pd.concat([tea.drop(['price'], axis=1), tea['price'].apply(pd.Series)], axis=1)

tea = tea.loc[ (tea.description.str.contains('Decaf') & tea.description.str.contains('Tea'))]

# create size_a column
tea['size_a'] = tea['size'].apply(lambda x: x.split())

# create size_oz column (can compare oz, lb, fl oz)
tea['size_oz'] = np.nan
# oz, oz., fl oz, fl oz.
# lb, lb.
tea['size_oz'] = tea['size_a'].apply(lambda x: float(x[0]) if ( (x[-1] == 'oz' or x[-1] == 'oz.') and (len(x) == 2 or len(x) == 3) )
                                    else ( float(x[0])*oz_in_gal if ( (x[-1] == 'gal' or x[-1] == 'gal.') and (len(x) == 2) ) else np.nan ) )

# create size_each column (bunch, ct, each)
tea['size_ct'] = np.nan
tea['size_ct'] = tea['size_a'].apply(lambda x: float(x[0]) if (x[-1] == 'ct' ) and (len(x) == 2) else np.nan )

# check to see if any products remain that need sizing information
print(tea.loc[ (tea['size_oz'].isna() & tea['size_ct'].isna()), ['description', 'size_a', 'size']])

# column creation
tea['regular_per_size_oz'] = tea['regular']/tea['size_oz']
tea['promo_per_size_oz'] = tea['promo']/tea['size_oz']
tea['pct_change_regular_to_promo_size_oz']=((tea.promo_per_size_oz - tea.regular_per_size_oz)/tea.regular_per_size_oz)*100

tea['regular_per_size_ct'] = tea['regular']/tea['size_ct']
tea['promo_per_size_ct'] = tea['promo']/tea['size_ct']
tea['pct_change_regular_to_promo_size_ct']=((tea.promo_per_size_ct - tea.regular_per_size_ct)/tea.regular_per_size_ct)*100

# size_oz price
tea_size_oz = pd.concat([
    tea[['description','size','regular', 'promo', 'regular_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'regular_per_size_oz':'per_size_oz'}),
    tea.loc[tea.promo_per_size_oz>0,['description','size','regular', 'promo', 'promo_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'promo_per_size_oz':'per_size_oz'})
    ]).sort_values(by=['per_size_oz']).drop_duplicates(subset='description', keep='first')
tea_size_oz['per_size_rank'] = tea_size_oz.groupby('per_size_oz')['per_size_oz'].transform('mean').rank(method='dense',ascending=True)

# size_each price
tea_size_ct = pd.concat([
    tea[['description','size','regular', 'promo', 'regular_per_size_ct', 'pct_change_regular_to_promo_size_ct']].dropna().rename(columns={'regular_per_size_ct':'per_size_ct'}),
    tea.loc[tea.promo_per_size_ct>0,['description','size','regular', 'promo', 'promo_per_size_ct', 'pct_change_regular_to_promo_size_ct']].dropna().rename(columns={'promo_per_size_ct':'per_size_ct'})
    ]).sort_values(by=['per_size_ct']).drop_duplicates(subset='description', keep='first')
tea_size_ct['per_size_rank'] = tea_size_ct.groupby('per_size_ct')['per_size_ct'].transform('mean').rank(method='dense',ascending=True)