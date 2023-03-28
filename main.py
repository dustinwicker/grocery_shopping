import os
import base64
import config
import requests
import json
import pandas as pd
import numpy as np

oz_in_lb = 16

os.environ['kroger_client_id'] = 'groceryprice-70f7c4386bf8eae37dfb0ae863aa267c7804631109058859652'
os.environ['kroger_client_secret'] = '9wYCMCSJm3mg0sRWNp7Hlaaykk9wIi_HUzZX8ACh'

client_id = os.environ['kroger_client_id']
client_secret = os.environ['kroger_client_secret']

# Authentication requires base64 encoded id:secret, which is precalculated here
encoded_client_token = base64.b64encode(f"{client_id}:{client_secret}".encode('ascii')).decode('ascii')

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
params = {
        'filter.locationId': edgewater_location_id,
        'filter.fulfillment':'csp',
        'filter.term': 'fresh vegatables', #apples #kale #spinach,
        'filter.limit':50#,
        #'filter.start':50
}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
v = pd.DataFrame(json.loads(response_three.text)['data'])
params = {
        'filter.locationId': edgewater_location_id,
        'filter.fulfillment':'csp',
        'filter.term': 'fresh vegatables', #apples #kale #spinach,
        'filter.limit':50,
        'filter.start':50
}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
v_ = pd.DataFrame(json.loads(response_three.text)['data'])
ve=pd.concat([v,v_],axis=0)
ve = ve.drop(columns=['productId', 'upc', 'images', 'itemInformation', 'temperature'])
ve = pd.concat([ve.drop(['items'], axis=1), ve['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
ve = pd.concat([ve.drop(['price'], axis=1), ve['price'].apply(pd.Series)], axis=1)
ve['size'].unique()

ve['size_oz'] = np.nan
ve.loc[ve['size'].str.contains('lb|oz'), 'size_oz'] = ve.loc[ve['size'].str.contains('lb|oz'), 'size']

ve.loc[ve['size'].str.contains('lb', na=False), 'size_oz'] = \
    ve.loc[ve['size'].str.contains('lb', na=False), 'size'].str.replace('lb','').str.strip().astype(float) * oz_in_lb

ve.loc[ve['size_oz'].str.contains('oz',na=False), 'size_oz'] = \
    ve.loc[ve['size_oz'].str.contains('oz',na=False), 'size_oz'].str.replace('oz','').str.strip().astype(float)

ve['regular_per_size_oz'] = ve['regular']/ve['size_oz']
ve['promo_per_size_oz'] = ve['promo']/ve['size_oz']
ve['pct_change']=((ve.promo_per_size_oz - ve.regular_per_size_oz)/ve.regular_per_size_oz)*100
ve.loc[ve['promo_per_size_oz']>0.0].sort_values(by=['promo_per_size_oz', 'regular_per_size_oz'])
ve[['brand','description','regular','promo','regularPerUnitEstimate','promoPerUnitEstimate','size_oz',
   'regular_per_size_oz', 'promo_per_size_oz', 'pct_change']].loc[ve['promo_per_size_oz']>0.0].sort_values(by=['promo_per_size_oz', 'regular_per_size_oz'])

ve.sort_values(by=['promo_per_size_oz', 'regular_per_size_oz'])
ve[['brand','description','regular','promo','regularPerUnitEstimate','promoPerUnitEstimate','size_oz',
   'regular_per_size_oz', 'promo_per_size_oz', 'pct_change']].sort_values(by=['promo_per_size_oz', 'regular_per_size_oz'])

ve['size_each'] = np.nan
ve.loc[ve['size'].str.contains('bunch|ct|each'), 'size_each'] = ve.loc[ve['size'].str.contains('bunch|ct|each'), 'size']

ve.loc[ve['size']=='each','size'] = '1 each'
ve.loc[ve['size'].str.contains('bunch|ct|each', na=False), 'size_each'] = \
    ve.loc[ve['size'].str.contains('bunch|ct|each', na=False), 'size'].str.replace('bunch','').str.replace('ct','').str.replace('each','').str.strip().astype(float)
ve['regular_per_size_each'] = ve['regular']/ve['size_each']
ve['promo_per_size_each'] = ve['promo']/ve['size_each']

ve[['description','regular_per_size_oz']].dropna()
ve[['description','regular_per_size_oz']].dropna()