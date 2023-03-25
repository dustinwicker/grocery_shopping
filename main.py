import os
import base64
import config
import requests
import json
import pandas as pd

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
        'filter.term': 'kale' #apples
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
