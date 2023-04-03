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

api_url = 'https://api.kroger.com/v1'

# obtain access token
def obtain_access_token():
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
    return access_token

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

# obtain access token using function
access_token = obtain_access_token()

# search products
url = api_url + '/products'

# coffee
filter_term = 'whole bean coffee'
params = {
        'filter.locationId': edgewater_location_id,
        'filter.fulfillment': 'csp',
        'filter.term': filter_term,
        'filter.limit': 50
}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
coffee = pd.DataFrame(json.loads(response_three.text)['data'])
params = {
        'filter.locationId': edgewater_location_id,
        'filter.fulfillment':'csp',
        'filter.term': filter_term,
        'filter.limit': 50,
        'filter.start': 50
}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
c = pd.DataFrame(json.loads(response_three.text)['data'])
coffee = pd.concat([coffee, c],axis=0)
coffee = coffee.drop(columns=['productId', 'upc', 'images', 'itemInformation', 'temperature'])
coffee = pd.concat([coffee.drop(['items'], axis=1), coffee['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
coffee = pd.concat([coffee.drop(['price'], axis=1), coffee['price'].apply(pd.Series)], axis=1)
coffee['size_oz'] = np.nan
coffee.loc[coffee['size'].str.contains('lb|oz'), 'size_oz'] = coffee['size']
coffee.loc[coffee['size_oz'].str.contains('oz',na=False), 'size_oz'] = \
    coffee.loc[coffee['size_oz'].str.contains('oz',na=False), 'size_oz'].str.replace('oz','').str.strip().astype(float)
coffee['regular_per_size_oz'] = coffee['regular']/coffee['size_oz']
coffee['promo_per_size_oz'] = coffee['promo']/coffee['size_oz']
coffee['pct_change_regular_to_promo_size_oz']=((coffee.promo_per_size_oz - coffee.regular_per_size_oz)/coffee.regular_per_size_oz)*100

# check to see if any products remain that need sizing information
print(coffee.loc[ coffee['size_oz'].isna(), ['description', 'size']])

coffee_size_oz = pd.concat([
    coffee[['description','size','regular', 'promo', 'regular_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'regular_per_size_oz':'per_size_oz'}),
    coffee.loc[coffee.promo_per_size_oz>0,['description','size','regular', 'promo', 'promo_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'promo_per_size_oz':'per_size_oz'})
    ]).sort_values(by=['per_size_oz']).drop_duplicates(subset='description', keep='first')
coffee_size_oz['per_size_rank'] = coffee_size_oz.groupby('per_size_oz')['per_size_oz'].transform('mean').rank(method='dense',ascending=True)
print(coffee_size_oz)

# decaf coffee
filter_term = 'whole bean decaf coffee'
params = {
        'filter.locationId': edgewater_location_id,
        'filter.fulfillment':'csp',
        'filter.term': filter_term,
        'filter.limit': 50
}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
decaf_coffee = pd.DataFrame(json.loads(response_three.text)['data'])
# params = {
#         'filter.locationId': edgewater_location_id,
#         'filter.fulfillment':'csp',
#         'filter.term': filter_term,
#         'filter.limit': 50,
#         'filter.start': 50
# }
# response_three = requests.get(url, headers=headers, params=params, verify=False)
# print(response_three.status_code)
# decaf_c = pd.DataFrame(json.loads(response_three.text)['data'])
# decaf_coffee = pd.concat([decaf_coffee, decaf_c, axis=0])
decaf_coffee = decaf_coffee.drop(columns=['productId', 'upc', 'images', 'itemInformation', 'temperature'])
decaf_coffee = pd.concat([decaf_coffee.drop(['items'], axis=1), decaf_coffee['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
decaf_coffee = pd.concat([decaf_coffee.drop(['price'], axis=1), decaf_coffee['price'].apply(pd.Series)], axis=1)
decaf_coffee['size_oz'] = np.nan
decaf_coffee.loc[decaf_coffee['size'].str.contains('lb|oz'), 'size_oz'] = decaf_coffee['size']
decaf_coffee.loc[decaf_coffee['size_oz'].str.contains('oz',na=False), 'size_oz'] = \
    decaf_coffee.loc[decaf_coffee['size_oz'].str.contains('oz',na=False), 'size_oz'].str.replace('oz','').str.strip().astype(float)
decaf_coffee['regular_per_size_oz'] = decaf_coffee['regular']/decaf_coffee['size_oz']
decaf_coffee['promo_per_size_oz'] = decaf_coffee['promo']/decaf_coffee['size_oz']
decaf_coffee['pct_change_regular_to_promo_size_oz']=((decaf_coffee.promo_per_size_oz - decaf_coffee.regular_per_size_oz)/decaf_coffee.regular_per_size_oz)*100

# check to see if any products remain that need sizing information
print(decaf_coffee.loc[ decaf_coffee['size_oz'].isna(), ['description', 'size']])

decaf_coffee_size_oz = pd.concat([
    decaf_coffee[['description','size','regular', 'promo', 'regular_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'regular_per_size_oz':'per_size_oz'}),
    decaf_coffee.loc[decaf_coffee.promo_per_size_oz>0,['description','size','regular', 'promo', 'promo_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'promo_per_size_oz':'per_size_oz'})
    ]).sort_values(by=['per_size_oz']).drop_duplicates(subset='description', keep='first')
decaf_coffee_size_oz['per_size_rank'] = decaf_coffee_size_oz.groupby('per_size_oz')['per_size_oz'].transform('mean').rank(method='dense',ascending=True)
print(decaf_coffee_size_oz)

# vegetables
filter_term = 'fresh vegetables'
params = {'filter.locationId': edgewater_location_id,
          'filter.fulfillment': 'csp',
          'filter.term': filter_term,
          'filter.limit': 50}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
v = pd.DataFrame(json.loads(response_three.text)['data'])

ve = pd.DataFrame()
for s in [50,100,150,200,250]:
    print(s)
    params = {'filter.locationId': edgewater_location_id,
          'filter.fulfillment':'csp',
          'filter.term': filter_term,
          'filter.limit': 50,
          'filter.start': s}
    response_three = requests.get(url, headers=headers, params=params, verify=False)
    print(response_three.status_code)
    v_ = pd.DataFrame(json.loads(response_three.text)['data'])
    ve = pd.concat([v, v_], axis=0)
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
# lb, lb., lbs
ve['size_oz'] = ve['size_a'].apply(lambda x: float(x[0]) if ( (x[-1] == 'oz' or x[-1] == 'oz.') and (len(x) == 2 or len(x) == 3) )
                                    else ( float(x[0])*oz_in_lb if ( (x[-1] == 'lb' or x[-1] == 'lb.' or x[-1] == 'lbs') and (len(x) == 2) ) else np.nan ) )

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
params = {'filter.locationId': edgewater_location_id,
          'filter.fulfillment':'csp',
          'filter.term': 'decaf tea', #apples #kale #spinach,
          'filter.limit': 50}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
decaf_tea = pd.DataFrame(json.loads(response_three.text)['data'])

# herbal
params = {'filter.locationId': edgewater_location_id,
          'filter.fulfillment':'csp',
          'filter.term': 'herbal tea', #apples #kale #spinach,
          'filter.limit': 50}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
h = pd.DataFrame(json.loads(response_three.text)['data'])

he = pd.DataFrame()
for s in [50,100,150]:
    print(s)
    params = {'filter.locationId': edgewater_location_id,
              'filter.fulfillment':'csp',
              'filter.term': 'herbal tea', #apples #kale #spinach,
              'filter.limit': 50,
              'filter.start':s}
    response_three = requests.get(url, headers=headers, params=params, verify=False)
    print(response_three.status_code)
    h_ = pd.DataFrame(json.loads(response_three.text)['data'])
    he = pd.concat([he, h_], axis=0)

tea = pd.concat([decaf_tea, h, he], axis=0)
tea = tea.drop(columns=['productId', 'upc', 'images', 'itemInformation', 'temperature'])
tea = pd.concat([tea.drop(['items'], axis=1), tea['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
tea = pd.concat([tea.drop(['price'], axis=1), tea['price'].apply(pd.Series)], axis=1)

# clean up misc. sizes - check these on kroger site ###
# size gives ct (number of tea bags) and oz (weight of package) - only need ct
tea.loc[(tea['description'].str.contains('Tea Bags|Teabags')) &
        (tea['size'].str.contains('ct')) & (tea['size'].str.contains('oz')), 'size'] = \
    tea.loc[ ( tea['description'].str.contains('Tea Bags|Teabags') ) &
         ( tea['size'].str.contains('ct')) & ( tea['size'].str.contains('oz')), 'size' ].apply(lambda x : x[:x.find('ct')+len('ct')])

# tea.loc[tea['size'] == '10 qt', 'size'] = ####
tea.loc[tea['size'] == '12 bottles / 16 fl oz', 'size'] = str(12*16) + ' fl oz'
tea.loc[tea['size'] == '12 ct / 1.16 oz', 'size'] = '12 ct'
tea.loc[tea['size'] == '16 ct / .99 oz', 'size'] = '16 ct'
tea.loc[tea['size'] == '16 ct / 1.13 oz', 'size'] = '16 ct'
tea.loc[tea['size'] == '4 ct / 12 oz', 'size'] = str(4*12) + ' fl oz'

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
tea['size_ct'] = tea['size_a'].apply(lambda x: float(x[0]) if (x[-1] == 'ct' ) and (len(x) == 2)
                                     else ( float(x[0][:-2]) if x[0][-2:] == 'ct' else np.nan) )

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

# fruit
url = api_url + '/products'
headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
}

params = {'filter.locationId': edgewater_location_id,
          'filter.fulfillment': 'csp',
          'filter.term': 'fruit',
          'filter.limit': 50}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
f = pd.DataFrame(json.loads(response_three.text)['data'])

fr = pd.DataFrame()
for s in range(50,250,50):
    print(s)
    params = {'filter.locationId': edgewater_location_id,
              'filter.fulfillment':'csp',
              'filter.term': 'fruit', #apples #kale #spinach,
              'filter.limit': 50,
              'filter.start':s}
    response_three = requests.get(url, headers=headers, params=params, verify=False)
    print(response_three.status_code)
    f_ = pd.DataFrame(json.loads(response_three.text)['data'])
    fr = pd.concat([fr, f_], axis=0)

fruit = pd.concat([f, fr], axis=0)
fruit = fruit.drop(columns=['productId', 'upc', 'images', 'itemInformation', 'temperature'])
fruit = pd.concat([fruit.drop(['items'], axis=1), fruit['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
fruit = pd.concat([fruit.drop(['price'], axis=1), fruit['price'].apply(pd.Series)], axis=1)

# create size_a column
fruit['size_a'] = fruit['size'].apply(lambda x: x.split())

# create size_oz column (can compare oz, lb, fl oz)
fruit['size_oz'] = np.nan
# oz, oz., fl oz, fl oz.
# lb, lbs
fruit['size_oz'] = fruit['size_a'].apply(lambda x: float(x[0]) if ( (x[-1] == 'oz' or x[-1] == 'oz.') and (len(x) == 2 or len(x) == 3) )
                                    else ( float(x[0])*oz_in_lb if ( (x[-1] == 'lb' or x[-1] == 'lbs') and (len(x) == 2) ) else np.nan ) )

# create size_each column (bunch, ct, each)
fruit['size_each'] = np.nan
fruit['size_each'] = fruit['size_a'].apply(lambda x: float(x[0]) if (x[-1] == 'ct' or x[-1] == 'each') and (len(x) == 2) else np.nan )

# check to see if any products remain that need sizing information
print(fruit.loc[ (fruit['size_oz'].isna() & fruit['size_each'].isna()), ['description', 'size_a', 'size']])

# column creation
fruit['regular_per_size_oz'] = fruit['regular']/fruit['size_oz']
fruit['promo_per_size_oz'] = fruit['promo']/fruit['size_oz']
fruit['pct_change_regular_to_promo_size_oz']=((fruit.promo_per_size_oz - fruit.regular_per_size_oz)/fruit.regular_per_size_oz)*100

fruit['regular_per_size_each'] = fruit['regular']/fruit['size_each']
fruit['promo_per_size_each'] = fruit['promo']/fruit['size_each']
fruit['pct_change_regular_to_promo_size_each']=((fruit.promo_per_size_each - fruit.regular_per_size_each)/fruit.regular_per_size_each)*100

# size_oz price
fruit_size_oz = pd.concat([
    fruit[['description','size','regular', 'promo', 'regular_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'regular_per_size_oz':'per_size_oz'}),
    fruit.loc[fruit.promo_per_size_oz>0,['description','size','regular', 'promo', 'promo_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'promo_per_size_oz':'per_size_oz'})
    ]).sort_values(by=['per_size_oz']).drop_duplicates(subset='description', keep='first')
fruit_size_oz['per_size_rank'] = fruit_size_oz.groupby('per_size_oz')['per_size_oz'].transform('mean').rank(method='dense',ascending=True)

# size_each price
fruit_size_each = pd.concat([
    fruit[['description','size','regular', 'promo', 'regular_per_size_each', 'pct_change_regular_to_promo_size_each']].dropna().rename(columns={'regular_per_size_each':'per_size_each'}),
    fruit.loc[fruit.promo_per_size_each>0,['description','size','regular', 'promo', 'promo_per_size_each', 'pct_change_regular_to_promo_size_each']].dropna().rename(columns={'promo_per_size_each':'per_size_each'})
    ]).sort_values(by=['per_size_each']).drop_duplicates(subset='description', keep='first')
fruit_size_each['per_size_rank'] = fruit_size_each.groupby('per_size_each')['per_size_each'].transform('mean').rank(method='dense',ascending=True)

# obtain access token using function
access_token = obtain_access_token()

# salad dressing
filter_term = 'salad dressing'
url = api_url + '/products'
headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
}

params = {'filter.locationId': edgewater_location_id,
          'filter.fulfillment': 'csp',
          'filter.term': filter_term,
          'filter.limit': 50}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
sa = pd.DataFrame(json.loads(response_three.text)['data'])

sal = pd.DataFrame()
for s in range(50,250,50):
    print(s)
    params = {'filter.locationId': edgewater_location_id,
              'filter.fulfillment': 'csp',
              'filter.term': filter_term, #apples #kale #spinach,
              'filter.limit': 50,
              'filter.start': s}
    response_three = requests.get(url, headers=headers, params=params, verify=False)
    print(response_three.status_code)
    s_ = pd.DataFrame(json.loads(response_three.text)['data'])
    sal = pd.concat([sal, s_], axis=0)

salad_dressing = pd.concat([sa, sal], axis=0)
salad_dressing = salad_dressing.drop(columns=['productId', 'upc', 'images', 'itemInformation', 'temperature'])
salad_dressing = pd.concat([salad_dressing.drop(['items'], axis=1), salad_dressing['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
salad_dressing = pd.concat([salad_dressing.drop(['price'], axis=1), salad_dressing['price'].apply(pd.Series)], axis=1)

# create size_a column
salad_dressing['size_a'] = salad_dressing['size'].apply(lambda x: x.split())

# clean up misc. sizes - check these on kroger site ###
# size gives ct (number of tea bags) and oz (weight of package) - only need ct
salad_dressing.loc[ ( salad_dressing['size'].str.contains('ct')) & ( salad_dressing['size'].str.contains('oz')), 'size' ] = \
    salad_dressing.loc[ ( salad_dressing['size'].str.contains('ct')) & ( salad_dressing['size'].str.contains('oz')), 'size' ].apply(lambda x : x[:x.find('ct')+len('ct')])

salad_dressing.loc[ ( salad_dressing['size'].str.contains('ct')) & ( salad_dressing['size'].str.contains('oz')), 'size' ].apply(lambda x : x.split()).apply(lambda x : x[0] ).astype(float) * \
salad_dressing.loc[ ( salad_dressing['size'].str.contains('ct')) & ( salad_dressing['size'].str.contains('oz')), 'size' ].apply(lambda x : x.split()).apply(lambda x : x[3] ).astype(float)
