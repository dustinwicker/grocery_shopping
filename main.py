import os
import json
import base64
import config
import requests
import json
import pandas as pd
import numpy as np

# Increase maximum width in characters of columns - will put all columns in same line in console readout
pd.set_option('expand_frame_repr', False)
# Increase number of rows printed out in console
pd.set_option('display.max_rows', 250)
# Able to read entire value in each column (no longer truncating values)
pd.set_option('display.max_colwidth', None)

# Load in json
with open('info.json', 'r') as i:
    info = json.loads(i.read())

client_id = info['kroger_client_id']
client_secret = info['kroger_client_secret']
# Authentication requires base64 encoded id:secret, which is precalculated here
encoded_client_token = base64.b64encode(f"{client_id}:{client_secret}".encode('ascii')).decode('ascii')

# Measurement conversions
oz_in_lb, oz_in_qt, oz_in_l, oz_in_gal = 16, 32, 33.81, 128
# filter parameters (csp indicates pick up availability)
filter_fulfillment, filter_limit = 'csp', 50

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

# Define products url and necessary headers info to search products
def product_search():
    url = api_url + '/products'
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    return url, headers

# obtain access token using function
access_token = obtain_access_token()

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

# obtain access token using function
access_token = obtain_access_token()
url, headers = product_search()

# Search products at the Edgewater King Soopers
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
filter_term = 'decaf coffee'
params = {
        'filter.locationId': edgewater_location_id,
        'filter.fulfillment': 'csp',
        'filter.term': filter_term,
        'filter.limit': 50
}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
decaf_coffee = pd.DataFrame(json.loads(response_three.text)['data'])
decaf_coffee = decaf_coffee.drop(columns=['productId', 'upc', 'images', 'itemInformation', 'temperature'])
decaf_coffee = pd.concat([decaf_coffee.drop(['items'], axis=1), decaf_coffee['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
decaf_coffee = pd.concat([decaf_coffee.drop(['price'], axis=1), decaf_coffee['price'].apply(pd.Series)], axis=1)
print(decaf_coffee.shape)
# Remove coffee pods from options
decaf_coffee = decaf_coffee[~(decaf_coffee['description'].str.findall(r'Coffe{1,2}.*Pods').map(lambda d: len(d)) > 0)]
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

# coffee creamer
filter_term = 'non dairy coffee creamer'
params = {'filter.locationId': edgewater_location_id, 'filter.fulfillment': filter_fulfillment,
          'filter.term': filter_term, 'filter.limit': filter_limit}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
meta = pd.DataFrame(json.loads(response_three.text)['meta'])
if meta.loc['total'].values[0] > filter_limit:
    c = pd.DataFrame(json.loads(response_three.text)['data'])
    cc = pd.DataFrame()
    filter_start = 50
    for s in range(filter_start,meta.loc['total'].values[0],filter_start):
        params.update({'filter.start': s})
        print(s)
        response_three = requests.get(url, headers=headers, params=params, verify=False)
        print(response_three.status_code)
        c_ = pd.DataFrame(json.loads(response_three.text)['data'])
        cc = pd.concat([cc, c_], axis=0)
coffee_creamer = pd.concat([c, cc],axis=0)
coffee_creamer = coffee_creamer.drop(columns=['productId', 'upc', 'images', 'itemInformation', 'temperature'])
coffee_creamer = pd.concat([coffee_creamer.drop(['items'], axis=1), coffee_creamer['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
coffee_creamer = pd.concat([coffee_creamer.drop(['price'], axis=1), coffee_creamer['price'].apply(pd.Series)], axis=1)

# clean up misc. sizes
coffee_creamer.loc[coffee_creamer['size'] == '1/2 gal', 'size'] = '0.5 gal'

# create size_a column
coffee_creamer['size_a'] = coffee_creamer['size'].apply(lambda x: x.split())
# create size_oz column (can compare oz, lb, fl oz, qt, l)
coffee_creamer['size_oz'] = np.nan
# oz, oz., fl oz, fl oz., lb, lb.
coffee_creamer['size_oz'] = coffee_creamer['size_a'].apply(
    lambda x: float(x[0]) if ( (x[-1] == 'oz' or x[-1] == 'oz.') and (len(x) == 2 or len(x) == 3) )
    else ( float(x[0])*oz_in_qt if ( (x[-1] == 'qt' or x[-1] == 'qt.') and (len(x) == 2) )
    else ( float(x[0])*oz_in_l if ( (x[-1] == 'l' or x[-1] == 'l.') and (len(x) == 2) )
    else ( float(x[0])*oz_in_gal if ( (x[-1] == 'gal' or x[-1] == 'gal.') and (len(x) == 2) )
    else np.nan ) ) ) )

coffee_creamer['regular_per_size_oz'] = coffee_creamer['regular']/coffee_creamer['size_oz']
coffee_creamer['promo_per_size_oz'] = coffee_creamer['promo']/coffee_creamer['size_oz']
coffee_creamer['pct_change_regular_to_promo_size_oz']=((coffee_creamer.promo_per_size_oz - coffee_creamer.regular_per_size_oz)/coffee_creamer.regular_per_size_oz)*100

# check to see if any products remain that need sizing information
print(coffee_creamer.loc[ coffee_creamer['size_oz'].isna(), ['description', 'size']])

coffee_creamer_size_oz = pd.concat([
    coffee_creamer[['description','size','regular', 'promo', 'regular_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'regular_per_size_oz':'per_size_oz'}),
    coffee_creamer.loc[coffee_creamer.promo_per_size_oz>0,['description','size','regular', 'promo', 'promo_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'promo_per_size_oz':'per_size_oz'})
    ]).sort_values(by=['per_size_oz']).drop_duplicates(subset='description', keep='first')
coffee_creamer_size_oz['per_size_rank'] = coffee_creamer_size_oz.groupby('per_size_oz')['per_size_oz'].transform('mean').rank(method='dense',ascending=True)
print(coffee_creamer_size_oz)

# non dairy milk (can be used coffee creamer, stand alone beverage, milkshade ingredient
filter_term = 'non dairy milk'
params = {'filter.locationId': edgewater_location_id, 'filter.fulfillment': filter_fulfillment,
          'filter.term': filter_term, 'filter.limit': filter_limit}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
meta = pd.DataFrame(json.loads(response_three.text)['meta'])
if meta.loc['total'].values[0] > filter_limit:
    m = pd.DataFrame(json.loads(response_three.text)['data'])
    mm = pd.DataFrame()
    filter_start = 50
    for s in range(filter_start,meta.loc['total'].values[0],filter_start):
        params.update({'filter.start': s})
        print(s)
        response_three = requests.get(url, headers=headers, params=params, verify=False)
        print(response_three.status_code)
        m_ = pd.DataFrame(json.loads(response_three.text)['data'])
        mm = pd.concat([mm, m_], axis=0)
non_dairy_milk = pd.concat([m, mm],axis=0)
non_dairy_milk = non_dairy_milk.drop(columns=['productId', 'upc', 'images', 'itemInformation', 'temperature'])
non_dairy_milk = pd.concat([non_dairy_milk.drop(['items'], axis=1), non_dairy_milk['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
non_dairy_milk = pd.concat([non_dairy_milk.drop(['price'], axis=1), non_dairy_milk['price'].apply(pd.Series)], axis=1)

# clean up misc. sizes
non_dairy_milk.loc[non_dairy_milk['size'] == '1/2 gal', 'size'] = '0.5 gal'
non_dairy_milk.loc[non_dairy_milk['size'] == '6 ct / 8 fl oz', 'size'] = str(6*8) + ' fl oz'

# create size_a column
non_dairy_milk['size_a'] = non_dairy_milk['size'].apply(lambda x: x.split())
# create size_oz column (can compare oz, lb, fl oz, qt, l)
non_dairy_milk['size_oz'] = np.nan
# oz, oz., fl oz, fl oz., lb, lb.
non_dairy_milk['size_oz'] = non_dairy_milk['size_a'].apply(
    lambda x: float(x[0]) if ( (x[-1] == 'oz' or x[-1] == 'oz.') and (len(x) == 2 or len(x) == 3) )
    else ( float(x[0])*oz_in_qt if ( (x[-1] == 'qt' or x[-1] == 'qt.') and (len(x) == 2) )
    else ( float(x[0])*oz_in_l if ( (x[-1] == 'l' or x[-1] == 'l.') and (len(x) == 2) )
    else ( float(x[0])*oz_in_gal if ( (x[-1] == 'gal' or x[-1] == 'gal.') and (len(x) == 2) )
    else np.nan ) ) ) )

non_dairy_milk['regular_per_size_oz'] = non_dairy_milk['regular']/non_dairy_milk['size_oz']
non_dairy_milk['promo_per_size_oz'] = non_dairy_milk['promo']/non_dairy_milk['size_oz']
non_dairy_milk['pct_change_regular_to_promo_size_oz']=((non_dairy_milk.promo_per_size_oz - non_dairy_milk.regular_per_size_oz)/non_dairy_milk.regular_per_size_oz)*100

# check to see if any products remain that need sizing information
print(non_dairy_milk.loc[ non_dairy_milk['size_oz'].isna(), ['description', 'size']])

non_dairy_milk_size_oz = pd.concat([
    non_dairy_milk[['description','size','regular', 'promo', 'regular_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'regular_per_size_oz':'per_size_oz'}),
    non_dairy_milk.loc[non_dairy_milk.promo_per_size_oz>0,['description','size','regular', 'promo', 'promo_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'promo_per_size_oz':'per_size_oz'})
    ]).sort_values(by=['per_size_oz']).drop_duplicates(subset='description', keep='first')
non_dairy_milk_size_oz['per_size_rank'] = non_dairy_milk_size_oz.groupby('per_size_oz')['per_size_oz'].transform('mean').rank(method='dense',ascending=True)
print(non_dairy_milk_size_oz)

# vegetables
filter_term = 'fresh vegetables'
params = {'filter.locationId': edgewater_location_id, 'filter.fulfillment': filter_fulfillment,
          'filter.term': filter_term, 'filter.limit': filter_limit}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
meta = pd.DataFrame(json.loads(response_three.text)['meta'])
if meta.loc['total'].values[0] > filter_limit:
    v = pd.DataFrame(json.loads(response_three.text)['data'])
    vv = pd.DataFrame()
    filter_start = 50
    for s in range(filter_start,meta.loc['total'].values[0],filter_start):
        try:
            params.update({'filter.start': s})
            print(params, s)
            response_three = requests.get(url, headers=headers, params=params, verify=False)
            print(response_three.status_code)
            v_ = pd.DataFrame(json.loads(response_three.text)['data'])
            vv = pd.concat([vv, v_], axis=0)
        except KeyError:
            pass
v = pd.concat([v, vv],axis=0)

v = v.drop(columns=['productId', 'upc', 'images', 'itemInformation', 'temperature'])
v = pd.concat([v.drop(['items'], axis=1), v['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
v = pd.concat([v.drop(['price'], axis=1), v['price'].apply(pd.Series)], axis=1)
print(v['size'].value_counts())
# clean up misc. sizes
v.loc[v['size'] == 'each', 'size'] = '1 each'
v.loc[v['size'] == '1 pt / 10 oz', 'size'] = '10 oz'
v.loc[v['size'] == '4 ct / 3 oz', 'size'] = '12 oz'
v.loc[v['size'] == '4 ct / 10.5 oz', 'size'] = '42 oz'
v.loc[v['size'] == '4 ct / 15.25 oz', 'size'] = '61 oz'

# create size_a column
v['size_a'] = v['size'].apply(lambda x: x.split())

# create size_oz column (can compare oz, lb, fl oz)
v['size_oz'] = np.nan
# oz, oz., fl oz, fl oz.
# lb, lb., lbs
v['size_oz'] = v['size_a'].apply(
    lambda x: float(x[0]) if ( (x[-1] == 'oz' or x[-1] == 'oz.') and (len(x) == 2 or len(x) == 3) )
    else ( float(x[0])*oz_in_lb if ( (x[-1] == 'lb' or x[-1] == 'lb.' or x[-1] == 'lbs') and (len(x) == 2) )
    else np.nan ) )

# create size_each column (bunch, ct, each)
v['size_each'] = np.nan
v['size_each'] = v['size_a'].apply(
    lambda x: float(x[0]) if (x[-1] == 'ct' or x[-1] == 'each' or x[-1] == 'bunch' ) and (len(x) == 2)
    else np.nan )

# check to see if any products remain that need sizing information
print(v.loc[ (v['size_oz'].isna() & v['size_each'].isna()), ['description', 'size_a', 'size']])

# column creation
v['regular_per_size_oz'] = v['regular']/v['size_oz']
v['promo_per_size_oz'] = v['promo']/v['size_oz']
v['pct_change_regular_to_promo_size_oz']=((v.promo_per_size_oz - v.regular_per_size_oz)/v.regular_per_size_oz)*100

v['regular_per_size_each'] = v['regular']/v['size_each']
v['promo_per_size_each'] = v['promo']/v['size_each']
v['pct_change_regular_to_promo_size_each']=((v.promo_per_size_each - v.regular_per_size_each)/v.regular_per_size_each)*100

# size_oz price
veg_size_oz = pd.concat([
    v[['description','size','regular', 'promo', 'regular_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'regular_per_size_oz':'per_size_oz'}),
    v.loc[v.promo_per_size_oz>0,['description','size','regular', 'promo', 'promo_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'promo_per_size_oz':'per_size_oz'})
    ]).sort_values(by=['per_size_oz']).drop_duplicates(subset='description', keep='first')
veg_size_oz['per_size_rank'] = veg_size_oz.groupby('per_size_oz')['per_size_oz'].transform('mean').rank(method='dense',ascending=True)

# size_each price
veg_size_each = pd.concat([
    v[['description','size','regular', 'promo', 'regular_per_size_each', 'pct_change_regular_to_promo_size_each']].dropna().rename(columns={'regular_per_size_each':'per_size_each'}),
    v.loc[v.promo_per_size_each>0,['description','size','regular', 'promo', 'promo_per_size_each', 'pct_change_regular_to_promo_size_each']].dropna().rename(columns={'promo_per_size_each':'per_size_each'})
    ]).sort_values(by=['per_size_each']).drop_duplicates(subset='description', keep='first')
veg_size_each['per_size_rank'] = veg_size_each.groupby('per_size_each')['per_size_each'].transform('mean').rank(method='dense',ascending=True)

print(veg_size_oz)
print(veg_size_each)

# decaf and herbal tea
filter_term = 'decaf tea'
# decaf
params = {'filter.locationId': edgewater_location_id,
          'filter.fulfillment': 'csp',
          'filter.term': filter_term,
          'filter.limit': 50}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
decaf_tea = pd.DataFrame(json.loads(response_three.text)['data'])
# herbal
filter_term = 'herbal tea'
params = {'filter.locationId': edgewater_location_id,
          'filter.fulfillment': 'csp',
          'filter.term': filter_term,
          'filter.limit': 50}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
h = pd.DataFrame(json.loads(response_three.text)['data'])

he = pd.DataFrame()
for s in [50, 100, 150]:
    print(s)
    params = {'filter.locationId': edgewater_location_id,
              'filter.fulfillment': 'csp',
              'filter.term': filter_term,
              'filter.limit': 50,
              'filter.start': s}
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

print(tea_size_oz)
print(tea_size_ct)

# fruit
filter_term = 'fruit'
params = {'filter.locationId': edgewater_location_id, 'filter.fulfillment': filter_fulfillment,
          'filter.term': filter_term, 'filter.limit': filter_limit}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
meta = pd.DataFrame(json.loads(response_three.text)['meta'])
if meta.loc['total'].values[0] > filter_limit:
    f = pd.DataFrame(json.loads(response_three.text)['data'])
    ff = pd.DataFrame()
    filter_start = 50
    for s in range(filter_start,meta.loc['total'].values[0],filter_start):
        try:
            params.update({'filter.start': s})
            print(params, s)
            response_three = requests.get(url, headers=headers, params=params, verify=False)
            print(response_three.status_code)
            f_ = pd.DataFrame(json.loads(response_three.text)['data'])
            ff = pd.concat([ff, f_], axis=0)
        except KeyError:
            pass
fruit = pd.concat([f, ff],axis=0)
print(fruit.shape)
fruit = fruit.drop(columns=['productId', 'upc', 'images', 'itemInformation', 'temperature'])
fruit = pd.concat([fruit.drop(['items'], axis=1), fruit['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
fruit = pd.concat([fruit.drop(['price'], axis=1), fruit['price'].apply(pd.Series)], axis=1)

print(fruit['size'].value_counts())
# clean up misc. sizes
fruit.loc[fruit['size'] == '1 pt / 10 oz', 'size'] = '10 oz'
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

print(fruit_size_oz)
print(fruit_size_each)

# salad dressing
filter_term = 'salad dressing'
params = {'filter.locationId': edgewater_location_id, 'filter.fulfillment': filter_fulfillment,
          'filter.term': filter_term, 'filter.limit': filter_limit}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
meta = pd.DataFrame(json.loads(response_three.text)['meta'])
if meta.loc['total'].values[0] > filter_limit:
    sa = pd.DataFrame(json.loads(response_three.text)['data'])
    ss = pd.DataFrame()
    filter_start = 50
    for s in range(filter_start,meta.loc['total'].values[0],filter_start):
        try:
            params.update({'filter.start': s})
            print(params, s)
            response_three = requests.get(url, headers=headers, params=params, verify=False)
            print(response_three.status_code)
            s_ = pd.DataFrame(json.loads(response_three.text)['data'])
            ss = pd.concat([ss, s_], axis=0)
        except KeyError:
            pass
salad_dressing = pd.concat([sa, ss], axis=0)
salad_dressing = salad_dressing.drop(columns=['productId', 'upc', 'images', 'itemInformation', 'temperature'])
salad_dressing = pd.concat([salad_dressing.drop(['items'], axis=1), salad_dressing['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
salad_dressing = pd.concat([salad_dressing.drop(['price'], axis=1), salad_dressing['price'].apply(pd.Series)], axis=1)
print(salad_dressing.shape)
print(salad_dressing['size'].value_counts())
# clean up misc. sizes - check these on kroger site ###
# size gives ct (number of tea bags) and oz (weight of package) - only need ct
salad_dressing.loc[ ( salad_dressing['size'].str.contains('ct')) & ( salad_dressing['size'].str.contains('oz')), 'size' ] = \
    ( salad_dressing.loc[ ( salad_dressing['size'].str.contains('ct')) & ( salad_dressing['size'].str.contains('oz')), 'size' ].apply(lambda x : x.split()).apply(lambda x : x[0] ).astype(float) * \
    salad_dressing.loc[ ( salad_dressing['size'].str.contains('ct')) & ( salad_dressing['size'].str.contains('oz')), 'size' ].apply(lambda x : x.split()).apply(lambda x : x[3] ).astype(float) ).astype(str) + \
    ' ' + salad_dressing.loc[ ( salad_dressing['size'].str.contains('ct')) & ( salad_dressing['size'].str.contains('oz')), 'size' ].apply(lambda x : x.split()).apply(lambda x : x[-1] )
# create size_a column
salad_dressing['size_a'] = salad_dressing['size'].apply(lambda x: x.split())

# create size_oz column (can compare oz, lb, fl oz)
salad_dressing['size_oz'] = np.nan
# oz, oz., fl oz, fl oz.
salad_dressing['size_oz'] = salad_dressing['size_a'].apply(lambda x: float(x[0]) if ( (x[-1] == 'oz' or x[-1] == 'oz.') and ( len(x) == 2 or len(x) == 3 ) ) else np.nan )

# create size_each column (ct)
salad_dressing['size_each'] = np.nan
salad_dressing['size_each'] = salad_dressing['size_a'].apply(lambda x: float(x[0]) if (x[-1] == 'ct' ) and (len(x) == 2)
                                     else ( float(x[0][:-2]) if x[0][-2:] == 'ct' else np.nan) )

# check to see if any products remain that need sizing information
print(salad_dressing.loc[ (salad_dressing['size_oz'].isna() & salad_dressing['size_each'].isna()), ['description', 'size_a', 'size']])

# column creation
salad_dressing['regular_per_size_oz'] = salad_dressing['regular']/salad_dressing['size_oz']
salad_dressing['promo_per_size_oz'] = salad_dressing['promo']/salad_dressing['size_oz']
salad_dressing['pct_change_regular_to_promo_size_oz']= ((salad_dressing.promo_per_size_oz - salad_dressing.regular_per_size_oz)/salad_dressing.regular_per_size_oz)*100

salad_dressing['regular_per_size_each'] = salad_dressing['regular']/salad_dressing['size_each']
salad_dressing['promo_per_size_each'] = salad_dressing['promo']/salad_dressing['size_each']
salad_dressing['pct_change_regular_to_promo_size_each']= ((salad_dressing.promo_per_size_each - salad_dressing.regular_per_size_each)/salad_dressing.regular_per_size_each)*100

# size_oz price
salad_dressing_size_oz = pd.concat([
    salad_dressing[['description','size','regular', 'promo', 'regular_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'regular_per_size_oz':'per_size_oz'}),
    salad_dressing.loc[salad_dressing.promo_per_size_oz>0,['description','size','regular', 'promo', 'promo_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'promo_per_size_oz':'per_size_oz'})
    ]).sort_values(by=['per_size_oz']).drop_duplicates(subset='description', keep='first')
salad_dressing_size_oz['per_size_rank'] = salad_dressing_size_oz.groupby('per_size_oz')['per_size_oz'].transform('mean').rank(method='dense',ascending=True)

# size_each price
salad_dressing_size_each = pd.concat([
    salad_dressing[['description','size','regular', 'promo', 'regular_per_size_each', 'pct_change_regular_to_promo_size_each']].dropna().rename(columns={'regular_per_size_each':'per_size_each'}),
    salad_dressing.loc[salad_dressing.promo_per_size_each>0,['description','size','regular', 'promo', 'promo_per_size_each', 'pct_change_regular_to_promo_size_each']].dropna().rename(columns={'promo_per_size_each':'per_size_each'})
    ]).sort_values(by=['per_size_each']).drop_duplicates(subset='description', keep='first')
salad_dressing_size_each['per_size_rank'] = salad_dressing_size_each.groupby('per_size_each')['per_size_each'].transform('mean').rank(method='dense',ascending=True)

print(salad_dressing_size_oz)
print(salad_dressing_size_each)

# eggs
filter_term = 'eggs'
params = {'filter.locationId': edgewater_location_id, 'filter.fulfillment': filter_fulfillment,
          'filter.term': filter_term, 'filter.limit': filter_limit}
response_three = requests.get(url, headers=headers, params=params, verify=False)
print(response_three.status_code)
meta = pd.DataFrame(json.loads(response_three.text)['meta'])
if meta.loc['total'].values[0] > filter_limit:
    e = pd.DataFrame(json.loads(response_three.text)['data'])
    ee = pd.DataFrame()
    filter_start = 50
    for s in range(filter_start,meta.loc['total'].values[0],filter_start):
        try:
            params.update({'filter.start': s})
            print(params, s)
            response_three = requests.get(url, headers=headers, params=params, verify=False)
            print(response_three.status_code)
            e_ = pd.DataFrame(json.loads(response_three.text)['data'])
            ee = pd.concat([ss, s_], axis=0)
        except KeyError:
            pass
eggs = pd.concat([e, ee], axis=0)
# limit results to only those with 'egg' in the description
eggs = eggs.loc[eggs.description.str.contains('egg',case=False)]
eggs = eggs.drop(columns=['productId', 'upc', 'images', 'itemInformation', 'temperature'])
eggs = pd.concat([eggs.drop(['items'], axis=1), eggs['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
eggs = pd.concat([eggs.drop(['price'], axis=1), eggs['price'].apply(pd.Series)], axis=1)

# Misc sizes - need to fix all
eggs.loc[eggs['size'] == 'large dozen', 'size'] = '12 ct'
# only look at ct for now - will need to fix (liquid eggs for this weekend possibly?)
eggs = eggs.loc[eggs['size'].str[-2:]== 'ct']

# create size_a column
eggs['size_a'] = eggs['size'].apply(lambda x: x.split())

# create size_oz column (can compare oz, lb, fl oz)
#eggs['size_oz'] = np.nan
# oz, oz., fl oz, fl oz.
#eggs['size_oz'] = eggs['size_a'].apply(lambda x: float(x[0]) if ( (x[-1] == 'oz' or x[-1] == 'oz.') and ( len(x) == 2 or len(x) == 3 ) ) else np.nan )

# create size_each column (ct)
eggs['size_each'] = np.nan
eggs['size_each'] = eggs['size_a'].apply(lambda x: float(x[0]) if (x[-1] == 'ct' ) and (len(x) == 2)
                                     else ( float(x[0][:-2]) if x[0][-2:] == 'ct' else np.nan) )

# check to see if any products remain that need sizing information
print(salad_dressing.loc[ (salad_dressing['size_oz'].isna() & salad_dressing['size_each'].isna()), ['description', 'size_a', 'size']])

# column creation
#salad_dressing['regular_per_size_oz'] = salad_dressing['regular']/salad_dressing['size_oz']
#salad_dressing['promo_per_size_oz'] = salad_dressing['promo']/salad_dressing['size_oz']
#salad_dressing['pct_change_regular_to_promo_size_oz']= ((salad_dressing.promo_per_size_oz - salad_dressing.regular_per_size_oz)/salad_dressing.regular_per_size_oz)*100

eggs['regular_per_size_each'] = eggs['regular']/eggs['size_each']
eggs['promo_per_size_each'] = eggs['promo']/eggs['size_each']
eggs['pct_change_regular_to_promo_size_each']= ((eggs.promo_per_size_each - eggs.regular_per_size_each)/eggs.regular_per_size_each)*100

# size_oz price
# salad_dressing_size_oz = pd.concat([
#     salad_dressing[['description','size','regular', 'promo', 'regular_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'regular_per_size_oz':'per_size_oz'}),
#     salad_dressing.loc[salad_dressing.promo_per_size_oz>0,['description','size','regular', 'promo', 'promo_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'promo_per_size_oz':'per_size_oz'})
#     ]).sort_values(by=['per_size_oz']).drop_duplicates(subset='description', keep='first')
# salad_dressing_size_oz['per_size_rank'] = salad_dressing_size_oz.groupby('per_size_oz')['per_size_oz'].transform('mean').rank(method='dense',ascending=True)

# size_each price
eggs_size_each = pd.concat([
    eggs[['description','size','regular', 'promo', 'regular_per_size_each', 'pct_change_regular_to_promo_size_each']].dropna().rename(columns={'regular_per_size_each':'per_size_each'}),
    eggs.loc[eggs.promo_per_size_each>0,['description','size','regular', 'promo', 'promo_per_size_each', 'pct_change_regular_to_promo_size_each']].dropna().rename(columns={'promo_per_size_each':'per_size_each'})
    ]).sort_values(by=['per_size_each']).drop_duplicates(subset='description', keep='first')
eggs_size_each['per_size_rank'] = eggs_size_each.groupby('per_size_each')['per_size_each'].transform('mean').rank(method='dense',ascending=True)

#print(eggs_size_oz)
print(eggs_size_each)