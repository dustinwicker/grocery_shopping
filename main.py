import os
import base64
import requests
import json
import pandas as pd
import numpy as np
import datetime as dt
import pytz
# import seaborn as sns

# Increase maximum width in characters of columns - will put all columns in same line in console readout
pd.set_option('expand_frame_repr', False)
# Increase number of rows printed out in console
pd.set_option('display.max_rows', 250)
# Able to read entire value in each column (no longer truncating values)
pd.set_option('display.max_colwidth', None)

# Current directory
os.chdir("C:/Users/dustin.wicker/PycharmProjects/grocery_shopping")
# Load in json
with open('info.json', 'r') as i:
    info = json.loads(i.read())
client_id = info['kroger_client_id']
client_secret = info['kroger_client_secret']
# Authentication requires base64 encoded id:secret, which is precalculated here
encoded_client_token = base64.b64encode(f"{client_id}:{client_secret}".encode('ascii')).decode('ascii')
# Measurement conversions
oz_in_lb, oz_in_qt, oz_in_l, oz_in_gal = 16, 32, 33.81, 128
# oz, lb, other variation finder for size column
oz_finder = 'oz'  # oz, oz., fl oz, fl oz.
lb_finder = 'lb'  # lb, lb., lbs
each_finder = ['ct', 'bunch', 'each']
# Mountain timezone
pytz_mtn = pytz.timezone('US/Mountain')
api_url = 'https://api.kroger.com/v1'
filter_fulfillment, filter_limit, filter_start = 'csp', 50, 50


def obtain_access_token():
    """access token"""
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


def location(zipcode, address, city):
    """location information to search store for products"""
    url = api_url + '/locations'
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    params = {
        'filter.zipCode.near': f'{zipcode}'
    }
    response = requests.get(url, headers=headers, params=params, verify=False)
    print(response.status_code)

    # Create DataFrame
    location_df = pd.DataFrame(json.loads(response.text)['data'])[['locationId', 'chain', 'address']]
    # Drop address column, split address column (dict) into separate columns
    location_df = pd.concat([location_df.drop(['address'], axis=1), location_df['address'].apply(pd.Series)], axis=1)
    location_id = location_df.loc[(location_df.addressLine1.str.contains(address)) & (location_df.city == city), 'locationId'].values[0]
    return location_id


def product_search(filter_term):
    """search term and return info in form of DataFrame"""
    # products url and necessary headers info to search products
    url = api_url + '/products'
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}' }
    # filter parameters (csp indicates pick up availability)
    params = {'filter.locationId': edgewater_location_id, 'filter.fulfillment': f'{filter_fulfillment}',
          'filter.term': f'{filter_term}', 'filter.limit': filter_limit}
    response = requests.get(url, headers=headers, params=params, verify=False)
    print(response.status_code)
    meta = pd.DataFrame(json.loads(response.text)['meta'])
    if meta.loc['total'].values[0] > filter_limit:
        a = pd.DataFrame(json.loads(response.text)['data'])
        aa = pd.DataFrame()
        f_s = filter_start
        for s in range(f_s, meta.loc['total'].values[0], f_s):
            try:
                params.update({'filter.start': s})
                print(params, s)
                response = requests.get(url, headers=headers, params=params, verify=False)
                print(response.status_code)
                a_ = pd.DataFrame(json.loads(response.text)['data'])
                aa = pd.concat([aa, a_], axis=0)
            except KeyError:
                pass
        df = pd.concat([a, aa],axis=0)
    else:
        df = pd.DataFrame(json.loads(response.text)['data'])
    df = df.drop(columns=['productId', 'upc', 'images', 'itemInformation', 'temperature'])
    df = pd.concat([df.drop(['items'], axis=1), df['items'].apply(lambda x: x[0]).apply(pd.Series)], axis=1)
    df = pd.concat([df.drop(['price'], axis=1), df['price'].apply(pd.Series)], axis=1)
    print(df.shape)
    return df


def column_creation(df):
    df['regular_per_size_oz'] = df['regular']/df['size_oz']
    df['promo_per_size_oz'] = df['promo']/df['size_oz']
    df['pct_change_regular_to_promo_size_oz']=((df.promo_per_size_oz - df.regular_per_size_oz)/df.regular_per_size_oz)*100
    df['regular_per_size_each'] = df['regular']/df['size_each']
    df['promo_per_size_each'] = df['promo']/df['size_each']
    df['pct_change_regular_to_promo_size_each']=((df.promo_per_size_each - df.regular_per_size_each)/df.regular_per_size_each)*100


# obtain access token using function
access_token = obtain_access_token()
edgewater_location_id = location(zipcode=80204, address='1725 Sheridan', city='Edgewater')
# vegetables
# want to visual fruit and veggies and have available via mobile (google sheets with tabs for each completed dataframe?)
veg = product_search(filter_term='fresh vegetables')
# clean up misc. sizes (add description or upc to make more exact? or could make too specific?) - those with "/", 'each'
print(veg.loc[veg['size'].str.contains('/'), 'size'])
print(veg.loc[veg['size'] =='each', 'size'])
veg.loc[veg['size'] == 'each', 'size'] = '1 each'
veg.loc[veg['size'] == '1 pt / 10 oz', 'size'] = '10 oz'
veg.loc[veg['size'] == '4 ct / 3 oz', 'size'] = '12 oz'
veg.loc[veg['size'] == '4 ct / 10.5 oz', 'size'] = '42 oz'
veg.loc[veg['size'] == '4 ct / 15.25 oz', 'size'] = '61 oz'
print(veg['size'].value_counts())
# create size_ column
veg['size_'] = veg['size'].apply(lambda x: x.split(" ", 1))
print(veg['size'].apply(lambda x: x.split(" ", 1)[1]).value_counts())

# create size_oz column (can compare oz, lb, fl oz)
veg['size_oz'] = veg['size_'].apply(lambda x: float(x[0]) if oz_finder in x[-1] else (float(x[0])*oz_in_lb if lb_finder in x[-1]
                                                                                 else np.nan))
# create size_each column (bunch, ct, each) #pk? 5/14/2023
veg['size_each'] = veg['size_'].apply(lambda x: float(x[0]) if any([q for q in each_finder if q in x[-1]]) else np.nan)

# check to see if any products remain that need sizing information
print(veg.loc[(veg['size_oz'].isna() & veg['size_each'].isna()), ['description', 'size', 'size_']])
# drop for now 5/14/2023
veg = veg.loc[~(veg['size_oz'].isna() & veg['size_each'].isna())]

column_creation(df=veg)
# size_oz price
veg_size_oz = pd.concat([
    veg[['description','size','regular', 'promo', 'regular_per_size_oz',
         'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'regular_per_size_oz':'per_size_oz'}),
    veg.loc[veg.promo_per_size_oz>0,['description','size','regular', 'promo', 'promo_per_size_oz',
                                     'pct_change_regular_to_promo_size_oz']].dropna().
    rename(columns={'promo_per_size_oz':'per_size_oz'})]).sort_values(by=['per_size_oz']).\
    drop_duplicates(subset='description', keep='first')
# veg_size_oz['per_size_rank'] = veg_size_oz.groupby('per_size_oz')['per_size_oz'].transform('mean').rank(method='dense',ascending=True)
veg_size_oz['runtime_mst'] = dt.datetime.now(pytz_mtn)
# size_each price
veg_size_each = pd.concat([
    veg[['description','size','regular', 'promo', 'regular_per_size_each', 'pct_change_regular_to_promo_size_each']].dropna().rename(columns={'regular_per_size_each':'per_size_each'}),
    veg.loc[veg.promo_per_size_each>0,['description','size','regular', 'promo', 'promo_per_size_each', 'pct_change_regular_to_promo_size_each']].dropna().rename(columns={'promo_per_size_each':'per_size_each'})
    ]).sort_values(by=['per_size_each']).drop_duplicates(subset='description', keep='first')
# veg_size_each['per_size_rank'] = veg_size_each.groupby('per_size_each')['per_size_each'].transform('mean').rank(method='dense',ascending=True)
veg_size_each['runtime_mst'] = dt.datetime.now(pytz_mtn)

# fruit
fruit = product_search(filter_term='fruit')
# clean up misc. sizes
print(fruit.loc[fruit['size'].str.contains('/'), 'size'])
print(fruit.loc[fruit['size'] == 'each', 'size'])
fruit.loc[fruit['size'] == '1 pt / 10 oz', 'size'] = '10 oz'
print(fruit['size'].value_counts())
# create size_ column
fruit['size_'] = fruit['size'].apply(lambda x: x.split(" ", 1))
print(veg['size_'].apply(lambda x : x[1]).value_counts())

# create size_oz column (can compare oz, lb, fl oz)
fruit['size_oz'] = fruit['size_'].apply(lambda x: float(x[0]) if oz_finder in x[-1] else (float(x[0])*oz_in_lb if lb_finder in x[-1]
                                                                                 else np.nan))
# create size_each column (bunch, ct, each) #pk? 5/14/2023
fruit['size_each'] = fruit['size_'].apply(lambda x: float(x[0]) if any([q for q in each_finder if q in x[-1]]) else np.nan )

# check to see if any products remain that need sizing information
print(fruit.loc[(fruit['size_oz'].isna() & fruit['size_each'].isna()), ['description', 'size_', 'size']])

# column creation
column_creation(df=fruit)
# size_oz price
fruit_size_oz = pd.concat([
    fruit[['description', 'size', 'regular', 'promo', 'regular_per_size_oz',
           'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'regular_per_size_oz':'per_size_oz'}),
    fruit.loc[fruit.promo_per_size_oz > 0, ['description','size','regular', 'promo', 'promo_per_size_oz',
                                            'pct_change_regular_to_promo_size_oz']].dropna().
    rename(columns={'promo_per_size_oz':'per_size_oz'})
    ]).sort_values(by=['per_size_oz']).drop_duplicates(subset='description', keep='first')
fruit_size_oz['runtime_mst'] = dt.datetime.now(pytz_mtn)
# size_each price
fruit_size_each = pd.concat([
    fruit[['description', 'size', 'regular', 'promo', 'regular_per_size_each',
           'pct_change_regular_to_promo_size_each']].dropna().rename(columns={'regular_per_size_each':'per_size_each'}),
    fruit.loc[fruit.promo_per_size_each > 0, ['description', 'size', 'regular', 'promo', 'promo_per_size_each',
                                              'pct_change_regular_to_promo_size_each']].dropna().
    rename(columns={'promo_per_size_each':'per_size_each'})]).sort_values(by=['per_size_each']).\
    drop_duplicates(subset='description', keep='first')
fruit_size_each['runtime_mst'] = dt.datetime.now(pytz_mtn)

# could keep upc for drop_duplicates to use as subset
veg_fruit_size_oz = pd.concat([veg_size_oz, fruit_size_oz], axis=0).\
    drop_duplicates(subset=['description', 'size']).reset_index(drop=True).sort_values(by=['per_size_oz'])
veg_fruit_size_each = pd.concat([veg_size_each, fruit_size_each], axis=0).\
    drop_duplicates(subset=['description', 'size']).reset_index(drop=True).sort_values(by=['per_size_each'])

veg_fruit_size_oz['per_size_rank'] = veg_fruit_size_oz.groupby('per_size_oz')['per_size_oz'].transform('mean').\
    rank(method= 'dense', ascending=True)
veg_fruit_size_each['per_size_rank'] = veg_fruit_size_each.groupby('per_size_each')['per_size_each'].transform('mean').\
    rank(method='dense',ascending=True)



# import certifi
# certifi.where()
# veg_fruit_size_oz.per_size_oz.describe()
#
#
# sns.barplot(data=veg_fruit_size_oz, x='description', y='per_size_oz')
# sns.histplot(data=veg_fruit_size_oz, x='per_size_oz', kde=True)

# coffee
coffee = product_search(filter_term='whole bean coffee')
# clean up misc. sizes
print(coffee.loc[coffee['size'].str.contains('/'), 'size'])
print(coffee.loc[coffee['size'] =='each', 'size'])
print(coffee['size'].value_counts())
# create size_ column
coffee['size_'] = coffee['size'].apply(lambda x: x.split(" ", 1))
print(coffee['size_'].apply(lambda x: x[1]).value_counts())```
# create size_oz column (can compare oz, lb, fl oz)
coffee['size_oz'] = coffee['size_'].apply(lambda x: float(x[0]) if oz_finder in x[-1] else (float(x[0])*oz_in_lb if lb_finder in x[-1]
                                                                                 else np.nan))
# check to see if any products remain that need sizing information
print(coffee.loc[coffee['size_oz'].isna(), ['description', 'size_', 'size']])
# column creation
column_creation(df=coffee)
coffee_size_oz = pd.concat([
    coffee[['description','size','regular', 'promo', 'regular_per_size_oz', 'pct_change_regular_to_promo_size_oz']].
    dropna().rename(columns={'regular_per_size_oz':'per_size_oz'}),
    coffee.loc[coffee.promo_per_size_oz>0,['description','size','regular', 'promo', 'promo_per_size_oz',
                                           'pct_change_regular_to_promo_size_oz']].dropna().
    rename(columns={'promo_per_size_oz':'per_size_oz'})]).sort_values(by=['per_size_oz']).\
    drop_duplicates(subset='description', keep='first')
coffee_size_oz['runtime_mst'] = dt.datetime.now(pytz_mtn)
print(coffee_size_oz)

# decaf coffee
decaf_coffee = product_search(filter_term='decaf coffee')
# Remove coffee pods from options
decaf_coffee = decaf_coffee[~(decaf_coffee['description'].str.findall(r'Coffe{1,2}.*Pods').map(lambda d: len(d)) > 0)]
# clean up misc. sizes
print(decaf_coffee.loc[decaf_coffee['size'].str.contains('/'), 'size'])
print(decaf_coffee.loc[decaf_coffee['size'] =='each', 'size'])
print(decaf_coffee['size'].value_counts())
# create size_ column
decaf_coffee['size_'] = decaf_coffee['size'].apply(lambda x: x.split(" ", 1))
print(decaf_coffee['size_'].apply(lambda x: x[1]).value_counts())
# create size_oz column (can compare oz, lb, fl oz)
decaf_coffee['size_oz'] = decaf_coffee['size_'].apply(lambda x: float(x[0]) if oz_finder in x[-1] else (float(x[0])*oz_in_lb if lb_finder in x[-1]
                                                                                 else np.nan))
# check to see if any products remain that need sizing information
print(decaf_coffee.loc[decaf_coffee['size_oz'].isna(), ['description', 'size_', 'size']])
# column creation
column_creation(df=decaf_coffee)
decaf_coffee_size_oz = pd.concat([
    decaf_coffee[['description','size','regular', 'promo', 'regular_per_size_oz', 'pct_change_regular_to_promo_size_oz']].
    dropna().rename(columns={'regular_per_size_oz':'per_size_oz'}),
    decaf_coffee.loc[decaf_coffee.promo_per_size_oz>0,['description','size','regular', 'promo', 'promo_per_size_oz',
                                           'pct_change_regular_to_promo_size_oz']].dropna().
    rename(columns={'promo_per_size_oz':'per_size_oz'})]).sort_values(by=['per_size_oz']).\
    drop_duplicates(subset='description', keep='first')
decaf_coffee_size_oz['runtime_mst'] = dt.datetime.now(pytz_mtn)
print(decaf_coffee_size_oz)

coffee_size_oz_df = pd.concat([coffee_size_oz, decaf_coffee_size_oz], axis=0).\
    drop_duplicates(subset=['description', 'size']).reset_index(drop=True).sort_values(by=['per_size_oz'])

# coffee creamer
coffee_creamer = product_search(filter_term='non dairy coffee creamer')
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
coffee_creamer_size_oz['runtime_mst'] = dt.datetime.now(pytz_mtn)

# non dairy milk (can be used coffee creamer, stand alone beverage, milkshade ingredient
non_dairy_milk = product_search(filter_term='non dairy milk')
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
non_dairy_milk_size_oz['runtime_mst'] = dt.datetime.now(pytz_mtn)

# Create dataframe
milk_coffee_creamer_size_oz_df = pd.concat([
    coffee_creamer_size_oz, non_dairy_milk_size_oz], axis=0).sort_values(by=['per_size_oz'])

# decaf and herbal tea
decaf_tea = product_search(filter_term='decaf tea')
herbal_tea = product_search(filter_term='herbal tea')
tea = pd.concat([decaf_tea, herbal_tea], axis=0)

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

# salad dressing
salad_dressing = product_search(filter_term='salad dressing')
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
print('\n')
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
print(eggs.shape)
print(eggs['size'].value_counts())
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
# fix - print(eggs.loc[ (eggs['size_oz'].isna() & eggs['size_each'].isna()), ['description', 'size_a', 'size']])

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

# peanut butter
peanut_butter = product_search(filter_term='peanut butter')

# Remove products that contain peanut butter (i.e. Reese's Peanut Butter Cups)
peanut_butter = peanut_butter.loc[~(peanut_butter.description.str.contains('Candy|Bar|Cereal|Cookie|Cups|Granola|Cracker|Treats|'
                                                           'Ice Cream|Protein Shake|Dessert|Pretzel|Sandwich|'
                                                           'Chocolate|Bone|Biscuits|Creme Pies|Mix|Baking Chips|'
                                                           'Clusters|Wafers'))]
print(peanut_butter.shape)
print(peanut_butter[['description', 'size']])

# # clean up misc. sizes
peanut_butter.loc[peanut_butter['size'] == '8 ct / 1.15 oz', 'size'] = str(float(8*1.15)) + ' oz'

# Fix size issue - get ones with slashes' index
# str.strip
# Remove dots from last position in each string
# str.split(' ')

# create size_a column
peanut_butter['size_a'] = peanut_butter['size'].apply(lambda x: x.split())

# create size_oz column (can compare oz, lb, fl oz)
peanut_butter['size_oz'] = np.nan
# oz, oz.
peanut_butter['size_oz'] = peanut_butter['size_a'].apply(lambda x: float(x[0]) if ( (x[-1] == 'oz' or x[-1] == 'oz.') and (len(x) == 2 or len(x) == 3) )
                                                                                    else np.nan )

# check to see if any products remain that need sizing information
print(peanut_butter.loc[ peanut_butter['size_oz'].isna(), ['description', 'size_a', 'size']])

# column creation
peanut_butter['regular_per_size_oz'] = peanut_butter['regular']/peanut_butter['size_oz']
peanut_butter['promo_per_size_oz'] = peanut_butter['promo']/peanut_butter['size_oz']
peanut_butter['pct_change_regular_to_promo_size_oz']=((peanut_butter.promo_per_size_oz - peanut_butter.regular_per_size_oz)/peanut_butter.regular_per_size_oz)*100

# size_oz price
peanut_butter_size_oz = pd.concat([
    peanut_butter[['description','size','regular', 'promo', 'regular_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'regular_per_size_oz':'per_size_oz'}),
    peanut_butter.loc[peanut_butter.promo_per_size_oz>0,['description','size','regular', 'promo', 'promo_per_size_oz', 'pct_change_regular_to_promo_size_oz']].dropna().rename(columns={'promo_per_size_oz':'per_size_oz'})
    ]).sort_values(by=['per_size_oz']).drop_duplicates(subset='description', keep='first')
peanut_butter_size_oz['per_size_rank'] = peanut_butter_size_oz.groupby('per_size_oz')['per_size_oz'].transform('mean').rank(method='dense',ascending=True)
print(peanut_butter_size_oz)

# laundry detergent
laundry_detergent = product_search(filter_term='laundry detergent')
laundry_detergent['size'].apply(lambda x: x.split(" ", 1)[0])
laundry_detergent['size'].apply(lambda x: x.split(" ", 1)[1]).value_counts()
