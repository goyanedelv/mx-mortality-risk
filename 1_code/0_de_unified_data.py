# import libraries
import pandas as pd

# Read data
data_census = pd.read_csv('0_data/2_intermediate/piramide_estratificada.csv', engine='python')

data_mortal = pd.read_csv('0_data/2_intermediate/simplified_2017_mortality.csv', engine='python')

# Explore
data_census.columns
data_mortal.columns

data_census.head(10)
data_mortal.info

# Standardizing column names
## Census data
rename_dict_census = {'SEXO' : 'sexo', 'EDAD' : 'edad', 'OCUPACION_C' : 'ocupacion', 'ESCOACUM' : 'escolaridad', 'SITUA_CONYUGAL' : 'edo_civil', 'Municipio' : 'municipio'}
data_census.rename(columns = rename_dict_census, inplace = True)
data_census = data_census.drop('Unnamed: 0', 1)

## Mortality data
rename_dict_mortal = {'edad_fallecimiento' : 'edad', 'escolarida' : 'escolaridad'}
data_mortal.rename(columns = rename_dict_mortal, inplace = True)
data_mortal = data_mortal.drop('Unnamed: 0', 1)

data_mortal = data_mortal.drop(['ent_resid_2', 'mun_resid_2', 'tloc_ocurr', 'anio_nacim'], 1)

# Pasting municipio name and other features to data_mortal
muni_ft = pd.read_excel('0_data/1_raw/caract_municipio_mx.xls')
muni_ft = muni_ft.drop(['year', 'state_code', 'municipio_code', 'state_name_official', 'state_abbr', 'state_abbr_official'], 1)

muni_ft.rename(columns = {'municipio_name' : 'municipio'}, inplace = True)
data_mortal = data_mortal.merge(muni_ft, on = 'region', how = 'left')

# Pasting municipio features to data_census (### sum inconsistency is here)
data_census = data_census.merge(muni_ft, on = 'region', how = 'left')


# Standardizing Ocupacion
## Exploring a little
dc_ocup = data_census[['ocupacion']].drop_duplicates()
dm_ocup = data_mortal[['ocupacion']].drop_duplicates()

## We will keep only first digit of the ocupacion variable in data_census (overarching category)
data_census['ocupacion'] = data_census['ocupacion'].astype(str).str[0]
data_census.loc[ data_census['ocupacion'] == 'n','ocupacion'] = '99'

data_mortal.loc[ data_mortal['ocupacion'] > 9,'ocupacion'] = 99
data_mortal['ocupacion'] = data_mortal['ocupacion'].astype(str)

# Standardizing Escolaridad
## explore escolaridad
dc_esco = data_census[['escolaridad']].drop_duplicates()
dm_esco = data_mortal[['escolaridad']].drop_duplicates()

## Create a dictionary
gral = list(range(0,24+1)) + [99]
deta = [1] + [3]*5 + [4] + [5]*2 + [6] + [7] + [8]*2 + [9]*5 +[10]*7 + [99]
dict_esco = dict(zip(gral, deta))

## Apply dictionary
data_census['escolaridad'] = data_census['escolaridad'].map(dict_esco).fillna(data_census['escolaridad'])

## On mortality, if 88 then 0
data_mortal.loc[data_mortal['escolaridad'] == 88, 'escolaridad'] = 0

## On both mortality and census, if 99 create new var with value 1. If not 0. 
data_census['esco_avail'] = 0
data_mortal['esco_avail'] = 0

data_census.loc[data_census['escolaridad'] == 99, 'esco_avail'] = 1
data_mortal.loc[data_mortal['escolaridad'] == 99, 'esco_avail'] = 1

data_census.loc[data_census['escolaridad'].isna(), 'esco_avail'] = 1
data_mortal.loc[data_mortal['escolaridad'].isna(), 'esco_avail'] = 1

## Also, move 99 to average at that age and sex.
### explore new escolaridad
dc_esco = data_census[['escolaridad']].drop_duplicates()
dm_esco = data_mortal[['escolaridad']].drop_duplicates()

### temporary solution if 99 -> 0. Ideal: use a regression
data_mortal.loc[data_mortal['escolaridad'] == 99, 'escolaridad'] = 0

data_census.loc[data_census['escolaridad'] == 99, 'escolaridad'] = 0
data_census.loc[data_census['escolaridad'].isna(), 'escolaridad'] = 0

### explore new escolaridad
dc_esco = data_census[['escolaridad']].drop_duplicates()
dm_esco = data_mortal[['escolaridad']].drop_duplicates()

# Standardizing Edo Civil
### explore edo civil
dc_edociv = data_census[['edo_civil']].drop_duplicates()
dm_edociv = data_mortal[['edo_civil']].drop_duplicates()

### Mortalidad
#### mort: 1 soltera; 2 divorciada; 3 viuda; 4 union libre; 5 casada; 6 separada; 8 no aplica, menores de 12; 9 no especificado
#### census: aparentemente igual, confirmar

## temporary solution:
data_census['edo_civil'] = data_census['edo_civil'].fillna(0)

# Int or float to string
data_census['edo_civil'] = data_census['edo_civil'].astype(str)
data_census['ocupacion'] = data_census['ocupacion'].astype(str)

data_mortal['edo_civil'] = data_mortal['edo_civil'].astype(str)
data_mortal['ocupacion'] = data_mortal['ocupacion'].astype(str)

# Explore what we have so far
data_census.dtypes
data_mortal.dtypes

# Transform metro_area into a dummy
data_mortal.loc[-data_mortal['metro_area'].isna(),'metro_area'] = 1
data_mortal['metro_area'] = data_mortal['metro_area'].fillna(0)

data_census.loc[-data_census['metro_area'].isna(),'metro_area'] = 1
data_census['metro_area'] = data_census['metro_area'].fillna(0)

# create a factor = 1 on deaths data
data_mortal['factor'] = 1

# create a causa_nombre variable
data_census['causa_nombre'] = 'Vive'

# dummy vive
data_mortal['death'] = 1
data_census['death'] = 0

# row-bind both datasets
len(data_census.columns)
len(data_mortal.columns)

data_for_ml = pd.concat([data_census, data_mortal], ignore_index=True)

# Save data
data_for_ml.to_csv('0_data/3_primary/data_for_ml.csv', index=False)

## To-do's
### regression-like impuation of escolaridad
### substract deaths to census