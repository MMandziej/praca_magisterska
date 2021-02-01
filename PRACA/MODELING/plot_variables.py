import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

os.chdir(r'C:\Users\user\Desktop\Git\magisterka\SKRYPTY\MODELING')
sns.set_style('whitegrid')

#### FULL DATA ####
"""data = pd.read_excel(r'model_training/data/RAW/dataset_undersampled_inputed_to_plot.xlsx',
                     converters={
    'NIP': str, 'Label': int, 'RevertedLicenses': int, 'NoWebsite': str,
    'ExecutivesCount': int, 'RodzajRejestru': str, 'CAACEksport': str,
    'ExternalIdsOthers': int, 'PublicMail': str, 'Wiek': int, 'SpecialLegalForm': str,
    'MainPKD': str, 'JednostkiLokalne': int, 'DescriptionNull': str, 'LegalForm': str,
    'MainNAICSCodes': str, 'EMISLegalForm': str, 'Voivodeship': str, 'AuditDaysAgo': int,
    'ProfitBeforeIncomeTax': float, 'RepresentationCount': int, 'OperatingProfitEBIT': float,
    'NumberOfEmployees': int, 'RiskyRemovalBasis': str, 'CAACImport': str,
    'VirtualAccountsPresence': str, 'LatestMarketCapitalization': float, 'TotalEquity': float,
    'AffiliatesCount': int, 'PreviousNamesCount': int, 'PreviousNameChangeYearsAgo': int,
    'RyzykowneDzialalnosciDodatkowe': str, 'NoMail': str, 'RyzykownaDzialalnoscGlowna': str,
    'FormaWlasnosci': str, 'ExpiredLicenses': int, 'OwnersCount': int, 'DeclaredAccountsCount': int,
    'NoFax': str, 'NetProfitLossForThePeriod': float, 'SecondaryPKDCount': int,
    'RemovalDaysAgo': int, 'DividendSum': float, 'AdresLokal': str, 'AdresBiuroWirtualne': str,
    'EntityListedInVATRegistry': str, 'ActiveLicenses': int, 'PhoneNotPresent': str,
    'TotalAssets': float, 'RegisteredCapitalValue': float, 'NetSalesRevenue': float,
    'EmployeeBenefitExpense': float, 'PropertyPlantAndEquipment': float,
    'CashandCashEquivalents': float, 'IssuedCapital': float, 'WorkingCapital': float,
    'RetainedEarnings': float, 'TotalLiabilities': float, 'CurrentLiabilities': float,
    'NonCurrentLiabilities': float, 'IncomeTax': float, 'DepreciationImpairment': float,
    'DepreciationAmortization': float, 'CurrentAssets': float, 'ROE': float, 'ROA': float, 'ROS': float,
    'A1': float, 'A2': float, 'A3': float, 'A4': float, 'A5': float, 'P3': float, 'X8': float,
    'X9': float, 'X10': float, 'X11': float, 'X13': float, 'X14': float, 'BruttoMargin': float})
"""

nips = pd.read_excel(r'model_training/data/RAW/dataset_undersampled_inputed_to_plot.xlsx')[['NIP']]
nips_list = [str(i) for i in list(nips['NIP'])]
data = pd.read_csv(r'model_training/data/RAW/dataset_inputed_plot.csv')

str_cols = [
    'NIP', 'Voivodeship', 'LegalForm', 'EMISLegalForm', 'RodzajRejestru',
    'FormaWlasnosci', 'SpecialLegalForm', 'MainPKD', 'RyzykownaDziałalnoscGlowna',
    'RyzykowneDzialalnosciDodatkowe', 'NoWebsite', 'PublicMail', 'NoMail',
    'NoFax', 'AdresBiuroWirtualne', 'AdresLokal', 'CAACImport', 'CAACEksport',
    'EntityListedInVATRegistry', 'VirtualAccountsPresence', 'RiskyRemovalBasis',
    'PhoneNotPresent', 'MainNAICSCodes', 'DescriptionNull']

int_cols = [
    'Wiek', 'JednostkiLokalne', 'SecondaryPKDCount', 'ActiveLicenses', 'RevertedLicenses', 'ExpiredLicenses',
    'DeclaredAccountsCount',  'RepresentationCount', 'NumberOfEmployees',
    'ExecutivesCount', 'OwnersCount', 'AffiliatesCount', 'ExternalIdsOthers',
    'AuditDaysAgo', 'PreviousNamesCount', 'PreviousNameChangeYearsAgo']

float_cols = [
    'LatestMarketCapitalization', 'RegisteredCapitalValue', 'DividendSum',
    'NetSalesRevenue', 'OperatingProfitEBIT', 'EmployeeBenefitExpense',
    'TotalAssets', 'NetProfitLossForThePeriod', 'PropertyPlantAndEquipment',
    'CashandCashEquivalents', 'TotalEquity', 'IssuedCapital', 'WorkingCapital',
    'RetainedEarnings', 'TotalLiabilities', 'CurrentLiabilities',
    'NonCurrentLiabilities', 'ProfitBeforeIncomeTax', 'IncomeTax', 'DepreciationImpairment',
    'DepreciationAmortization', 'CurrentAssets', 'ROE', 'ROA', 'ROS',
    'A1', 'A2', 'A3', 'A4', 'A5', 'P3', 'X8', 'X9', 'X10', 'X11', 'X13', 'X14',
    'BruttoMargin']#,'P4', 'X6', 'RevenueToCash', 'RevenueToWages'
    
date_cols = ['DataUpadlosci']

data[str_cols] = data[str_cols].astype(str)
data[int_cols] = data[int_cols].apply(pd.to_numeric,
                                      errors='ignore',
                                      downcast='integer')
data[float_cols] = data[float_cols].apply(pd.to_numeric,
                                          downcast='float')
data[date_cols] = data[date_cols].apply(pd.to_datetime,
                                        errors='coerce',
                                        format='%Y-%m-%d')
# FILTER UNDERSAMPLED OBS
data_filtered = data[data['NIP'].isin(nips_list)]

for col in list(data_filtered):
    print(col, data_filtered[col].dtypes, data_filtered[col].isnull().sum())
data_filtered.to_excel(r'model_training/data/RAW/data_EDA.xlsx')

out_path = r'model_training/charts/'
### TO DO 
# FOR CAT COLS EXPORT TO XLSX CZESTOSCI I COUNTY W JAKIS SPOSOB TAK ZEBY BYLY
# JEDNE POD DRUGIM I WTEDY ZROB FORMATKE DO PLOTOWANIA
cat_cols = [
    'Voivodeship', 'LegalForm', 'EMISLegalForm', # 'RodzajRejestru',
    'FormaWlasnosci', 'SpecialLegalForm', 'MainPKD', 'RyzykownaDziałalnoscGlowna',
    'RyzykowneDzialalnosciDodatkowe', 'NoWebsite', 'PublicMail', 'NoMail',
    'NoFax', 'AdresBiuroWirtualne', 'AdresLokal', 'CAACImport', 'CAACEksport',
    'EntityListedInVATRegistry', 'VirtualAccountsPresence', 'RiskyRemovalBasis',
    'PhoneNotPresent', 'MainNAICSCodes', 'DescriptionNull']

num_cols = [
    'Wiek', 'JednostkiLokalne', 'SecondaryPKDCount', 'ActiveLicenses', 'RevertedLicenses', 'ExpiredLicenses',
    'DeclaredAccountsCount',  'RepresentationCount', 'NumberOfEmployees',
    'LatestMarketCapitalization', 'ExecutivesCount', 'OwnersCount', 'AffiliatesCount', 'ExternalIdsOthers',
    'RegisteredCapitalValue', 'AuditDaysAgo', 'PreviousNamesCount',
    'PreviousNameChangeYearsAgo', 'DividendSum', 'NetSalesRevenue',
    'OperatingProfitEBIT', 'EmployeeBenefitExpense', 'TotalAssets', 'NetProfitLossForThePeriod',
    'PropertyPlantAndEquipment', 'CashandCashEquivalents', 'TotalEquity', 'IssuedCapital',
    'WorkingCapital', 'RetainedEarnings', 'TotalLiabilities', 'CurrentLiabilities',
    'NonCurrentLiabilities', 'ProfitBeforeIncomeTax', 'IncomeTax', 'DepreciationImpairment',
    'DepreciationAmortization', 'CurrentAssets', 'ROE', 'ROA', 'ROS', 'A1', 'A2', 'A3', 'A4', 'A5',
    'P3', 'X8', 'X9', 'X10', 'X11', 'X13', 'X14', 'BruttoMargin'#,
    #'P4', 'X6', 'RevenueToCash', 'RevenueToWages'
    ]


for v in cat_cols:
    df_plot = (data_filtered.groupby(v)['Label']
               .value_counts(normalize=True)
               .mul(100)
               .rename('Udział')
               .reset_index())
    
    data_filtered.loc[data_filtered.Voivodeship == 'Other', 'Voivodeship'] = 'INNE'
    fig=sns.countplot(x=v, hue='Label',
                      data=data_filtered)
    fig.set_xticklabels(fig.get_xticklabels(), 
                          rotation=90, 
                          horizontalalignment='right')
    fig.set(ylabel='Liczba firm', xlabel='Województwo')
    
    print(v, data_filtered[v].value_counts(normalize=True).mul(100).round(2))

for v in num_cols:
    data_mean, data_std = data_filtered[v].mean(), data_filtered[v].std()
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    data_plot = data_filtered[(data_filtered[v] > lower) &
                              (data_filtered[v] < upper)]
    df_trimmed = data_plot[[v, 'Label']]
    #sns.distplot(data_plot[v], bins=10, kde=True)
    #ax = sns.boxplot(x=v, data=df_trimmed)
    print(f"{v}. Średnia: {round(data_plot[v].mean(), 2)}, mediana: {round(data_plot[v].median(), 2)}, odchylenie standardowe: {round(data_plot[v].std(), 2)}")

          'Średnia:', round(data_plot[v].mean(), 2), ',',
          'mediana:', round(data_plot[v].median(), 2), ',',
          'odchylenie standardowe:', round(data_plot[v].std(), 2)))
    
### DIVIDE NUM COLS ###
cont_cols = [
    'Wiek', 'NumberOfEmployees', 'LatestMarketCapitalization', 'RegisteredCapitalValue',
    'AuditDaysAgo', 'DividendSum', 'NetSalesRevenue',
    'OperatingProfitEBIT', 'EmployeeBenefitExpense', 'TotalAssets', 'NetProfitLossForThePeriod',
    'PropertyPlantAndEquipment', 'CashandCashEquivalents', 'TotalEquity', 'IssuedCapital',
    'WorkingCapital', 'RetainedEarnings', 'TotalLiabilities', 'CurrentLiabilities',
    'NonCurrentLiabilities', 'ProfitBeforeIncomeTax', 'IncomeTax', 'DepreciationImpairment',
    'DepreciationAmortization', 'CurrentAssets', 'ROE', 'ROA', 'ROS', 'A1', 'A2', 'A3', 'A4', 'A5',
    'P3', 'X8', 'X9', 'X10', 'X11', 'X13', 'X14', 'BruttoMargin']




for metric in float_cols[:10]:
    # BOXPLOT
    #print(metric, data[metric].min(), data[metric].max())
    data_mean, data_std = mean(data_filtered[metric]), std(data_filtered[metric])
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    data_plot = data_filtered[(data_filtered[metric] > lower) &
                              (data_filtered[metric] < upper)]
    #print(metric, data_plot[metric].min(), data_plot[metric].max())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=data_plot[metric], color='lightblue', showfliers=False)
    ax.set_xlim(0.9*data_plot[metric].min(), 1.1*data_plot[metric].max())

    """
    plt.savefig(out_path + 'box//' + metric + '.png')
    # SCATTER
    # DISTPLOT
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.distplot(data_plot[metric], hist=True, kde=False,
                 bins=6, color='darkblue',
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 4})
    ax.set_xlim(0.9*data_plot[metric].min(), 1.1*data_plot[metric].max())
    plt.savefig(out_path + 'dist//' + metric + '.png')"""
    # VIOLINPLOT
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x=data_plot[metric])
    ax.set_xlim(0.9*data_plot[metric].min(), 1.1*data_plot[metric].max())
    plt.title(metric)
    plt.savefig(out_path + 'violin//' + metric + '.png')


for var in cat_cols[:5]:
    # BOXPLOT
    fig, ax = plt.subplots(figsize=(15, 10))
    ax = sns.countplot(x=var, data=data, color='yellow', stacked=True)
    plt.xticks(rotation=90)
    plt.title(var)
    plt.savefig(out_path + 'count//' + var + '.png')
    #plt.show()
    
    
### BORUTA
boruta_charts = pd.read_excel(r'C:\Users\user\Desktop\Git\magisterka\PRACA\charts.xlsx',
                              sheet_name='borutka')
boruta_charts_data = boruta_charts.sort_values(by='meanImp', ascending=False)
boruta_charts_data_conf = boruta_charts_data[boruta_charts_data.decision == 'Confirmed']
boruta_charts_data_conf.columns = [
    'Zmienna', 'Średni wpływ', 'Mediana wpływu', 'Minimalny wpływ',
    'Maksymalny wpływ', 'Norm hits', 'Decyzja']


plt.figure(figsize=(20, 35))
plt.xticks(fontsize=30)
plt.yticks(size=30)
ax = sns.barplot(y="Zmienna", x="Średni wpływ",
                 data=boruta_charts_data_conf[47:], color='green') #palette="Greens_r")
plt.xlabel('Waga średniego wpływu', fontsize=30)
plt.xlim((3, 35))


### POPULACJA
population_data = pd.DataFrame({'Label': [0, 1],
                                'Liczba firm': [33455, 18735]})
ax = sns.barplot(x="Label", y="Liczba firm", data=population_data)
