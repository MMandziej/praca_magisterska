from datetime import date
import json
import pandas as pd
import time as t

from copy import deepcopy
from math import floor, ceil
from random import randint
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

#from keys_coded import GHC_CREDS

from vatcheck import get_nip_name, slice_export_jsons, parse_vat_status, check_shared_accs, create_output,\
                     export_output, export_vatcheck_to_sql, score_removal_reason, initial_cols, cols_rename, known_cols

ID_CLASS = 'NIP'
INPUT_FILE = r"ID_LIST.csv"
JSON_OUTPUT_PATH = r"JSONS\\"
XLSX_OUTPUT_PATH = r"OUT\\"
PROJECT_NAME = "TestInput"

# Define selenium settings
options = webdriver.ChromeOptions()
preferences = {"download.default_directory": r"C:\Users\user\Desktop\SGH\OUT"}
options.add_experimental_option("prefs", preferences)


def generate_url_list(id_class, company_ids_list):
    url_list = []
    input_company_ids = []
    for i in company_ids_list:
        url = 'https://wl-api.mf.gov.pl/api/search/' + id_class.lower() + 's/' + str(i) + '/?date=' + str(date.today())
        url_list.append(url)
        for nip in i.split(","):
            input_company_ids.append(nip)
    return url_list, input_company_ids


def collect_export_jsons(url_list):
    #driver = webdriver.Chrome(chrome_options=options)
    driver = webdriver.Chrome(ChromeDriverManager().install())
    json_files = {}
    count = 496
    try:
        for url in url_list:
            driver.get(url)
            file_html = driver.page_source
            plik = json.loads(file_html[84:-20])['result']['subjects']#120:-20     
            json_files[count] = plik
            print("Batch %s downloaded successfully" % str(count))
            t.sleep(randint(15, 60))
            
            with open(JSON_OUTPUT_PATH + 'VATCHECK_' + str(count) + '_' +
                      str(date.today()) + '.json', 'w') as file:
                json.dump(plik, file)
            count += 1
    except:
        print('Failed to download batch %s' %count)
    return json_files

nip_list = list(pd.read_csv(r"ID_LIST.csv", header=None)[0])
nip_list = [str(i) for i in nip_list]


sliced_valid_ids_list = []
start_param = 0
if len(nip_list) >= 30:
    end_param = 30
    if len(nip_list) < 300:
        print("Input shall be divided into %s batches." % ceil(len(nip_list)/30))
    else:
        print("""Input shall be divided into {} batches.
              Warning: According to doccumentation input to big to collect all data.
              Please downsize to 300 entites at most.""".format(ceil(len(nip_list)/30)))
else:
    end_param = len(nip_list)
    sliced_valid_ids_list.append(','.join(nip_list[start_param:end_param]))

while len(nip_list) > end_param:
    sliced_ids = ','.join(nip_list[start_param:end_param])
    sliced_valid_ids_list.append(sliced_ids)
    start_param += 30
    print('Prawie koniec' + str(start_param))
    if len(nip_list) - end_param <= 30:
        end_param = len(nip_list)
        sliced_ids = ','.join(nip_list[start_param:end_param])
        sliced_valid_ids_list.append(sliced_ids)
        print("Input successfully divided into {} batches.".format(len(sliced_valid_ids_list)))
        break
    else:
        end_param += 30


nips_errors = {}
nip_list, nips_invalid, log_df, client_nips_accounts = get_nip_name(INPUT_FILE, ID_CLASS)
urls, company_ids_all = generate_url_list(ID_CLASS, sliced_valid_ids_list) #nip_list
batch_jsons = collect_export_jsons(urls[496:])
merged_jsons = slice_export_jsons(batch_jsons, JSON_OUTPUT_PATH)
parsed_jsons, nips_errors, parsed_nips_accounts = parse_vat_status(merged_jsons, nips_errors,
                                                                   company_ids_all, client_nips_accounts)
all_accounts, shared_accs = check_shared_accs(parsed_nips_accounts)
vat_output = create_output(parsed_jsons, nips_errors)
export_output(vat_output, log_df, XLSX_OUTPUT_PATH)
#table = export_vatcheck_to_sql(PROJECT_NAME, GHC_CREDS, vat_output[0])

jsony_przed_wyjebaniem = deepcopy(batch_jsons)