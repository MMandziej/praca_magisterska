"""
Script used to retrieve and parse register data from EMIS API based on
getFullCompany method and list of companies IDs. Companies IDs have to be
 fist converted into EMIS IDs which is included in script.

Data are processed and exported to xlsx file and used for investigating
analysis purposes and FAIT project.
"""
import datetime as d
import pandas as pd
import time as t

from keys_coded import USER_NAME, PASSWORD

from emis import get_session_id, get_nip_name, get_isic_list, get_company_info,\
                 flatten_json, parse_employees, parse_owners, parse_executives, parse_naics,\
                 parse_external_ids, cols_drop, cols_to_drop_list, output_rename,\
                 known_cols, len_guids, cols_guids_to_drop

START = t.time()

############################ USER PART ###########################
######## DEFINE METHOD, ID CLASS, INPUT NAD OUTPUT PATHS #########
METHOD = 'getFullCompany'
ID_CLASS = 'NIP'
INPUT_FILE = r'C:\Users\mmandziej001\Desktop\Projects\EMIS\NIP_list.xlsx'
JSON_OUTPUT_PATH = r'C:\Users\mmandziej001\Desktop\Projects\EMIS\OUTPUT_API\JSONS_REGISTER\\'
XLSX_OUTPUT_PATH = r'C:\Users\mmandziej001\Desktop\Projects\EMIS\OUTPUT_API\\'
######################## END OF USER PART ########################

def parse_company_info(collected_jsons: dict,
                       collected_errors: dict) -> list:
    """ Parse collected JSONs with register data and errors to list od dicts

    Parameters
    ----------
    collected_jsons : list (of dicts)
        a list of dicts storing company id (key) and collected JSON (value)
    collected_errors : dictionary
        a dictionary stoprzedring NIP (key) and error (value) collected while
        acquiring EMIS IDs and JSONS with financial data

    Returns
    -------
    result_list: list
        a list storing dictionaries with register data for companies
    collected_errors : dictionary
        a dictionary storing NIP (key) and error (value) collected while
        acquiring EMIS IDs and JSONS with financial data
    """

    result_list = []
    # Iterate over dictionary {NIP : JSON}
    for i in collected_jsons:
        #try:
            dictionary = collected_jsons[i]
            dictionary['nip'] = i
            dictionary['Data rozpoczęcia współpracy'] = None
            # Drop unnecessary columns
            for k in cols_drop:
                dictionary.pop(k, None)
            # Parse collected JSON file
            result = flatten_json(dictionary)
            #Parse numbrer of employees to single field
            parse_employees(result)
            parse_owners(result)
            parse_executives(result)
            parse_naics(result)
            parse_external_ids(result)
            # Create DataFrame as row and to list of all companies
            result_df = pd.DataFrame([result])
            result_list.append(result_df)
        #except Exception as exc:
            # If data could not be parsed raise exception and append error
            # Append existing list of errors
            #error = 'Code: {c}, Message: {m}'.format(c=type(exc).__name__, m=str(exc))
            #collected_errors[i] = error
    print("Data successfully parsed for %s companies." % len(result_list))
    return result_list, collected_errors


def create_register_output(result_list: list, parsed_errors: dict) -> list:
    """ Create output tables in pandas DataFrames from parsedJSONs

    Parameters
    ----------
    result_list: list(of dicts)
         list of parsed dictionaries storing processed register data prepared
         for output
    parsed_errors: dictionary
        a dictionary storing NIP (key) and error (value) collected while
        acquiring EMIS IDs and JSONS with financial data and parsing JSONS

    Returns
    -------
    full_output : pd.DataFrame
        a DataFrame storing aggregated both valid nad ivalid collected register data
    output : pd.DataFrame
        a DataFrame storing only valid collected register data
    invalid_output : pd.DataFrame
    a DataFrame storing errors collected while collecting EMIS IDs / JSONS and
    parsing register data
    """

    if result_list:
        # Concat all rows into output table
        output = pd.concat(result_list, ignore_index=True, sort=False)
        cols_to_drop = []
        for col in output.columns:
            for length in len_guids:
                if col[length:] in cols_guids_to_drop:
                    cols_to_drop.append(col)

        cols_to_drop.extend(cols_to_drop_list)
        output.drop(columns=cols_to_drop, inplace=True)
        output = output.rename(output_rename, axis='columns').sort_values(by=['NIP'])

        all_cols = list(output)
        unknown_cols = [c for c in all_cols if c not in known_cols]
        cols = known_cols + unknown_cols
        output = output[cols].sort_values(by=['NIP'], na_position='last')
    else:
        # Create empty DataFrame if there is no valid output
        cols = known_cols
        output = pd.DataFrame(columns=cols)

    numeric_cols = ['NumberOfEmployees', 'FewEmployees', 'OwnersCount', 'ExecutivesCount']
    # Fill null values and change type of columns
    output = output.fillna('')
    output = output.astype(str)
    output[numeric_cols] = output[numeric_cols].apply(pd.to_numeric, errors='coerce')
    if parsed_errors:
        # Create DataFrame from list of errors
        invalid_output = pd.DataFrame.from_dict(parsed_errors, orient='index').reset_index()
        invalid_output.columns = ['NIP', 'Error']
        invalid_output.sort_values(by=['NIP'], inplace=True, na_position='last')
    else:
        # Create empty DataFrame if there is no valid output
        invalid_output = pd.DataFrame(columns=['NIP', 'Error'])

    # Merge both outputs
    full_output = pd.concat([output, invalid_output.drop(['Error'], axis=1)], sort=True)
    full_output.drop_duplicates(keep='first', subset='NIP', inplace=True)
    full_output.reset_index(drop=True)
    #full_output = full_output[cols].sort_values(by=['NIP'], na_position='last')
    print("Output successfully created.")
    return full_output, output, invalid_output


def export_output(output_parsed: pd.DataFrame):
    """ Split parsed register data into separated tables and
    export each into Excel sheet in defined file. Exports also EMIS_input table
    which is later used for FAIT purposes.

    Parameters
    ----------
    output_parsed : pd.DataFrame
        a DataFrame storing aggregated parsed financial data merged with invalid output

    """
    try:
        output_list = [i for i in output_parsed]
        sheet_names = ['FullOutput', 'ValidOutput', 'InvalidOutput']

        with pd.ExcelWriter(XLSX_OUTPUT_PATH +
                            'EMIS_REGISTER_' +
                            str(d.date.today()) +
                            '.xlsx') as writer:
            for sheet_name, table in zip(sheet_names, output_list):
                table.to_excel(writer, sheet_name, index=False)
            output_list[0].to_excel(writer, sheet_name='EMIS_input', columns=[
                'NIP', 'EMISID', 'Data rozpoczęcia współpracy'], index=False)
            log_df.to_excel(writer, sheet_name="LOG", index=False)
        print("Output successfully exported to declared location: " + XLSX_OUTPUT_PATH[:-2])
    except PermissionError:
        print("Close previous output file.")
    except Exception as e:
        # Raise exception and print Error if export unsuccessfull
        error = 'Code: {c}, Message: {m}'.format(c=type(e).__name__, m=str(e))
        print(error)

############### GET OUTPUT ##################
downloaded_o = pd.read_excel(r"C:\Users\mmandziej001\Desktop\Projects\FAIT\Prediction Module\POLAND_DANE\REGISTER\Operational\operational_downloaded_all.xlsx")
downloaded_o = downloaded_o[~pd.isnull(downloaded_o['NIP'])]
downloaded_o[['NIP']] = downloaded_o[['NIP']].astype(int).astype(str)



all_o = pd.read_excel(r"C:\Users\mmandziej001\Desktop\Projects\FAIT\Prediction Module\POLAND_DANE\Operational_ALL.xlsx")
all_o = all_o[~pd.isnull(all_o['NIP'])]
all_o[['NIP']] = all_o[['NIP']].astype(int).astype(str)

to_download = all_o[all_o['NIP'].isin(list(all_o[['NIP']]))]
nips_errors = {}
nips_isics = {}
for idx, v in enumerate(nips_invalid):
    nips_isics[idx] = v

nips = pd.read_excel(INPUT_FILE, header=None)
nips_isics = dict(zip([str(i) for i in list(nips[1])],
                      [str(i) for i in list(nips[0])]))

import os, shutil, json
source = r'C:/Users/mmandziej001/Desktop/Projects/EMIS/OUTPUT_API/JSONS_REGISTER/'
dest1 = r'C:/Users/mmandziej001/Desktop/Projects/FAIT/Prediction Module/POLAND_DANE/REGISTER/Operational/ALL/'
files = os.listdir(source)

for f in files:
    try:
        shutil.move(source+f, dest1)
    except:
        print(f)

nips_errors = {}
nips_jsons_loaded = {}
nips_jsons_fin = []
path_to_json = r'C:\Users\mmandziej001\Desktop\Projects\FAIT\Prediction Module\POLAND_DANE\FINANCE\Investigated'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json') and
              'INVALID' not in pos_json]
for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        print(index)
        try:
            nips_jsons_loaded[js[:10]] = json.load(json_file)
        except:
            print(js + ' invalid file')
##
print("Downloading and parsing register data pending. Start time: %s." % d.datetime.now())
nip_list, nips_invalid, log_df = get_nip_name(INPUT_FILE, ID_CLASS)
session_id = get_session_id(USER_NAME, PASSWORD)
nips_isics, nips_errors = get_isic_list(nip_list, session_id, ID_CLASS)
nips_jsons, nips_errors = get_company_info(METHOD,
                                           nips_isics,
                                           nips_errors,
                                           session_id,
                                           JSON_OUTPUT_PATH)

parsed_jsons, nips_errors = parse_company_info(nips_jsons_loaded, nips_errors)
register_output = create_register_output(parsed_jsons, nips_errors)
export_output(register_output)

END = t.time()
execution_time = END - START
print("Execution time: ", str(execution_time))
