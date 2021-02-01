import base64
import datetime as d
import errno
import pandas as pd
import time as t
import os

from pwg_ordinary import get_regon_list, get_data, parse_set_update, parse_last_update

############## DEFINE API KEYS ##############
public_key = 'krs4po'
private_key = '12ef683460534adc6352e8d633cb17ad'

####### DEFINE INPUT AND OUTPUT KEYS ########
INPUT_FILE = r'C:\Users\mmandziej001\Desktop\Projects\PWG\INPUT_API\NIP_list.xlsx'
SETUPDATE_OUTPUT_PATH = r'C:\Users\mmandziej001\Desktop\Projects\PWG\\'
GETLASTUPDATE_OUTPUT_PATH = r'C:\Users\mmandziej001\Desktop\Projects\PWG\\'
GETLASTDOC_OUTPUT_PATH = r'C:\Users\mmandziej001\Desktop\Projects\PWG\\'
PDF_OUTPUT_PATH = r'C:\Users\mmandziej001\Desktop\Projects\PWG\\'
LOG_OUTPUT_PATH = r'C:\Users\mmandziej001\Desktop\Projects\PWG\\'

######## DEFINE SLEEP TIME IN SECONDS #######
sleep_time = 960

################## GET PDF ##################
def parse_pdf(getLastDoc_output):
    valid_output = getLastDoc_output[0] 
    pdf_valid_guid = '' 
    regon_status_date = {}
    output_dict = {}
    
    for regon, table in valid_output.items():
        try:
            if 'bin' in table['data']:
                bin_coded = table['data']['bin']
                if len(bin_coded) == 0:
                    status_bin = 'Bin empty'
                    update_date = 'Date empty'
                else:
                    bin_decoded = base64.b64decode(bin_coded)
                    output_dict[regon] = bin_decoded
                    status_bin = 'Bin valid'
                    update_date = table['data']['last_doc_date']    
            else:
                status_bin = table['data']['message']
                update_date = 'Date empty'
            regon_status_date[regon] = [status_bin, update_date]
        except Exception as e :
            error = 'Code: {c}, Message: {m}'.format(c = type(e).__name__, m = str(e))
            status_bin = error
            update_date = 'Date invalid'
            regon_status_date[regon] = [status_bin, update_date]

    for valid_regon, bin_valid in output_dict.items():
        try:
            filepath = os.path.join(PDF_OUTPUT_PATH, str(valid_regon))
            os.makedirs(filepath)  
            with open(os.path.join(filepath, valid_regon + '{}.pdf'.format(pdf_valid_guid)), 'wb') as fp:
                fp.write(bin_valid)
        except OSError as e:
            if e.errno == errno.EEXIST:
                regon_status_date[valid_regon][0] = 'Could not create directory file'
        except TypeError:
            regon_status_date[valid_regon][0] = 'Wrong NIP/KRS/REGON format'
        except:
            regon_status_date[valid_regon][0] = 'Bin invalid'
    
    output = pd.DataFrame.from_dict(regon_status_date, orient = 'index')
    output.reset_index(inplace=True)
    output.columns = ['REGON', 'BinStatus', 'GLD_LastDocDate']
    output.to_excel(LOG_OUTPUT_PATH + 'PDF_Output_PARSED_' + str(d.date.today()) + '.xlsx', index=False)
    return output

################## CREATE LOG #######o##########
def create_log():
    log_df = parse_SetUpdateFirma[0].merge(parse_LastUpdate[0], how = 'outer', on = 'REGON', sort = True)
    log_df = log_df.merge(parsePDF_output, how = 'outer', on = 'REGON', sort = True)
    log_df = log_df[['REGON', 'Code', 'Message', 'LL_LastDocDate', 'BinStatus', 'GLD_LastDocDate']]
    log_df.sort_values(by = ['REGON'], inplace=True)
    log_df.to_excel(LOG_OUTPUT_PATH + 'KRS_log_' + str(d.date.today()) + '.xlsx', index=False)
    return log_df 

############################ GET OUPTUT #################################
############ GET NIP NAD CASE LIST ############
regon_list = get_regon_list(INPUT_FILE)[0]

################# SET UPDATE ##################
setUpdateFirma_output = get_data('setUpdateFirma', regon_list, SETUPDATE_OUTPUT_PATH)
parse_SetUpdateFirma = parse_set_update(setUpdateFirma_output, LOG_OUTPUT_PATH)

################### SLEEP #####################
t.sleep(sleep_time)

############# CHECK LAST UPDATE  ##############
getLastUpdate_output = get_data('getLastUpdate', regon_list, GETLASTUPDATE_OUTPUT_PATH)
parse_LastUpdate = parse_last_update(getLastUpdate_output, LOG_OUTPUT_PATH)

############## GET LAST DOC ###################
getLastDoc_output = get_data('getLastDoc', regon_list, GETLASTDOC_OUTPUT_PATH)
parsePDF_output = parse_pdf(getLastDoc_output)

###############################################
log = create_log()
