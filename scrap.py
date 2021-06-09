import os
import asyncio
import requests
import argparse
import logging
import pandas as pd
from functools import partial

def set_logger(model_dir, log_name):
    '''Set logger to write info to terminal and save in a file.

    Args:
        model_dir: (string) path to store the log file

    Returns:
        None
    '''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #Don't create redundant handlers everytime set_logger is called
    if not logger.handlers:

        #File handler with debug level stored in model_dir/generation.log
        fh = logging.FileHandler(os.path.join(model_dir, log_name))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s: %(levelname)s: %(message)s'))
        logger.addHandler(fh)

        #Stream handler with info level written to terminal
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(sh)
    
    return logger

nist_url = "https://webbook.nist.gov/cgi/cbook.cgi"

async def scrap_data(cas_ls, params, data_dir):
    spectra_path = os.path.join(data_dir, params['Type'].lower(), '')
    if not os.path.exists(spectra_path): os.makedirs(spectra_path)

    async def get_data(cas_id, params):
        return await loop.run_in_executor(None, partial(requests.get, nist_url, params={**params, 'JCAMP': f'C{cas_id}'}))

    futures = [asyncio.ensure_future(get_data(cas_id, params)) for cas_id in cas_ls]
    result = await asyncio.gather(*futures)
    for idx, response in enumerate(result):
        if response.text != '##TITLE=Spectrum not found.\n##END=\n':
            with open(f'{spectra_path}{cas_ls[idx]}.jdx', 'wb') as data:
                data.write(response.content)

async def scrap_inchi(cas_ls, params, data_dir):
    async def get_inchi(cas_id, params):
        response = await loop.run_in_executor(None, partial(requests.get, nist_url, params={**params, 'GetInChI': f'C{cas_id}'}))
        return response.content.decode("utf-8")
        
    futures = [asyncio.ensure_future(get_inchi(cas_id, params)) for cas_id in cas_ls]
    result = await asyncio.gather(*futures)
    with open(os.path.join(data_dir, 'inchi.txt'), 'a') as data:
        data.write('cas_id\tinchi\n')
        for idx, res in enumerate(result):
            data.write('{}\t{}\n'.format(cas_ls[idx], res))

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default= './data', help = "Directory path to store scrapped data")
parser.add_argument('--cas_list', default= 'species.txt', help = "File containing CAS number and formula of molecules")
parser.add_argument('--scrap_IR', default= True, help = "Whether to download IR or not")
parser.add_argument('--scrap_MS', default= True, help = "Whether to download MS or not")
parser.add_argument('--scrap_InChi', default= True, help = "Whether to download InChi or not")
args = parser.parse_args()
assert os.path.isfile(args.cas_list), "No file named {} exists".format(args.cas_list)

data_dir = args.save_dir
if not os.path.exists(data_dir):
	os.makedirs(data_dir)
set_logger(data_dir, 'scrap.log')

logging.info('Loading CAS file')
cas_df = pd.read_csv(args.cas_list, sep='\t', names = ['name', 'formula', 'cas'], header = 0)
cas_df.dropna(subset=['cas'], inplace=True)
cas_df.cas = cas_df.cas.str.replace('-', '')
cas_ids = list(cas_df.cas)

loop = asyncio.get_event_loop()

logging.info('Scrap Mass spectra')
if args.scrap_MS:
    params = params={'JCAMP': '',  'Index': 0, 'Type': 'Mass'}
    loop.run_until_complete(scrap_data(cas_ids, params, data_dir))

logging.info('Scrap IR spectra')
if args.scrap_IR:
	params={'JCAMP': '', 'Type': 'IR', 'Index': 0}	
	loop.run_until_complete(scrap_data(cas_ids, params, data_dir))

logging.info('Scrap InChi keys')
if args.scrap_InChi:
	params={}
	loop.run_until_complete(scrap_inchi(cas_ids, params, data_dir))

loop.close()

logging.info('End Scrapping')