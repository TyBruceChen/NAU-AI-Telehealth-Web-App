# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:39:24 2024

@author: tc922
"""

from zipfile import ZipFile
import os
import time

def save_cypher(enc_data, enc_path, seed = ""):
    """
    generate a bunch of files which contain binary cypher information
    
    enc_data: list of binaries, containing cyphertexts
    enc_path: the folder where files will be stored
    """
#generate cyphertext files

    os.makedirs(enc_path + seed)
    
    for i,data_chunk in enumerate(enc_data):
        data_path = os.path.join(enc_path + seed,'enc_data-'+str(i))
        with open (data_path,'wb') as data:
            data.write(data_chunk)

def compress_cypher(mpath, zip_name, seed = ""):
    """
    mpath : string
        Where the cypher files are stored.
    zip_file_name : string
        the path + zip file name (no '.zip')
    seed : string
        the suffix of files name.
    """
#compress to zip file
    #ts = time.time()

#mpath = './encrypted_data'
#zip_file_name = 'temp_cyphertexts.zip'

    mpath = mpath + seed
    cyphertext_list = os.listdir(mpath)
    cyphertext_path_list = []
    for cyphertext in cyphertext_list:
        cyphertext_path_list.append(os.path.join(mpath,cyphertext))
        
    zip_name = zip_name + seed
    with ZipFile(zip_name + '.zip','w') as zipf:
        for i,cypher_path in enumerate(cyphertext_path_list):
            zipf.write(cypher_path,arcname=cyphertext_list[i])
            
    #print(time.time() - ts)

def extract_cypher(zip_name, extraction_path = "../", seed = ""):
    """
    zip_name: string
        the path + zip file name (no '.zip')
    extraction_path: string
        the folder that zip file will be extract to
    seed: string
        the suffix of files name (should be the same as cypher_folder_suffix for same client)
    """
#extract files
    
    zip_path = zip_name + seed +'.zip'
    
    with ZipFile(zip_path,'r') as zipf:
        zipf.extractall(path = extraction_path)
        
def delete_files(file_name):
    """
    delete the specified file
    """
    os.remove(file_name)
    print(f'Successfully delete file: {file_name}')
    
    
def delete_folder(file_name):
    """
    delete the specified folder
    """
    os.rmdir(file_name)
    print(f'Successfully delete folder: {file_name}')