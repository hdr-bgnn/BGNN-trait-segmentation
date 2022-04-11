#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 16:35:19 2022

@author: thibault
"""

import os
import pathlib
import json
import sys
import pandas as pd
    
def get_list_2 (input_directory, ext=".json"):
    '''
    Create a list of the absolute path of the files contained in "input_directory" 
    with exetension "ext".
    '''

    extension = '*'+'.json'
    files_list = [str(_) for _ in pathlib.Path(os.path.abspath(input_directory)).glob(extension)]
    
    return files_list

def merge_JsonFiles(files_list, output_json, output_csv=None):
    '''
    merge the json file from the "files_list" and saved the combine result in output
    '''
    
    # if output file doesn't exit create result, if it does load in result from output
    if not os.path.isfile(output_json):
        result = {}
    else :
        with open(output_json, 'r') as infile:
            result = json.load(infile)
            
    for f1 in files_list:
        with open(f1, 'r') as infile:
            result = {**result, **json.load(infile)}
    
    with open(output_json, 'w') as output_file:
        json.dump(result, output_file)
       
    # save as cvs
    if output_csv !=None:
        
        df = pd.DataFrame.from_dict(result)            
        df.to_csv(output_csv, index=True)    
    
def main(input_directory, output_json, output_csv):
    
    files_list = get_list_2 (input_directory, ext=".json")
    merge_JsonFiles(files_list, output_json, output_csv)
    
    
if __name__ == '__main__':
    
    input_dir = sys.argv[1]
    output_json = sys.argv[2]
    output_csv = sys.argv[3]
    
    main(input_dir, output_json, output_csv)
    
