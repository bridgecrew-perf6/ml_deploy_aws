import pandas as pd
import numpy as np
import os
import subprocess
import time

def test_pth_test_pred():
    pth_results=[]
    python = 'python'
    pth_script_path = './src/predict.py'
    pth_model_folder = './models/pth/'
    jobname = 'vl_blindTestSmall'
    pth_model = 'resnet50'
    pth_weight_path = f'{pth_model_folder}/{pth_model}/{pth_model}-transfer.pth'
    test_dir = 'data/processedSmall/test'
    
    result_base_dir = './src/output'
    cmd = f"{python} {pth_script_path} --jobname {jobname} --model_fn {pth_weight_path} --test_dir {test_dir} --result_base_dir {result_base_dir} --img_size 256"
    print(cmd)
    result = os.popen(cmd)
    print(result.read())
    print('finish cmd pth')
    result_dir=result_base_dir+"/"+jobname
    pred_pth_result_path = f'{result_dir}/blind_test_pred.csv'
    pred_df = pd.read_csv(pred_pth_result_path)

    img_list = pred_df['img_file']
    pred_class= pred_df['pred']
    actual_class =pred_df['actual']
    for i,j in zip(pred_class, img_list):
        sub_results = [
            ''.join(f'{x}' for x in i)
        ]
        whole_results=f"{pth_model}: {'; '.join(sub_results)}"
        pth_results.append([j,whole_results])
    pth_results_df = pd.DataFrame(pth_results, columns = ['img_file', 'results'])
    assert pth_results_df.loc[pth_results_df['img_file'].str.contains('anchor-warp-rope-buoy-beach-2881563.jpg'),'results'].values[0]  == 'resnet50: buoy'
if __name__ == '__main__':
    test_pth_test_pred()