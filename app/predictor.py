import flask
from flask import Flask, flash, render_template, url_for, send_from_directory, redirect, request, abort, jsonify, session
from flask_session import Session
import sys
import os
from werkzeug.utils import secure_filename
import json
from collections import defaultdict
from datetime import datetime
import glob
import pandas as pd
import numpy as np
import subprocess
import time
import glob


app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = './app/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def main():
    return render_template("header.html")


@app.route('/demo/', methods=['POST', 'GET'])
def emerson():
    # model list
    session['cnn_model_folder'] = './models/pth'
    session['result_base_dir'] = './src/output'
    session['upload_path'] = './app/static/uploads'
    session['pth_script_path'] = './src/predict.py'
    session['python'] = 'python'
        
    cnn_model_folder = session.get('cnn_model_folder')
    cnn_models = os.listdir(cnn_model_folder)
    print('render cnn_models',cnn_models)
    return render_template("demo.html",
                           cnn_models=cnn_models)


@app.route('/predictresult/', methods=['POST'])
def predictresult():
    cnn_model_folder = session.get('cnn_model_folder')
    result_base_dir = session.get('result_base_dir')
    upload_path = session.get('upload_path')
    pth_script_path = session.get('pth_script_path')
    python = session.get('python')
    
    # Load data from form and create a cmd line call
    # predict_data_decode = request.data.decode().split("&")
    # predict_data = defaultdict(list)
    # for data in predict_data_decode:
    #     k,v = data.split("=")
    #     predict_data[k].append(v)
    predict_data = request.form
    print(predict_data)
    jobname = predict_data['predictJobName'] + \
            datetime.now().strftime("_%Y%m%d_%H%M%S")

    os.mkdir(f'{result_base_dir}/{jobname}')
    if 'file' not in request.files:
            print('No file part')
    if request.form.get('gridImgIn') == 'upload':
        files = request.files.getlist('file')
    elif len(request.files.getlist('folder')) > 0:
        files = request.files.getlist('folder')
    else:
        print("Please select images or input file path.")
    n_img = len(files)
    print(files)
    # if user does not select file, browser also
    # submit an empty part without filename
    image_path = f'{upload_path}/{jobname}'
    print(image_path)
    os.mkdir(image_path)
    if n_img == 0:
        print("can't read or find file")
    elif n_img == 1:
        file = files[0]
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = f'{image_path}/{filename}'
            file.save(save_path)
            print('one image upload complete')
    else:
        for file in files:
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = f'{image_path}/{filename}'
                file.save(save_path)
        print(f"{n_img} multiple images upload complete")
    
    # predict_data = {'predictJobName': [], 'predictSelectCnnModel': [], 'predictSelectYoloModel': [], 
    # 'PredictYoloConfidenceThreshold': [], 'gridImgIn': [], 'predictServerPath': []})
    cnn_model = request.form.get('predictSelectCnnModel')
    if cnn_model is None:
        flash("Please choose models.")
    print('selected cnn_model',cnn_model)
    if cnn_model is not None:
        cnn_model_list = request.form.getlist('predictSelectCnnModel')
        print('get cnn_model_list',cnn_model_list)
        # cnn_output_path = f'{result_base_dir}/{jobname}'
        for cnn_model in cnn_model_list:
            cnn_weight_path = f'{cnn_model_folder}/{cnn_model}/{cnn_model}-transfer.pth'
            # cnn_output_path = f'{result_base_dir}/{jobname}/{cnn_model}'
            # os.mkdir(cnn_output_path)
            cmd = f"{python} {pth_script_path} --jobname {jobname} --model_fn {cnn_weight_path} --test_dir {image_path} --result_base_dir {result_base_dir} --img_size 256"
            print('cmd:',cmd)
            result = os.popen(cmd)
            print('result:',result.read())
        print('finish cmd pth')
        result_dir=result_base_dir+"/"+jobname
        pred_pth_result_path = f'{result_dir}/blind_test_pred.csv'
        # print('pred_pth_result_path',pred_pth_result_path)
        pred_df = pd.read_csv(pred_pth_result_path)
        pth_results=[]
        img_list = pred_df['img_file']
        pred_class= pred_df['pred']
        for i,j in zip(pred_class, img_list):
            sub_results = [
                ''.join(f'{x}' for x in i)
            ]
            whole_results=f"{cnn_model}: {'; '.join(sub_results)}"
            pth_results.append([j,whole_results])
        pth_results_df = pd.DataFrame(pth_results, columns = ['img_file', 'results'])
        print('pth_results_df',pth_results_df)

    if cnn_model is not None:
        combine_result_df = pth_results_df.groupby(
        'img_file').results.apply('\n'.join).reset_index()
        combine_final_result_list = combine_result_df.values.tolist()
        return_image_path_list = []
        return_image_path_list.append(f"./uploads/{jobname}/")
        print('return_image_path_list',return_image_path_list)

    print('pwd',os.getcwd())
        # for result in combine_final_result_list:
            # for d in return_image_path_list:
                # print(result[0])
    # Process result folder and display image and table

    # if cnn_model is not None:
    #     combine_result_df = pth_result_df.groupby(
    #     'img_file').results.apply('\n'.join).reset_index()
    #     combine_final_result_list = combine_result_df.values.tolist()
    #     return_image_path_list = []
    #     return_image_path_list.append(f"./app/static/uploads/{jobname}/")  

    # process yolo result
    # if cnn_model is not None and yolo_model is not None:
    #     combine_df = cnn_result_df.append(yolo_result_df, ignore_index=True)
    #     combine_result_df = combine_df.groupby(
    #         'img_file').results.apply('\n'.join).reset_index()
    #     # format like: [['img_file1', 'result1'], ['img_file1', 'result2']]
    #     combine_final_result_list = combine_result_df.values.tolist()
    #     return_image_path_list = []
    #     return_image_path_list.append(f"./uploads/{jobname}/")
    #     for yolo_model in yolo_model_list:
    #         return_image_path_list.append(
    #             f"./output/{jobname}/{yolo_model}/")
    # elif (cnn_model is not None) is not (yolo_model is not None):
    #     if cnn_model is not None:
    #         combine_result_df = cnn_result_df.groupby(
    #             'img_file').results.apply('\n'.join).reset_index()
    #         combine_final_result_list = combine_result_df.values.tolist()
    #         return_image_path_list = []
    #         return_image_path_list.append(f"./uploads/{jobname}/")
    #     else:
    #         combine_result_df = yolo_result_df.groupby(
    #             'img_file').results.apply('\n'.join).reset_index()
    #         combine_final_result_list = combine_result_df.values.tolist()
    #         return_image_path_list = []
    #         for yolo_model in yolo_model_list:
    #             return_image_path_list.append(
    #                 f"./output/{jobname}/{yolo_model}/")
      
    return jsonify({'image': render_template("predictresultImage.html",
                                             dir=return_image_path_list, results=combine_final_result_list),
                    'table': render_template("predictresultTable.html",
                                             results=combine_final_result_list)})


@app.route('/traincnnresult/', methods=['POST'])
def traincnnresult():
    cnn_model_folder = session.get('cnn_model_folder')
    python3 = session.get('python3')
    cnn_train_script = session.get('cnn_train_script')
    

    # Load data from form
    train_cnn_data = request.form
    print(train_cnn_data)
    jobname = train_cnn_data['trainCnnJobName'] + \
        datetime.now().strftime("_%Y%m%d_%H%M%S")
    trainCnnEpochs = int(train_cnn_data['trainCnnEpochs'])
    trainCnnImageResize = int(train_cnn_data['trainCnnImageResize'])
    trainCnnBatchSize = int(train_cnn_data['trainCnnBatchSize'])
    trainCnnTrainSize = int(train_cnn_data['trainCnnTrainSize'])
    trainCnnArch = train_cnn_data['trainCnnArch']
    trainClassType = train_cnn_data['trainClassType']
    trainImgColor = train_cnn_data['trainImgColor']
    trainCnnTestRatio = float(train_cnn_data['trainCnnTestRatio'])
    trainCnnInputPath = train_cnn_data['trainCnnInputPath']

    # create directories
    result_dir = f'{cnn_model_folder}/{jobname}'
    os.mkdir(result_dir)
    result_log_path = f'{result_dir}/cnn_crop_aug_train.out'

    # create cmd line to train
    if trainImgColor == 'color':
        color = ''
    else:
        color = ' --grayscale'

    if sys.platform == "linux":
        # the command below for running py files in the background in Linux and mac, use (ps ax | grep filename.py) to find its id, and use (kill -9 {job_id} &) to kill the job
        cmd = f"{python3} {cnn_train_script} --jobname '{jobname}' --directory '{trainCnnInputPath}' --result_base_dir '{cnn_model_folder}' --epochs {trainCnnEpochs} --img_size {trainCnnImageResize} --cnn_sel {trainCnnArch} --test_ratio {trainCnnTestRatio} --batch_size {trainCnnBatchSize} --target_train_size {trainCnnTrainSize} --class_type {trainClassType}{color} >> {result_log_path} &"
        print(cmd)
        # os.system(cmd)
        subprocess.Popen(cmd, close_fds=True, shell=True)
        # wait a few seconds
        time.sleep(5)
        # check process running and print output in Linux
        get_running_jobs_cmd = "ps aux | grep cnn_crop_aug_train"
        # if hostname == 'kuldldsccappo01.kul.apac.dell.com':
        #     get_running_jobs_cmd = f"{get_running_jobs_cmd}"
        jobs = subprocess.Popen(
            get_running_jobs_cmd, shell=True, stdout=subprocess.PIPE)
        output = set(jobs.stdout)
        print(output)
        job_list_full = [x.decode('utf-8').strip('\n')
                     for x in output if 'grep' not in x.decode('utf-8').strip('\n')]
        job_list = sorted([i.split("--")[1].split(" ")[1] for i in job_list_full])
        print(job_list)
    else:
        print("system platform is neither windows nor linux.")

    # parse output to the format
    log_output = ''
    output = {}
    for job in job_list:
        job_log_path = f'{cnn_model_folder}/{job}/cnn_crop_aug_train.out'
        print(job_log_path)
        if os.path.isfile(job_log_path):
            log_file = open(job_log_path, 'r')
            log_output = tail(log_file, 20)
        else:
            log_output = ''
        output[job] = log_output
    print(output)
    return jsonify({'cnnresult': render_template("trainCnnresult.html", job_list=job_list, output=output)})


@app.route('/traincnnresultrefresh/', methods=['POST'])
def traincnnresultrefresh():
    cnn_model_folder = session.get('cnn_model_folder')

    if sys.platform == "linux":
        # check process running and print output in Linux
        get_running_jobs_cmd = "ps aux | grep cnn_crop_aug_train"
        jobs = subprocess.Popen(
            get_running_jobs_cmd, shell=True, stdout=subprocess.PIPE)
        output = set(jobs.stdout)
        print(output)
        job_list_full = [x.decode('utf-8').strip('\n')
                         for x in output if 'grep' not in x.decode('utf-8').strip('\n')]
        job_list = sorted([i.split("--")[1].split(" ")[1]
                           for i in job_list_full])
        print(job_list)
    else:
        print("system platform is neither windows nor linux.")

    # parse output to the format
    log_output = ''
    output = {}
    for job in job_list:
        job_log_path = f'{cnn_model_folder}/{job}/cnn_crop_aug_train.out'
        print(job_log_path)
        if os.path.isfile(job_log_path):
            log_file = open(job_log_path, 'r')
            log_output = tail(log_file, 20)
        else:
            log_output = ''
        output[job] = log_output
    print(output)
    return jsonify({'cnnresultrefresh': render_template("trainCnnresult.html", job_list=job_list, output=output)})


@app.route('/trainyoloresult/', methods=['POST'])
def trainyoloresult():
    yolo_model_folder = session.get('yolo_model_folder')
    python3 = session.get('python3')
    yolo_train_script = session.get('yolo_train_script')

    # Load data from form
    train_yolo_data = request.form
    print(train_yolo_data)
    jobname = train_yolo_data['trainYoloJobName'] + \
        datetime.now().strftime("_%Y%m%d_%H%M%S")
    trainYoloEpochs = int(train_yolo_data['trainYoloEpochs'])
    trainYoloBatchSize = int(train_yolo_data['trainYoloBatchSize'])
    trainYoloTestRatio = float(train_yolo_data['trainYoloTestRatio'])
    trainYoloModelSize = train_yolo_data['trainYoloModelSize']
    trainYoloOptimizer = train_yolo_data['trainYoloOptimizer']
    trainYoloUrl = train_yolo_data['trainYoloUrl']

    # set trainYoloOptimizer value for cmd
    if trainYoloOptimizer is None:
        trainYoloOptimizer = ''
    else:
        trainYoloOptimizer = ' --adam'

    # create result directories
    result_dir = f'{yolo_model_folder}/{jobname}'
    os.mkdir(result_dir)
    result_log_path = f'{result_dir}/yolov5_train.out'

    message = ''
    message_color = ''
    if os.path.isdir(trainYoloUrl):
        train_yolo_img_list = [f for f in os.listdir(f'{trainYoloUrl}') if f.endswith('.jpg')]
        print(train_yolo_img_list)
        train_yolo_class_file_flag = os.path.isfile(
            f'{trainYoloUrl}/classes.txt')
        if train_yolo_class_file_flag:
            train_yolo_label_file = [f for f in os.listdir(f'{trainYoloUrl}') if f.endswith(
                '.txt')]
            train_yolo_label_file.remove('classes.txt')
            print(train_yolo_label_file)
            if len(train_yolo_img_list) > 0 and len(train_yolo_label_file) > 0:
                message = f'{len(train_yolo_img_list)} images found. {len(train_yolo_label_file)} label files found.'
                # "%s %s --jobname '%s' --result_base_dir '%s' --model_size %s --test_ratio %f --img_dir '%s' --epochs %d --batch-size %d",python,yoloTrainScript,jobname,yolomodelDir,yl_mdlsize,yl_testratio,gsub("/srv/shiny-server/Emerson","$EmersonBaseDir",yl_imgdir),yl_epochs,yl_batch_size)
                if sys.platform == "linux":
                    cmd = f"{python3} {yolo_train_script} --jobname '{jobname}' --result_base_dir '{yolo_model_folder}' --model_size {trainYoloModelSize} --test_ratio {trainYoloTestRatio} --img_dir '{trainYoloUrl}' --epochs {trainYoloEpochs} --batch-size {trainYoloBatchSize} {trainYoloOptimizer} >> {result_log_path} &"
                    print(cmd)
                    # os.system(cmd)
                    subprocess.Popen(cmd, close_fds=True, shell=True)
                    # wait a few seconds
                    time.sleep(5)
                    # check process running and print output in Linux
                    get_running_jobs_cmd = "ps aux | grep yolo5_train"
                    # if hostname == 'kuldldsccappo01.kul.apac.dell.com':
                    #     get_running_jobs_cmd = f"{get_running_jobs_cmd}"
                    jobs = subprocess.Popen(
                        get_running_jobs_cmd, shell=True, stdout=subprocess.PIPE)
                    output = set(jobs.stdout)
                    print(output)
                    job_list_full = [x.decode('utf-8').strip('\n')
                                    for x in output if 'grep' not in x.decode('utf-8').strip('\n')]
                    job_list = sorted([i.split("--")[1].split(" ")[1]
                                       for i in job_list_full])
                    print(job_list)
                else:
                    print("system platform is neither windows nor linux.")
            else:
                message = f'No images and/or label files found.'
                message_color = 'red'
        else:
            message = "classes.txt not found. Please prepare a classes.txt file with label names per row."
            message_color = 'red'
    else:
        message = 'Input a directory of images, label files and a classes.txt file.'
        message_color = 'red'

    # parse output to the format
    log_output = ''
    output = {}
    for job in job_list:
        job_log_path = f'{yolo_model_folder}/{job}/yolo5_train.out'
        print(job_log_path)
        if os.path.isfile(job_log_path):
            log_file = open(job_log_path, 'r')
            log_output = tail(log_file, 20)
        else:
            log_output = ''
        output[job] = log_output
    print(output)
    return jsonify({'yoloresult': render_template("trainYoloresult.html", job_list=job_list, output=output, message=message, message_color=message_color)})


@app.route('/trainyoloresultrefresh/', methods=['POST'])
def trainyoloresultrefresh():
    yolo_model_folder = session.get('yolo_model_folder')
    message = ''
    message_color = ''

    if sys.platform == "linux":
        # check process running and print output in Linux
        get_running_jobs_cmd = "ps aux | grep yolo5_train"
        jobs = subprocess.Popen(
            get_running_jobs_cmd, shell=True, stdout=subprocess.PIPE)
        output = set(jobs.stdout)
        print(output)
        job_list_full = [x.decode('utf-8').strip('\n')
                         for x in output if 'grep' not in x.decode('utf-8').strip('\n')]
        job_list = sorted([i.split("--")[1].split(" ")[1] for i in job_list_full])
        print(job_list)
    else:
        print("system platform is neither windows nor linux.")

    # parse output to the format
    log_output = ''
    output = {}
    for job in job_list:
        job_log_path = f'{yolo_model_folder}/{job}/yolo5_train.out'
        print(job_log_path)
        if os.path.isfile(job_log_path):
            log_file = open(job_log_path, 'r')
            log_output = tail(log_file, 20)
        else:
            log_output = ''
        output[job] = log_output
    print(output)
    return jsonify({'yoloresultrefresh': render_template("trainYoloresult.html", job_list=job_list, output=output, message=message, message_color=message_color)})


@app.route('/cnnmodelcompare/', methods=['POST'])
def cnnmodelcompare():
    cnn_model_folder = session.get('cnn_model_folder')

    cnn_model_list = request.form.getlist('compareCnnModel')
    compareCnnModelPlotOption = request.form['compareCnnModelPlotOption']
    compareCnnModelCol = request.form.getlist('compareCnnModelCol')
    print(request.form)
    # read param.csv into one dataframe and get the column names
    parameter_df_list = []
    chart_list = []
    
    for cnn_model in cnn_model_list:
        # parameters
        parameter = pd.read_csv(
            f'{cnn_model_folder}/{cnn_model}/param.csv')
        print(f'{cnn_model_folder}/{cnn_model}/param.csv')
        colnames = parameter.columns
        print(parameter)
        parameter_df_list.append(parameter)
        # training history
        if compareCnnModelPlotOption == 'separate':
            model_trace = pd.read_csv(
                f'{cnn_model_folder}/{cnn_model}/model_trace.csv')
            print(f'{cnn_model_folder}/{cnn_model}/model_trace.csv')
            chart = plotchart(model_trace, compareCnnModelCol, cnn_model)
            chart_list.append(chart)
    params = pd.concat(parameter_df_list)
    print(params)

   
    if compareCnnModelPlotOption == 'combine':
        for col in compareCnnModelCol:
            model_trace_list = []
            for cnn_model in cnn_model_list:
                model_trace = pd.read_csv(
                    f'{cnn_model_folder}/{cnn_model}/model_trace.csv')
                df = model_trace[[compareCnnModelCol[0]]]
                print(type(df))
                # this df is a pandas series
                df.columns = [cnn_model]
                model_trace_list.append(df)
            combine_model_trace_df = pd.concat(
                model_trace_list, axis=1)
            combine_model_trace_df['xc'] = combine_model_trace_df.index
            print(combine_model_trace_df)
            title = f"{col}"
            chart = plotchart(
                combine_model_trace_df, cnn_model_list, title, chart_type='scatter')
            chart_list.append(chart)
    return jsonify({'cnnmodelcompareresult': render_template("modelresult.html", chart_list=chart_list, params=params, colnames=colnames)})

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=5000, debug=True)
    