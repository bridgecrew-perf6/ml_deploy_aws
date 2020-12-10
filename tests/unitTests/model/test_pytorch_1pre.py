# from scripts.pytorch_cnn_20201019 import *
# from scripts.pytorch_cnn_dup import load_checkpoint
# from src import pytorch_cnn_20201019
# result_base_dir = 'src/Output'
# jobname='vl_test_fnb_no_aug'
# result_dir=result_base_dir+"/"+jobname
# checkpoint_path = f'{result_dir}/resnet50-transfer.pth' 
# model_choices =['googlenet','resnet50','squeezenet', 'vgg16', 'wide_resnet50_2']
# # print(model_choices)

# # model, optimizer = load_checkpoint(path=checkpoint_path)
# model_name = checkpoint_path.split('-')[0].split('/')[-1]
# assert (model_name in model_choices), "Path must have the correct model name"

# print('model',model)
# print('optimizer',optimizer)
# print('ok')
