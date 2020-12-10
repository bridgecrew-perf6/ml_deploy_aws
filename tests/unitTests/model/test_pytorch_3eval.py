# # from scripts import pytorch_cnn_dup 

# # from src import pytorch_cnn_train
# from src.pytorch_cnn_train import *
# from src.pytorch_cnn_predict import *
# import subprocess
# def test_pytorch_evaluation():
#     # cmd = "python src/pytorch_cnn_train.py  --jobname vl_small_fnb_no_aug_kul  --new_data_dir 'data/processed/FnBdataSmall' --result_base_dir 'src/Output'  --epochs 50 --max_epochs_stop 10 --img_size 256 --cnn_sel 'resnet50' --batch_size 32" 

#     # cmd = f"{python3} {cnn_script_path} --jobname {jobname} --model_base_dir {cnn_model_folder} --img_dir {image_path} --result_base_dir {result_base_dir} --model_list {cnn_model_list_formatted}"
#     # print(cmd)
#     # result = subprocess.run(cmd,shell=True)

#     # avg_valid_acc
#     model, history = train(
#         model,
#         criterion,
#         optimizer,
#         dataloaders['train'],
#         dataloaders['val'],
#         save_file_name=save_file_name,
#         max_epochs_stop=max_epochs_stop,
#         n_epochs=epochs,
#         print_every=1)
#     assert avg_valid_acc > 0.90, 'Accuracy on validation should be > 0.90'
        
#     print('finish cmd cnn')