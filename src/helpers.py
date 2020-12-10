import torch
from torchvision import datasets
from torch.utils.data import Dataset
import os
from PIL import Image
# from torch.utils.data import DataLoader
# from torchvision import transforms
# img_size=256
# test_dir = "data/processedSmall/test" 
# test_dir = './app/static/uploads/v_20201209_202635'
# image_transforms = {
#     'test':
#     transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.CenterCrop(size=224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     }

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = all_imgs

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        # original_tuple = super(CustomDataSet, self).__getitem__(idx)
        # path = self.imgs[idx][0]
        # tuple_with_path = ((path,))
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        # print('img_loc',img_loc)
        image = Image.open(img_loc).convert("RGB")

        if self.transform is not None:
            print('transformed')
            tensor_image = self.transform(image)

        else:
            tensor_image = image    

        return (tensor_image,img_loc)

# Define DataLoaders
# try:
#     print('run')
#     test_dataset = ImageFolderWithPaths(root=test_dir, transform=image_transforms['test']) # our custom dataset with filenames

# except RuntimeError:
# print('run custom dataset')
# test_dataset = CustomDataSet(test_dir, transform=image_transforms['test']) # our custom dataset with filenames

# test_dataset = ImageFolderWithPaths(root=test_dir, transform=image_transforms['test'])

# dataloaders = {
#             'blind_test': DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False,num_workers=0),
#         }  
# imgFile = []
# for inputs, paths in dataloaders['blind_test']:
#     # print(inputs, paths)
#     imgFile.extend(paths)

# print(imgFile)
# for inputs, labels, paths in dataloaders['blind_test']:
    # print(inputs, labels, paths)
#     imgFile.extend(paths)
#     # use the above variables freely
# print(imgFile)
# from torch.utils.data.dataset import Dataset

# class InfDataloader(Dataset):
#     """
#     Dataloader for Inference.
#     """
#     def __init__(self, img_folder, target_size=256):
#         self.imgs_folder = img_folder

#         self.img_paths = []

#         img_path = self.imgs_folder + '/'
#         img_list = os.listdir(img_path)
#         img_list.sort()
#         img_list.sort(key=lambda x: int(x[:-4]))  ##文件名按数字排序
#         img_nums = len(img_list)
#         for i in range(img_nums):
#             img_name = img_path + img_list[i]
#             self.img_paths.append(img_name)

#         # self.img_paths = sorted(glob.glob(self.imgs_folder + '/*'))

#         print(self.img_paths)


#         self.target_size = target_size
#         self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                               std=[0.229, 0.224, 0.225])

#     def __getitem__(self, idx):
#         """
#         __getitem__ for inference
#         :param idx: Index of the image
#         :return: img_np is a numpy RGB-image of shape H x W x C with pixel values in range 0-255.
#         And img_tor is a torch tensor, RGB, C x H x W in shape and normalized.
#         """
#         img = cv2.imread(self.img_paths[idx])
#         name = self.img_paths[idx]

#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Pad images to target size
#         img_np = pad_resize_image(img, None, self.target_size)
#         img_tor = img_np.astype(np.float32)
#         img_tor = img_tor / 255.0
#         img_tor = np.transpose(img_tor, axes=(2, 0, 1))
#         img_tor = torch.from_numpy(img_tor).float()
#         img_tor = self.normalize(img_tor)

#         return img_np, img_tor, name