import os
import numpy as np
from torch.utils.data import Dataset
import torch
import glob
from PIL import Image
from torchvision import transforms
import cv2
from data.perlin import rand_perlin_2d_np

anomal_p = 0.03

def passing_mvtec_argument(args):
    global argument

    argument = args

class TestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.png"))
        self.resize_shape=resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample

class TrainDataset(Dataset):

    def __init__(self,
                 root_dir,
                 anomaly_source_path=None,
                 resize_shape=None,
                 tokenizer=None,
                 caption : str = None,
                 latent_res : int = 64) :

        # [1] base image
        self.root_dir = root_dir
        image_paths, gt_paths, object_masks = [], [],[]
        folders = os.listdir(self.root_dir)
        for folder in folders :
            repeat, cat = folder.split('_')
            folder_dir = os.path.join(self.root_dir, folder)
            rgb_folder = os.path.join(folder_dir, 'xray')
            gt_folder = os.path.join(folder_dir, 'gt')
            object_folder = os.path.join(folder_dir, 'teeth')
            images = os.listdir(rgb_folder)
            for image in images :
                for _ in range(int(repeat)) :
                    image_path = os.path.join(rgb_folder, image)
                    image_paths.append(image_path)
                    gt_paths.append(os.path.join(gt_folder, image))
                    object_masks.append(os.path.join(object_folder, image))

        self.resize_shape=resize_shape
        self.caption = caption
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5]),])
        self.image_paths = image_paths
        self.gt_paths = gt_paths
        self.object_masks = object_masks
        self.latent_res = latent_res

        if anomaly_source_path is not None:
            self.anomaly_source_paths = []
            for ext in ["png", "jpg"]:
                self.anomaly_source_paths.extend(sorted(glob.glob(anomaly_source_path + f"/*/*/*.{ext}")))
        else:
            self.anomaly_source_paths = []

    def __len__(self):

        return len(self.image_paths)

    def torch_to_pil(self, torch_img):
        # torch_img = [3, H, W], from -1 to 1
        np_img = np.array(((torch_img + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
        pil = Image.fromarray(np_img)

    def augment_image(self, image, anomaly_source_img,
                      min_perlin_scale, max_perlin_scale,
                      min_beta_scale, max_beta_scale,
                      object_position, trg_beta):

        # [2] perlin noise
        while True :

            while True :
                # [1] size of noise :big perlin scale means smaller noise
                perlin_scalex = 2 ** (torch.randint(min_perlin_scale, max_perlin_scale, (1,)).numpy()[0])
                perlin_scaley = 2 ** (torch.randint(min_perlin_scale, max_perlin_scale, (1,)).numpy()[0])
                perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]),
                                                 (perlin_scalex, perlin_scaley))
                threshold = 0.3
                perlin_thr = np.where(perlin_noise > threshold,
                                      np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
                # smoothing
                perlin_thr = cv2.GaussianBlur(perlin_thr, (3,3), 0)
                # only on object
                total_object_pixel = int(self.resize_shape[0] * self.resize_shape[1])
                if object_position is not None:
                    total_object_pixel = np.sum(object_position)
                    perlin_thr = perlin_thr * object_position
                binary_2D_mask = (np.where(perlin_thr == 0, 0, 1)).astype(np.float32)  # [512,512,3]
                if np.sum(binary_2D_mask) > anomal_p * total_object_pixel :
                    break
            blur_3D_mask = np.expand_dims(perlin_thr, axis=2)  # [512,512,3]

            if trg_beta is None :
                while True :
                    # [1] how transparent the noise
                    # small = really transparent
                    beta = torch.rand(1).numpy()[0]
                    if max_beta_scale > beta > min_beta_scale :
                        break
            else:
                beta = trg_beta

            # big beta = transparent
            # if beta = 0 :only anomal
            # if beta = 1 -> image
            A = beta * image + (1 - beta) * anomaly_source_img.astype(np.float32) # merged
            augmented_image = (image * (1 - blur_3D_mask) + A * blur_3D_mask).astype(np.float32)
            anomal_img = np.array(Image.fromarray(augmented_image.astype(np.uint8)), np.uint8)

            binary_2d_pil = Image.fromarray((binary_2D_mask * 255).astype(np.uint8)).convert('L').resize((64, 64))
            anomal_mask_torch = torch.where((torch.tensor(np.array(binary_2d_pil)) / 255) > 0.5, 1, 0)
            if anomal_mask_torch.sum() > 0:
                break
        return anomal_img, anomal_mask_torch

    def get_input_ids(self, caption):
        tokenizer_output = self.tokenizer(caption, padding="max_length", truncation=True,return_tensors="pt")
        input_ids = tokenizer_output.input_ids
        attention_mask = tokenizer_output.attention_mask
        return input_ids, attention_mask

    def load_image(self, image_path, trg_h, trg_w, type='RGB'):
        image = Image.open(image_path)
        if type == 'RGB' :
            if not image.mode == "RGB":
                image = image.convert("RGB")
        elif type == 'L':
            if not image.mode == "L":
                image = image.convert("L")
        if trg_h and trg_w:
            image = image.resize((trg_w, trg_h), Image.BICUBIC)
        img = np.array(image, np.uint8)
        return img

    def __getitem__(self, idx):

        # [1] base
        img_idx = idx % len(self.image_paths)
        img_path = self.image_paths[img_idx]
        img = self.load_image(img_path, self.resize_shape[0], self.resize_shape[1],type='RGB')  # np.array,

        # [2] gt dir
        gt_path = self.gt_paths[img_idx]
        gt_img = np.array(Image.open(gt_path).convert('L').resize((self.latent_res, self.latent_res),Image.BICUBIC)) # 64,64
        gt_torch = torch.tensor(gt_img) / 255
        gt_torch = torch.where(gt_torch>0, 1, 0).unsqueeze(0)

        # [3] generate pseudo anomal
        teeth_path = self.object_masks[img_idx]
        teeth_img = self.load_image(teeth_path, self.resize_shape[0], self.resize_shape[1],type='L')  # np.array,
        teeth_np = np.array(teeth_img)/ 255
        background_position = np.where(teeth_np > 0.5, 0, 1)

        # [4]
        new_np = np.zeros_like(img)
        new_np[:, :, 0] = self.load_image(img_path, self.resize_shape[0], self.resize_shape[1],type='L')  # np.array,
        if argument.rgb_train :
            new_np[:, :, 1] = teeth_img
        else :
            #new_np[:, :, 1] = self.load_image(img_path, self.resize_shape[0], self.resize_shape[1],type='L')  # np.array,
            #new_np[:, :, 2] = self.load_image(img_path, self.resize_shape[0], self.resize_shape[1], type='L')  # np.array,
            new_np = self.load_image(img_path, self.resize_shape[0], self.resize_shape[1], type='RGB')  # np.array,
        rgb_pil = np.array(Image.fromarray(new_np.astype(np.uint8)).convert('RGB'))

        # [5] make pseudo anomal
        is_ok = 0
        anomal_pil = rgb_pil
        anomal_mask_torch = gt_torch
        if gt_torch.sum() == 0 :
            is_ok = 1
            """ normal sample, make pseudo sample"""
            anomal_src_idx = idx % len(self.anomaly_source_paths)
            anomal_dir = self.anomaly_source_paths[anomal_src_idx]
            pseudo_np = self.load_image(anomal_dir,
                                        self.resize_shape[0], self.resize_shape[1],type='RGB')
            # [2] mask
            anomal_img, anomal_mask_torch = self.augment_image(img, pseudo_np,
                                                               argument.min_perlin_scale,
                                                               argument.max_perlin_scale,
                                                               argument.min_beta_scale,
                                                               argument.max_beta_scale,
                                                               background_position,
                                                               argument.trg_beta)
            anomal_np = np.zeros_like(anomal_img)
            anomal_np[:, :, 0] = anomal_img[:,:,0]
            if argument.rgb_train:
                anomal_np[:, :, 1] = teeth_img
            else :
                anomal_np = anomal_img
            anomal_pil = np.array(Image.fromarray(anomal_np.astype(np.uint8)).convert('RGB'))

        if self.tokenizer is not None :
            input_ids, attention_mask = self.get_input_ids(self.caption) # input_ids = [77]
        else :
            input_ids = torch.tensor([0])

        return {'image': self.transform(rgb_pil),               # [3,512,512]
                "gt": gt_torch,                             # [1, 64, 64]
                'input_ids': input_ids.squeeze(0),
                'is_ok' : is_ok,
                'augment_img' : self.transform(anomal_pil),
                'augment_mask' : anomal_mask_torch}