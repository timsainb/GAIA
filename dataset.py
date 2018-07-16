import numpy as np
import requests
import matplotlib.pyplot as plt
import cv2 # normalizing images - this isn't really necessary and wasn't used in the results from the paper
import PIL # images
from PIL import Image
import scipy.ndimage # for filtering
from skimage.restoration import denoise_nl_means, estimate_sigma # some denoising, again not needed, and not used in paper
from tqdm import tqdm_notebook as tqdm
import os
import glob

import hashlib
import bz2
import zipfile
import base64
import cryptography.hazmat.primitives.hashes
import cryptography.hazmat.backends
import cryptography.hazmat.primitives.kdf.pbkdf2
import cryptography.fernet


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None
def save_response_content(response, destination, chunk_size=32 * 1024):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size), total=total_size,
                          unit='B', unit_scale=True, desc=destination):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
def rot90(v):
        return np.array([-v[1], v[0]])

def process_img(path, lm, size = 128, verbose = False):
    img = PIL.Image.open(path)
    eye_avg = (lm[0] + lm[1]) * 0.5 + 0.5 # middle of eyes
    mouth_avg = (lm[3] + lm[4]) * 0.5 + 0.5 # middle of mouth
    eye_to_eye = lm[1] - lm[0] # distance between eyes
    eye_to_mouth = mouth_avg - eye_avg # distance between mouth and eyes
    x = eye_to_eye - rot90(eye_to_mouth)
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = rot90(x)
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    zoom = size / (np.hypot(*x) * 2)

    eye_to_nose = np.linalg.norm(eye_avg - lm[2])
    mouth_to_nose = np.linalg.norm(mouth_avg - lm[2])
    left_half_face_mean = np.mean((lm[0],lm[3]), axis=0)
    right_half_face_mean = np.mean((lm[1],lm[4]), axis=0)
    left_to_nose = np.linalg.norm(left_half_face_mean - lm[2])
    right_to_nose = np.linalg.norm(right_half_face_mean - lm[2])

    top_bottom = eye_to_nose/np.linalg.norm(lm[0] - lm[1])
    left_right = left_to_nose/right_to_nose



    if verbose:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(6,6))
        ax.imshow(img)
        plt.scatter(lm[:,0], lm[:,1])
        #0 = left eye, 1 = right eye, 2=nose, 3 = left mouth, 4 = right mouth, 5 = nose
        plt.scatter(left_half_face_mean[0], left_half_face_mean[1])
        plt.scatter(right_half_face_mean[0], right_half_face_mean[1])
        plt.scatter(eye_avg[0], eye_avg[1])
        plt.scatter(mouth_avg[0], mouth_avg[1])
        plt.show()
        print('Top-bottom', top_bottom, 'left right', left_right)
        print(np.shape(img))
        print(lm)

    # Shrink.
    if verbose:print('shrink')
    shrink = int(np.floor(0.5 / zoom))
    if shrink > 1:
        resize = (int(np.round(float(img.size[0]) / shrink)), int(np.round(float(img.size[1]) / shrink)))
        img = img.resize(resize, PIL.Image.ANTIALIAS)
        quad /= shrink
        zoom *= shrink
    if verbose:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(6,6))
        ax.imshow(img)
        plt.show()

    # Crop.
    if verbose: print('crop')
    border = max(int(np.round(size * 0.1 / zoom)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]
    if verbose:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(6,6))
        ax.imshow(img)
        plt.show()

    # Simulate super-resolution.
    if verbose:print('superres')
    superres = int(np.exp2(np.ceil(np.log2(zoom))))
    if superres > 1:
        img = img.resize((img.size[0] * superres, img.size[1] * superres), PIL.Image.ANTIALIAS)
        quad *= superres
        zoom /= superres

    if verbose:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(6,6))
        ax.imshow(img)
        plt.show()

    # Pad.
    if verbose:print('pad')
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if max(pad) > border - 4:
        pad = np.maximum(pad, int(np.round(size * 0.3 / zoom)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.mgrid[:h, :w, :1]
        mask = 1.0 - np.minimum(np.minimum(np.float32(x) / pad[0], np.float32(y) / pad[1]), np.minimum(np.float32(w-1-x) / pad[2], np.float32(h-1-y) / pad[3]))
        blur = size * 0.02 / zoom
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.round(img), 0, 255)), 'RGB')
        quad += pad[0:2]

    if verbose:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(6,6))
        print(np.shape(img))
        ax.imshow(img)
        plt.show()

    # Transform.
    if verbose: print('transform')
    img = img.transform((4096, 4096), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    img = img.resize((size, size), PIL.Image.ANTIALIAS)
    img = np.asarray(img)#.transpose(2, 0, 1)
    if verbose:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(6,6))
        ax.imshow(img)
        plt.show()


    return img.astype('uint8').flatten(), top_bottom, left_right



def process_img_celeb_hq(celeba_dir,delta_dir,idx, lm, orig_idx, orig_file, proc_md5, final_md5, final_size = 128, verbose = False):
    full_size = 1024
    orig_path = os.path.join(celeba_dir, 'img_celeba', orig_file)
    img = PIL.Image.open(orig_path)
    eye_avg = (lm[0] + lm[1]) * 0.5 + 0.5 # middle of eyes
    mouth_avg = (lm[3] + lm[4]) * 0.5 + 0.5 # middle of mouth
    eye_to_eye = lm[1] - lm[0] # distance between eyes
    eye_to_mouth = mouth_avg - eye_avg # distance between mouth and eyes
    x = eye_to_eye - rot90(eye_to_mouth)
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = rot90(x)
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    zoom = full_size / (np.hypot(*x) * 2)

    # get info about rotation of face
    eye_to_nose = np.linalg.norm(eye_avg - lm[2])
    mouth_to_nose = np.linalg.norm(mouth_avg - lm[2])
    left_half_face_mean = np.mean((lm[0],lm[3]), axis=0)
    right_half_face_mean = np.mean((lm[1],lm[4]), axis=0)
    left_to_nose = np.linalg.norm(left_half_face_mean - lm[2])
    right_to_nose = np.linalg.norm(right_half_face_mean - lm[2])

    top_bottom = eye_to_nose/np.linalg.norm(lm[0] - lm[1])
    left_right = left_to_nose/right_to_nose



    if verbose:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(6,6))
        ax.imshow(img)
        plt.scatter(lm[:,0], lm[:,1])
        #0 = left eye, 1 = right eye, 2=nose, 3 = left mouth, 4 = right mouth, 5 = nose
        plt.scatter(left_half_face_mean[0], left_half_face_mean[1])
        plt.scatter(right_half_face_mean[0], right_half_face_mean[1])
        plt.scatter(eye_avg[0], eye_avg[1])
        plt.scatter(mouth_avg[0], mouth_avg[1])
        plt.show()
        print('Top-bottom', top_bottom, 'left right', left_right)
        print(np.shape(img))
        print(lm)

    # Shrink.
    if verbose:print('shrink')
    shrink = int(np.floor(0.5 / zoom))
    if shrink > 1:
        size = (int(np.round(float(img.size[0]) / shrink)), int(np.round(float(img.size[1]) / shrink)))
        img = img.resize(size, PIL.Image.ANTIALIAS)
        quad /= shrink
        zoom *= shrink

    if verbose:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(6,6))
        ax.imshow(img)
        plt.show()

    # Crop.
    if verbose: print('crop')
    border = max(int(np.round(1024 * 0.1 / zoom)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    if verbose:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(6,6))
        ax.imshow(img)
        plt.show()

    # Simulate super-resolution.
    if verbose:print('superres')
    superres = int(np.exp2(np.ceil(np.log2(zoom))))
    if superres > 1:
        img = img.resize((img.size[0] * superres, img.size[1] * superres), PIL.Image.ANTIALIAS)
        quad *= superres
        zoom /= superres

    if verbose:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(6,6))
        ax.imshow(img)
        plt.show()

    # Pad.
    if verbose:print('pad')
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if max(pad) > border - 4:
        pad = np.maximum(pad, int(np.round(1024 * 0.3 / zoom)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.mgrid[:h, :w, :1]
        mask = 1.0 - np.minimum(np.minimum(np.float32(x) / pad[0], np.float32(y) / pad[1]), np.minimum(np.float32(w-1-x) / pad[2], np.float32(h-1-y) / pad[3]))
        blur = 1024 * 0.02 / zoom
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.round(img), 0, 255)), 'RGB')
        quad += pad[0:2]

    if verbose:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(6,6))
        print(np.shape(img))
        ax.imshow(img)
        plt.show()

    # Transform.
    if verbose: print('transform')
    img = img.transform((4096, 4096), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    img = img.resize((full_size, full_size), PIL.Image.ANTIALIAS)
    img = (np.asarray(img).transpose(2, 0, 1))


    if verbose:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(6,6))
        ax.imshow(img.reshape((1024,1024,3)))
        plt.show()

    # Verify MD5.
    md5 = hashlib.md5()
    md5.update(img.tobytes())
    if md5.hexdigest() != proc_md5:
        print('Skipping file because hex did not Match')
        print((md5.hexdigest()), (proc_md5))
        return None, None, None, None
    assert md5.hexdigest() == proc_md5

    # Load delta image and original JPG.
    with zipfile.ZipFile(os.path.join(delta_dir, 'deltas%05d.zip' % (idx - idx % 1000)), 'r') as zip:
        delta_bytes = zip.read('delta%05d.dat' % idx)
    with open(orig_path, 'rb') as file:
        orig_bytes = file.read()

    # Decrypt delta image, using original JPG data as decryption key.
    algorithm = cryptography.hazmat.primitives.hashes.SHA256()
    backend = cryptography.hazmat.backends.default_backend()
    kdf = cryptography.hazmat.primitives.kdf.pbkdf2.PBKDF2HMAC(algorithm=algorithm, length=32, salt=orig_file, iterations=100000, backend=backend)
    key = base64.urlsafe_b64encode(kdf.derive(orig_bytes))
    delta = np.frombuffer(bz2.decompress(cryptography.fernet.Fernet(key).decrypt(delta_bytes)), dtype=np.uint8).reshape(3, 1024, 1024)

    # Apply delta image.
    img = img + delta

    # Verify MD5.
    md5 = hashlib.md5()
    md5.update(img.tobytes())
    assert md5.hexdigest() == final_md5
    img = img.transpose(1, 2, 0) # CHW => HWC
    im = Image.fromarray(img)

    # final resize of the image
    if final_size != 1024:
        im = scipy.misc.imresize(im,[final_size,final_size, 3])

    if verbose:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(6,6))
        ax.imshow(im)
        plt.show()
    return im.astype('uint8').flatten(), top_bottom, left_right, orig_idx


def get_fields_landmarks(celeba_dir, delta_dir, num_threads=4, num_tasks=100):
    print('Loading CelebA data from %s' % celeba_dir)
    glob_pattern = os.path.join(celeba_dir, 'img_celeba', '*.jpg')
    glob_expected = 202599
    if len(glob.glob(glob_pattern)) != glob_expected:
        print(len(glob.glob(glob_pattern)))
        print('Error: Expected to find %d images in %s' % (glob_expected, glob_pattern))
        return
    with open(os.path.join(celeba_dir, 'Anno', 'list_landmarks_celeba.txt'), 'rt') as file:
        landmarks = [[float(value) for value in line.split()[1:]] for line in file.readlines()[2:]]
        landmarks = np.float32(landmarks).reshape(-1, 5, 2)

    print('Loading CelebA-HQ deltas from %s' % delta_dir)

    glob_pattern = os.path.join(delta_dir, 'delta*.zip')
    glob_expected = 30
    if len(glob.glob(glob_pattern)) != glob_expected:
        print('Error: Expected to find %d zips in %s' % (glob_expected, glob_pattern))
        return
    with open(os.path.join(delta_dir, 'img_list.txt'), 'rt') as file:
        lines = [line.split() for line in file]
        fields = dict()
        for idx, field in enumerate(tqdm(lines[0])):
            type = int if field.endswith('idx') else str
            fields[field] = [type(line[idx]) for line in lines[1:]]
    return fields, landmarks
