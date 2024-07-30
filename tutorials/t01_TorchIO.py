# -*- coding: utf-8 -*-
# Modified from google colab
# - saving images, no display
# - use local datasets 


# prepare a working segmentation pipeline
# use medSeg models
# apply on the challenge dataset

"""Import modules:"""
# Commented out IPython magic to ensure Python compatibility.
import os, sys, enum, time, random, multiprocessing
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
#from IPython import display
from tqdm.auto import tqdm

import torch
import torchvision
import torchio as tio
import torch.nn.functional as F
import torch.nn as nn
from unet import UNet
import numpy as np



# Config
# ============== Config ===================
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

reset_results         = 0
doComputeHistogram    = 0
useWholeImages        = 0 

num_epochs            = 2
epoch_save_count      = 1

in_channels           = 1 # number of input channel e.g. for rgb = 3 
out_channels          = 2 # number of classes 
patch_size            = 0 #24
training_batch_size   = 16 #16
validation_batch_size = 16 #32

threshold             = 0.5
seed                  = 42  # for reproducibility
training_split_ratio = 0.9  # use 90% of samples for training, 10% for testing

dataset_url              = 'https://www.dropbox.com/s/ogxjwjxdv5mieah/ixi_tiny.zip?dl=0'
dataset_path             = 'data/ixi_tiny.zip'
dataset_dir_name         = 'data/ixi_tiny'
dataset_dir              = Path(dataset_dir_name)

histogram_landmarks_path = 'data/landmarks.npy'


# If the following values are False, the models will be downloaded and not computed
compute_histograms = False
train_whole_images = True # False = no trianing, use pretrainedmodel 
train_patches      = True # False = no trianing, use pretrainedmodel 


def plot_histogram(axis, tensor, num_positions=100, label=None, alpha=0.05, color=None):
    values = tensor.numpy().ravel()
    kernel = stats.gaussian_kde(values)
    positions = np.linspace(values.min(), values.max(), num=num_positions)
    histogram = kernel(positions)
    kwargs = dict(linewidth=1, color='black' if color is None else color, alpha=alpha)
    if label is not None:
        kwargs['label'] = label
    axis.plot(positions, histogram, **kwargs)

def plot_times(axis, losses, label):
    from datetime import datetime
    times, losses = losses.transpose(1, 0, 2)
    times = [datetime.fromtimestamp(x) for x in times.flatten()]
    axis.plot(times, losses.flatten(), label=label)

random.seed(seed)
torch.manual_seed(seed)
# %config InlineBackend.figure_format = 'retina'
num_workers = 0 # multiprocessing.cpu_count()
print("num_workers : ",num_workers)

plt.rcParams['figure.figsize'] = 12, 6
print('Last run on', time.ctime())
print('TorchIO version:', tio.__version__)

images_dir = dataset_dir / 'image'
labels_dir = dataset_dir / 'label'
image_paths = sorted(images_dir.glob('*.nii.gz'))
label_paths = sorted(labels_dir.glob('*.nii.gz'))

assert len(image_paths) == len(label_paths)


subjects = []
for (image_path, label_path) in zip(image_paths, label_paths):
    subject = tio.Subject(
        mri=tio.ScalarImage(image_path),
        brain=tio.LabelMap(label_path),
    )
    subjects.append(subject)
dataset = tio.SubjectsDataset(subjects)
print('Dataset size:', len(dataset), 'subjects')

"""Let's take a look at one of the subjects in the dataset."""
one_subject = dataset[0]
print(type(one_subject))
output_path = "data/one_subject.png"
one_subject.plot(show=False,output_path=output_path)

print(one_subject)
print(one_subject.mri)
print(one_subject.brain)

paths = image_paths


if compute_histograms:    
    fig, ax = plt.subplots(dpi=100)
    sTm = time.time()
    print("Preparing hist_original ...............")
    for path in tqdm(paths):
        tensor = tio.ScalarImage(path).data
        if 'HH' in path.name: color = 'red'
        elif 'Guys' in path.name: color = 'green'
        elif 'IOP' in path.name: color = 'blue'
        plot_histogram(ax, tensor, color=color)
    ax.set_xlim(-100, 2000)
    ax.set_ylim(0, 0.004);
    ax.set_title('Original histograms of all samples')
    ax.set_xlabel('Intensity')
    ax.grid()
    fig.savefig("data/hist_original.png")
    print("time: ", time.time()-sTm)
landmarks = tio.HistogramStandardization.train(
    image_paths,
    output_path=histogram_landmarks_path,
)
np.set_printoptions(suppress=True, precision=3)
print('\nTrained landmarks:', landmarks)

landmarks_dict = {'mri': landmarks}
histogram_transform = tio.HistogramStandardization(landmarks_dict)

if compute_histograms:
    fig, ax = plt.subplots(dpi=100)
    print("Preparing hist_standard ...............")
    sTm = time.time()
    for i ,sample in enumerate(tqdm(dataset)):
        standard = histogram_transform(sample)
        tensor = standard.mri.data
        path = str(sample.mri.path)
        if 'HH' in path: color = 'red'
        elif 'Guys' in path: color = 'green'
        elif 'IOP' in path: color = 'blue'
        plot_histogram(ax, tensor, color=color)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 0.02)
    ax.set_title('Intensity values of all samples after histogram standardization')
    ax.set_xlabel('Intensity')
    ax.grid()
    fig.savefig("data/hist_standard.png")
    print("time: ", time.time()-sTm)

#TODO: it seems it is not used!!!
znorm_transform = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
transform = tio.Compose([histogram_transform, znorm_transform])
sample = dataset[0]
znormed = transform(sample)
fig, ax = plt.subplots(dpi=100)
plot_histogram(ax, znormed.mri.data, label='Z-normed', alpha=1)
ax.set_title('Intensity values of one sample after z-normalization')
ax.set_xlabel('Intensity')
#ax.grid()
fig.savefig("data/hist_znormed.png")




"""## Training a network


"""

training_transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(4),
    tio.CropOrPad((48, 60, 48)),
    tio.RandomMotion(p=0.2),
    tio.HistogramStandardization({'mri': landmarks}),
    tio.RandomBiasField(p=0.3),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.RandomNoise(p=0.5),
    tio.RandomFlip(),
    tio.OneOf({
        tio.RandomAffine(): 0.8,
        tio.RandomElasticDeformation(): 0.2,
    }),
    tio.OneHot(),
])

validation_transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(4),
    tio.CropOrPad((48, 60, 48)),
    tio.HistogramStandardization({'mri': landmarks}),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.OneHot(),
])

num_subjects = len(dataset)
num_training_subjects   = int(training_split_ratio * num_subjects)
num_validation_subjects = num_subjects - num_training_subjects

num_split_subjects = num_training_subjects, num_validation_subjects
training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects)

training_set   = tio.SubjectsDataset(training_subjects, transform=training_transform)
validation_set = tio.SubjectsDataset(validation_subjects, transform=validation_transform)

print('Training set  : ', len(training_set), 'subjects')
print('Validation set: ', len(validation_set), 'subjects')

"""### Deep learning stuff"""
CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4

class Action(enum.Enum):
    TRAIN    = 'Training'
    VALIDATE = 'Validation'

def prepare_batch(batch, device):
    inputs  = batch['mri'][tio.DATA].to(device)
    targets = batch['brain'][tio.DATA].to(device)
    return inputs, targets

def get_dice_score(output, target, epsilon=1e-9):
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    dice_score = num / denom
    return dice_score

def get_dice_loss(output, target):
    return 1 - get_dice_score(output, target)

def get_model_and_optimizer(device):
    model = UNet(
        in_channels=1,
        out_classes=2,
        dimensions=3,
        num_encoding_blocks=3,
        out_channels_first_layer=8,
        normalization='batch',
        upsampling_type='linear',
        padding=True,
        activation='PReLU',
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    return model, optimizer

def run_epoch(epoch_idx, action, loader, model, optimizer):
    is_training = action == Action.TRAIN
    epoch_losses = []
    times = []
    model.train(is_training)
    for batch_idx, batch in enumerate(tqdm(loader)):
        inputs, targets = prepare_batch(batch, device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(is_training):
            logits = model(inputs)
            probabilities = F.softmax(logits, dim=CHANNELS_DIMENSION)
            batch_losses = get_dice_loss(probabilities, targets)
            batch_loss = batch_losses.mean()
            if is_training:
                batch_loss.backward()
                optimizer.step()
            times.append(time.time())
            epoch_losses.append(batch_loss.item())
    epoch_losses = np.array(epoch_losses)
    print(f'{action.value} mean loss: {epoch_losses.mean():0.3f}')
    return times, epoch_losses

def train(num_epochs, training_loader, validation_loader, model, optimizer, weights_stem):
    train_losses = []
    val_losses = []
    val_losses.append(run_epoch(0, Action.VALIDATE, validation_loader, model, optimizer))
    for epoch_idx in range(1, num_epochs + 1):
        print('Starting epoch', epoch_idx)
        train_losses.append(run_epoch(epoch_idx, Action.TRAIN, training_loader, model, optimizer))
        val_losses.append(run_epoch(epoch_idx, Action.VALIDATE, validation_loader, model, optimizer))
        #modelName = "whileImage" if useWhole else "patchImage"
        torch.save(model.state_dict(), f'results/res_{weights_stem}_epoch_{epoch_idx}.pth')
    return np.array(train_losses), np.array(val_losses)


def trainWholeImage():
    training_instance = training_set[42]  # transform is applied inside SubjectsDataset
    output_path = "results/res_training_instance.png"
    training_instance.plot(show=False,output_path=output_path)

    validation_instance = validation_set[42]
    output_path = "results/res_validation_instance.png"
    validation_instance.plot(show=False,output_path=output_path)

    print( "### Whole images ========================")

    training_batch_size = 16
    validation_batch_size = 2 * training_batch_size

    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=training_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=validation_batch_size,
        num_workers=num_workers,
    )

    print("Visualize axial slices of one batch........")

    one_batch = next(iter(training_loader))

    k = 24
    batch_mri  = one_batch['mri'][tio.DATA][..., k]
    batch_label = one_batch['brain'][tio.DATA][:, 1:, ..., k]
    slices = torch.cat((batch_mri, batch_label))
    image_path = 'results/res_batch_whole_images.png'
    torchvision.utils.save_image(
        slices,
        image_path,
        nrow=training_batch_size//2,
        normalize=True,
        scale_each=True,
        padding=0,
    )
    #display.Image(image_path)

    print("#### train_whole_images  ==================")
    model, optimizer = get_model_and_optimizer(device)
    weights_stem = 'whole_images'
    weights_path = "results/model_"+weights_stem+ "_state_dict.pth"
    if train_whole_images:
        train_losses, val_losses = train(num_epochs, training_loader, validation_loader, model, optimizer, weights_stem)
        checkpoint = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'weights': model.state_dict(),
        }
        torch.save(checkpoint, weights_path)
    else:
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['weights'])
        train_losses, val_losses = checkpoint['train_losses'], checkpoint['val_losses']
 
    fig, ax = plt.subplots()
    plot_times(ax, train_losses, 'Training')
    plot_times(ax, val_losses  , 'Validation')
    ax.grid()
    ax.set_xlabel('Time')
    ax.set_ylabel('Dice loss')
    ax.set_title('Training with whole images')
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig('results/res_losses_whole_images.png')


    print(" #### Test ========================")

    batch = next(iter(validation_loader))
    model.eval()
    inputs, targets = prepare_batch(batch, device)
    FIRST = 0
    FOREGROUND = 1
    with torch.no_grad():
        probabilities = model(inputs).softmax(dim=1)[:, FOREGROUND:].cpu()
    affine = batch['mri'][tio.AFFINE][0].numpy()
    subject = tio.Subject(
        mri=tio.ScalarImage(tensor=batch['mri'][tio.DATA][FIRST], affine=affine),
        label=tio.LabelMap(tensor=batch['brain'][tio.DATA][FIRST], affine=affine),
        predicted=tio.ScalarImage(tensor=probabilities[FIRST], affine=affine),
    )
    output_path = "results/res_subject_whole_images.png"
    subject.plot(figsize=(9, 8), cmap_dict={'predicted': 'RdBu_r'},show=False,output_path=output_path)
    for key in subject.keys():
       if isinstance(subject[key], tio.Image):
          print(f"{key} dtype:", subject[key].data.dtype)
          print(f"{key} range:", subject[key].data.min(), subject[key].data.max())

def trainPatchImage():

    training_batch_size   = 32
    validation_batch_size = 2 * training_batch_size

    patch_size            = 24
    samples_per_volume    = 5
    max_queue_length      = 300
    sampler = tio.data.UniformSampler(patch_size)

    patches_training_set = tio.Queue(
        subjects_dataset=training_set,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    patches_validation_set = tio.Queue(
        subjects_dataset=validation_set,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_subjects=False,
        shuffle_patches=False,
    )

    training_loader_patches   = torch.utils.data.DataLoader(patches_training_set, batch_size=training_batch_size)
    validation_loader_patches = torch.utils.data.DataLoader(patches_validation_set, batch_size=validation_batch_size)

    one_batch = next(iter(training_loader_patches))
    k = int(patch_size // 4)
    batch_mri = one_batch['mri'][tio.DATA][..., k]
    batch_label = one_batch['brain'][tio.DATA][:, 1:, ..., k]
    slices = torch.cat((batch_mri, batch_label))
    image_path = 'results/batch_patches.png'
    torchvision.utils.save_image(
        slices,
        image_path,
        nrow=training_batch_size,
        normalize=True,
        scale_each=True,
    )
    #display.Image(image_path)

    model, optimizer = get_model_and_optimizer(device)
    weights_stem = 'patches'
    weights_path = "results/model_"+weights_stem+ "_state_dict.pth"
    
    if train_patches:
        train_losses, val_losses = train(
            num_epochs,
            training_loader_patches,
            validation_loader_patches,
            model,
            optimizer,
            weights_stem,
        )
        checkpoint = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'weights': model.state_dict(),
        }
        torch.save(checkpoint, weights_path)
    else:
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['weights'])
        train_losses, val_losses = checkpoint['train_losses'], checkpoint['val_losses']
    fig, ax = plt.subplots()
    plot_times(ax, train_losses, 'Training')
    plot_times(ax, val_losses, 'Validation')
    ax.grid()
    ax.set_xlabel('Time')
    ax.set_ylabel('Dice loss')
    ax.set_title('Training with patches (subvolumes)')
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig('results/res_losses_patch_images.png')

    subject = random.choice(validation_set)
    input_tensor = sample.mri.data[0]
    patch_size = 48, 48, 48  # we can user larger patches for inference
    patch_overlap = 4, 4, 4
    grid_sampler = tio.inference.GridSampler(
        subject,
        patch_size,
        patch_overlap,
    )
    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=validation_batch_size)
    aggregator = tio.inference.GridAggregator(grid_sampler)

    model.eval()
    with torch.no_grad():
        for patches_batch in patch_loader:
            inputs = patches_batch['mri'][tio.DATA].to(device)
            locations = patches_batch[tio.LOCATION]
            probabilities = model(inputs).softmax(dim=CHANNELS_DIMENSION)
            aggregator.add_batch(probabilities, locations)

    foreground = aggregator.get_output_tensor()
    affine = subject.mri.affine
    prediction = tio.ScalarImage(tensor=foreground, affine=affine)
    subject.add_image(prediction, 'prediction')
    output_path = "results/res_subject_patch.png"
    subject.plot(figsize=(9, 8), cmap_dict={'prediction': 'RdBu_r'},show=False,output_path=output_path)
    for key in subject.keys():
       if isinstance(subject[key], tio.Image):
          print(f"{key} dtype:", subject[key].data.dtype)
          print(f"{key} range:", subject[key].data.min(), subject[key].data.max())

def main(useWholeImages=useWholeImages):
    print("useWholeImages: ",useWholeImages)
    if useWholeImages:
       trainWholeImage()
    else:
       trainPatchImage()
if __name__ == '__main__':
    print("useWholeImages: ",useWholeImages)
    main(useWholeImages=useWholeImages)
    