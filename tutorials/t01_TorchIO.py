# -*- coding: utf-8 -*-
#   Binary 3D segmentation example modified from google colab
#      https://colab.research.google.com/github/fepegar/torchio-notebooks/blob/main/notebooks/TorchIO_tutorial.ipynb
#   requires: 
#     pip3 install unet 
#     pip3 install torchio
#   

# [X] Understanding the loss function 
#     - Use standard loss function 
#     - use the dice: not good for learning
# [x] Add a single image inference     
#     - input 3D image, output a segmentation + dice score if groundtruth is available
# [ ] Monitor training:
#     - generate images during the training
# [ ] Compare patch vs whole e.g. time, accuracy
#     - use different iterations, patch size
# [ ] Add support for different models e.g. medSeg models
# [ ] Add support for different datasets
# [ ] Add support for instance multi-class segmentation
#     - test on binary vertebrae datasets

"""Import modules:"""
# Commented out IPython magic to ensure Python compatibility.
import os, sys, enum, time, random, multiprocessing, json, shutil
from pathlib import Path
from scipy import stats
from scipy.ndimage import zoom

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

# ============== Config ===================
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

doTraining               = 0 # training vs inference
isDoTestSpliting         = 0 # reset the training, validation and testing datasets

lossfunctionID           = 0
reset_results            = 0
doComputeHistogram       = 0
useWholeImages           = 1

num_epochs               = 10
epoch_save_count         = 1

new_3D_size = (48, 60, 48) # images will be resized to this size

in_channels              = 1  # number of input channel e.g. for rgb = 3 
out_channels             = 2  # number of classes 
patch_size               = 0  # 24
training_batch_size      = 16 # 16
validation_batch_size    = 16 # 32

threshold                = 0.5
seed                     = 42   # for reproducibility
training_split_ratios    = [0.50, 0.40, 0.10]  # training, validation, testing 


dataset_url              = 'https://www.dropbox.com/s/ogxjwjxdv5mieah/ixi_tiny.zip?dl=0'
dataset_zip_path         = 'data/ixi_tiny.zip'
datasetPath              = 'data/ixi_tiny'
datasetTrainPath         = datasetPath + '/train'
datasetTestPath          = datasetPath + '/test'

histogram_landmarks_path = 'data/landmarks.npy'

# inference arguments
if not doTraining:
    testImages      = sorted(os.listdir(datasetTestPath+"/image"))
    inputImagePath  = os.path.join(datasetTestPath,"image",random.choice(testImages))
    inputLabelPath  = inputImagePath.replace("image","label")
    print(inputImagePath)
    print(inputLabelPath)
    epochModelWholePath  = "results/model_whole_images_epoch_7.pth"  
    finalModelWholePath  = "results/model_whole_images_state_dict.pth"  # 50 epochs, 0.990 dice  
    epochModelPatchPath  = "results/model_patch_images_epoch_7.pth"  
    finalModelPatchPath  = "results/model_patch_images_state_dict.pth"  # 50 epochs, 0.990 dice  

    inputModelPath       = finalModelWholePath

    outputSegPath   = "results/result-label.nii.gz" 

# If the following values are False, the models will be downloaded and not computed
compute_histograms       = False
train_whole_images       = True # False = no trianing, use pretrainedmodel 
train_patches            = True # False = no trianing, use pretrainedmodel 

# model parameters 
in_channels = 1
out_classes = 2
dimensions  = 3
num_encoding_blocks = 3
out_channels_first_layer=  8
normalization= "batch"
upsampling_type = "linear"
padding = True
activation = "PReLU"
resample_to = 4
crop_or_pad_size = [48, 60, 48]                
#histogram_landmarks_lst = histogram_landmarks.tolist() 

def doTestSpliting():
   print("Reset Train and Test datasets .....")
   # must be done one time or when reset 
   # remove dataset folder
   # unzip the dataset
   # create train + test folders
   shutil.rmtree(datasetPath)
   import zipfile
   with zipfile.ZipFile(dataset_zip_path, 'r') as zip_file:
        zip_file.extractall("data")

   os.mkdir(datasetTrainPath)
   os.makedirs(datasetTestPath+"/image")        
   os.makedirs(datasetTestPath+"/label")        
   shutil.move(datasetPath+"/image",datasetTrainPath)
   shutil.move(datasetPath+"/label",datasetTrainPath)

   image_paths = sorted(os.listdir(datasetTrainPath+"/image"))
   label_paths = sorted(os.listdir(datasetTrainPath+"/label"))
   subjects = [[x,y] for x,y in zip(image_paths,label_paths)]
   print("Creating test datasets ...............")
   subjects = [ [x,y] for x,y in zip(image_paths,label_paths)]
   num_subjects = len(subjects)
   num_training_subjects  = int(training_split_ratios[0] * num_subjects)
   num_testing_subjects   = num_subjects - num_training_subjects
   num_split_subjects = num_training_subjects, num_testing_subjects
   training_subjects, testing_subjects = torch.utils.data.random_split(subjects, num_split_subjects)   
   print("Moving test subjects to a separated folder!!!")
   for subject in testing_subjects:
       imgSrc = os.path.join(datasetTrainPath,"image",subject[0])
       lblSrc = os.path.join(datasetTrainPath,"label",subject[1])
       imgDst = os.path.join(datasetTestPath+"/image")
       lblDst = os.path.join(datasetTestPath+"/label")
       shutil.move(imgSrc,imgDst)
       shutil.move(lblSrc,lblDst)

   image_paths = sorted(os.listdir(datasetTestPath+"/image"))
   label_paths = sorted(os.listdir(datasetTestPath+"/label"))
   testing_subjects = [[x,y] for x,y in zip(image_paths,label_paths)]
   return testing_subjects

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

def plot_losses(xData, yData, xTitle, yTitle, xLabel,yLabel, fTitle,figOutputPath):
    fig, ax = plt.subplots()
    plot_times(ax, np.array(xData), xTitle)
    plot_times(ax, np.array(yData), yTitle)
    ax.grid()
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(fTitle)
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(figOutputPath)

random.seed(seed)
torch.manual_seed(seed)
# %config InlineBackend.figure_format = 'retina'
num_workers = 0 # multiprocessing.cpu_count()
print("num_workers : ",num_workers)

plt.rcParams['figure.figsize'] = 12, 6
print('Last run on', time.ctime())
print('TorchIO version:', tio.__version__)

histogram_landmarks = None
if doTraining:
    print("Preparaing training, validation datasets ...............")
    testing_subjects_paths = None
    if isDoTestSpliting:
       testing_subjects_paths = doTestSpliting()
    else:
       image_paths = sorted(os.listdir(datasetTestPath+"/image"))
       label_paths = sorted(os.listdir(datasetTestPath+"/label"))
       testing_subjects_paths = [[x,y] for x,y in zip(image_paths,label_paths)]

    # save complete paths     
    image_paths = sorted(os.listdir(datasetTrainPath+"/image"))
    label_paths = sorted(os.listdir(datasetTrainPath+"/label"))
    subjects = [[x,y] for x,y in zip(image_paths,label_paths)]
    num_subjects = len(subjects)
    num_training_subjects  = int(training_split_ratios[0] * num_subjects)
    num_validation_subjects   = num_subjects - num_training_subjects
    num_split_subjects = num_training_subjects, num_validation_subjects
    training_subjects_paths, validation_subjects_paths = torch.utils.data.random_split(subjects, num_split_subjects)       
    num_testing_subjects = len(testing_subjects_paths)
    assert (num_training_subjects + num_validation_subjects + num_testing_subjects == num_subjects+num_testing_subjects), "Wrong training_split_ratios, sum should be 1.0"
    training_subjects_paths   = [[os.path.join(datasetTrainPath,"image",x[0]),os.path.join(datasetTrainPath,"label",x[1])] for x in training_subjects_paths]
    validation_subjects_paths = [[os.path.join(datasetTrainPath,"image",x[0]),os.path.join(datasetTrainPath,"label",x[1])] for x in validation_subjects_paths]
    testing_subjects_paths    = [[os.path.join(datasetTestPath,"image",x[0]), os.path.join(datasetTestPath,"label",x[1])]  for x in testing_subjects_paths]


    if compute_histograms:    
        fig, ax = plt.subplots(dpi=100)
        sTm = time.time()
        print("Preparing hist_original ...............")
        for path in tqdm(image_paths):
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

    print("Computing histogram_landmarks .............")
    histogram_landmarks = None
    if not os.path.exists(histogram_landmarks_path):
        histogram_landmarks = tio.HistogramStandardization.train(
            image_paths,
            output_path=histogram_landmarks_path,
        )
    else:
        histogram_landmarks = np.load(histogram_landmarks_path)

    np.set_printoptions(suppress=True, precision=3)
    print('\nTrained landmarks:', histogram_landmarks)
    landmarks_dict = {'mri': histogram_landmarks}
    histogram_transform = tio.HistogramStandardization(landmarks_dict)
    training_transform = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(4),
        tio.CropOrPad(new_3D_size),
        tio.RandomMotion(p=0.2),
        tio.HistogramStandardization({'mri': histogram_landmarks}),
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
        tio.CropOrPad(new_3D_size),
        tio.HistogramStandardization({'mri': histogram_landmarks}),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        tio.OneHot(),
    ])
    print ("Reading images .....")
    # Note: for large datasets we need dataloader with online augmentation
    training_subjects = []
    for (image_path, label_path) in training_subjects_paths:
        subject = tio.Subject(
            mri=tio.ScalarImage(image_path),
            brain=tio.LabelMap(label_path),
        )
        training_subjects.append(subject)
    print(training_subjects[0]        )
    training_dataset = tio.SubjectsDataset(training_subjects)

    validation_subjects = []
    for (image_path, label_path) in validation_subjects_paths:
        subject = tio.Subject(
            mri=tio.ScalarImage(image_path),
            brain=tio.LabelMap(label_path),
        )
        validation_subjects.append(subject)
    validation_dataset = tio.SubjectsDataset(validation_subjects)

    testing_subjects = []
    for (image_path, label_path) in testing_subjects_paths:
        subject = tio.Subject(
            mri=tio.ScalarImage(image_path),
            brain=tio.LabelMap(label_path),
        )
        testing_subjects.append(subject)
    testing_dataset = tio.SubjectsDataset(testing_subjects)

    training_dataset   = tio.SubjectsDataset(training_subjects, transform=training_transform)
    validation_dataset = tio.SubjectsDataset(validation_subjects, transform=validation_transform)
    testing_dataset    = tio.SubjectsDataset(testing_subjects, transform=validation_transform)

    print('training_dataset size   :', len(training_dataset), 'subjects')
    print('validation_dataset size :', len(validation_dataset), 'subjects')
    print('testing_dataset size    :', len(testing_dataset), 'subjects')

    """Let's take a look at one of the subjects in the dataset."""
    one_subject = training_dataset[0]
    print(type(one_subject))
    output_path = "data/one_subject.png"
    one_subject.plot(show=False,output_path=output_path)

    print(one_subject)
    print(one_subject.mri)
    print(one_subject.brain)

    znorm_transform = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    transform = tio.Compose([histogram_transform, znorm_transform])
    sample = training_dataset[0]
    znormed = transform(sample)
    fig, ax = plt.subplots(dpi=100)
    plot_histogram(ax, znormed.mri.data, label='Z-normed', alpha=1)
    ax.set_title('Intensity values of one sample after z-normalization')
    ax.set_xlabel('Intensity')
    #ax.grid()
    fig.savefig("data/hist_znormed.png")
else:
    histogram_landmarks = np.load(histogram_landmarks_path)

config = {
        "in_channels": in_channels,
        "out_classes": out_classes,
        "dimensions": dimensions,
        "num_encoding_blocks": num_encoding_blocks,
        "out_channels_first_layer": out_channels_first_layer,
        "normalization": normalization,
        "upsampling_type": upsampling_type,
        "padding": padding,
        "activation": activation,
        "resample_to": resample_to,
        "crop_or_pad_size": crop_or_pad_size,               
        "histogram_landmarks_lst": histogram_landmarks.tolist()  # Convert numpy array to list
}
if compute_histograms:
    fig, ax = plt.subplots(dpi=100)
    print("Preparing hist_standard ...............")
    sTm = time.time()
    for i ,sample in enumerate(tqdm(training_dataset)):
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

"""## Training a network


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

def get_soft_dice_score(output, target, epsilon=1e-9):
    # Computing soft Dice loss or differentiable Dice score.
    # the value is more "forgiving" during training, allowing the model to improve gradually
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    return (num / denom).mean()

def get_dice_score(output, target, epsilon=1e-9):
    # computing the dice, input can be numpy or torch tensors
    dice_score = 0.0
    if isinstance(output, torch.Tensor): 
        output = (output > 0.5).float()
        target = target.float()
        intersection = (output * target).sum()
        union = output.sum() + target.sum()
        dice_score =  (2. * intersection + epsilon) / (union + epsilon)   
        torch.tensor(dice_score, requires_grad=True)
    else:
        output = (output > 0.5)
        intersection = np.sum(output * target)
        union = np.sum(output) + np.sum(target)
        dice_score = (2. * intersection + epsilon) / (union + epsilon)

    return dice_score

def get_dice_loss(output, target):
    dice_score = get_dice_score(output, target)
    dice_loss  =  1 - get_soft_dice_score(output, target)    
    # this works but the model is not learing 
    #dice_loss  =  1 - dice_score
    return dice_loss, dice_score 

def get_cross_entropy_loss(output, target, epsilon=1e-9):
    # Compute dice score for monitoring (using your existing function)
    dice_score = get_dice_score(output, target)
  
    # Ensure output is probabilities
    if output.dim() == target.dim():
        output = output.unsqueeze(1)

    # Check if output is already in the correct format
    if output.shape[1] == 1:
        # If output is a single channel, assume it's the probability of the positive class
        output = torch.cat([1 - output, output], dim=1)
    
    # Convert probabilities to logits
    output = torch.log(output / (1 - output + epsilon) + epsilon)
    
    # Compute binary cross-entropy loss
    bce_loss = F.binary_cross_entropy_with_logits(output[:, 1], target.float())  
    return bce_loss, dice_score

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
            batch_loss = None
            dice_score = None
            if lossfunctionID==0:
               batch_loss, dice_score = get_dice_loss(probabilities, targets)
            elif lossfunctionID==1:
                 batch_loss, dice_score = get_cross_entropy_loss(probabilities, targets)
            else:
                print("Error! Unknow lossfunctionID .....................")

            if is_training:
                batch_loss.backward()
                optimizer.step()
            times.append(time.time())
            epoch_losses.append(batch_loss.item())
    epoch_losses = np.array(epoch_losses)
    print(f'{action.value} mean loss: {epoch_losses.mean():0.3f} dice socre: {dice_score:0.3f}')
    return times, epoch_losses, dice_score

def train(num_epochs, training_loader, validation_loader, model, optimizer,  weights_stem):
    train_losses = []
    val_losses = []
    times, epoch_losses, dice_score = run_epoch(0, Action.VALIDATE, validation_loader, model, optimizer)
    dice_score     = round(dice_score.item(), 2)
    sTm = time.time()
    val_losses.append([times, epoch_losses])
    for epoch_idx in range(1, num_epochs + 1):
        print('Starting epoch', epoch_idx)
        trn_times, trn_epoch_losses, trn_dice_score = run_epoch(epoch_idx, Action.TRAIN, training_loader, model, optimizer)
        train_losses.append([trn_times, trn_epoch_losses])
        val_times, val_epoch_losses, val_dice_score = run_epoch(epoch_idx, Action.VALIDATE, validation_loader, model, optimizer)
        val_losses.append([val_times, val_epoch_losses])
        #modelName = "whileImage" if useWhole else "patchImage"
        val_dice_score = round(val_dice_score.item(), 2)
        modelPath = f'results/model_{weights_stem}_epoch_{epoch_idx}.pth'
        if val_dice_score> dice_score:
            torch.save(model.state_dict(), modelPath)
            dice_score = val_dice_score
            with open(modelPath[:-4]+'.json', 'w') as f:
                json.dump(config, f)

        plot_losses(train_losses, val_losses, 'Training', 'Validation', 'Time','Loss', 'Training with '+weights_stem,
                    'results/res_losses_'+weights_stem+'.png')

        # fig, ax = plt.subplots()
        # plot_times(ax, np.array(train_losses), 'Training')
        # plot_times(ax, np.array(val_losses)  , 'Validation')
        # ax.grid()
        # ax.set_xlabel('Time')
        # ax.set_ylabel('Loss')
        # ax.set_title('Training with '+weights_stem)
        # ax.legend()
        # fig.autofmt_xdate()
        # fig.savefig('results/res_losses_'+weights_stem+'.png')

    print("training time: ", (time.time()-sTm)/60, " Minutes") 
    return np.array(train_losses), np.array(val_losses)

def trainWholeImage():
    training_instance = training_dataset[42]  # transform is applied inside SubjectsDataset
    output_path = "results/res_training_instance.png"
    training_instance.plot(show=False,output_path=output_path)

    validation_instance = validation_dataset[42]
    output_path = "results/res_validation_instance.png"
    validation_instance.plot(show=False,output_path=output_path)

    print( "### Whole images ========================")

    training_batch_size = 16
    validation_batch_size = 2 * training_batch_size

    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=training_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
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
    modelPath = "results/model_"+weights_stem+ "_state_dict.pth"
    if train_whole_images:
        train_losses, val_losses = train(num_epochs, training_loader, validation_loader, model, optimizer, weights_stem)
        checkpoint = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'weights': model.state_dict(),
        }
        torch.save(checkpoint, modelPath)
        with open(modelPath[:-4]+'.json', 'w') as f:
             json.dump(config, f)

    else:
        checkpoint = torch.load(modelPath, map_location=device)
        model.load_state_dict(checkpoint['weights'])
        train_losses, val_losses = checkpoint['train_losses'], checkpoint['val_losses']
 
    plot_losses(train_losses, val_losses, 'Training', 'Validation', 'Time','Loss', 'Training with '+weights_stem,
                    'results/res_losses_'+weights_stem+'.png')

    print(" #### Test ========================")
    testing_batch_size = 2
    testing_loader = torch.utils.data.DataLoader(
        testing_dataset,
        batch_size=testing_batch_size,
        num_workers=num_workers,
    )
    batch = next(iter(testing_loader))
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
        subjects_dataset=training_dataset,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    patches_validation_set = tio.Queue(
        subjects_dataset=validation_dataset,
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
    modelPath    = "results/model_"+weights_stem+ "_state_dict.pth"
    
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
        torch.save(checkpoint, modelPath)

    else:
        checkpoint = torch.load(modelPath, map_location=device)
        model.load_state_dict(checkpoint['weights'])        
        train_losses, val_losses = checkpoint['train_losses'], checkpoint['val_losses']

    plot_losses(train_losses, val_losses, 'Training', 'Validation', 'Time','Loss', 'Training with '+weights_stem,
                    'results/res_losses_'+weights_stem+'.png')

    subject = random.choice(validation_dataset)
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


def doInference(imagePath, modelPath, labelPath=None, outputPath=None, device='cpu'):
    print(" ===========================")
    print("          Inference     "    )
    print(" ===========================")
    
    # Load configuration
    configPath = os.path.splitext(modelPath)[0] + ".json"
    with open(configPath, 'r') as f:
        config = json.load(f)
    
    histogram_landmarks = np.array(config['histogram_landmarks_lst'])
    
    # Load image
    subject = tio.Subject(mri=tio.ScalarImage(imagePath))
    
    # Preprocess the image
    transform = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(config['resample_to']),
        tio.CropOrPad(config['crop_or_pad_size']),
        tio.HistogramStandardization({'mri': histogram_landmarks}),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    ])
    transformed = transform(subject)
    
    # Load model
    model = UNet(
        in_channels=config['in_channels'],
        out_classes=config['out_classes'],
        dimensions=config['dimensions'],
        num_encoding_blocks=config['num_encoding_blocks'],
        out_channels_first_layer=config['out_channels_first_layer'],
        normalization=config['normalization'],
        upsampling_type=config['upsampling_type'],
        padding=config['padding'],
        activation=config['activation'],
    ).to(device)
    
    checkpoint = torch.load(modelPath, map_location=device)
    if len(checkpoint) == 3:
        model.load_state_dict(checkpoint['weights'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    
    # Apply the model
    with torch.no_grad():
        input_tensor = transformed['mri'][tio.DATA].unsqueeze(0).to(device)
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        segmentation = probabilities.argmax(dim=1).squeeze().cpu().numpy()
    
    # Resize segmentation to match original image
    original_image = subject['mri']
    target_size = original_image.shape[1:]  # Exclude channel dimension
    # Create a TorchIO subject with the segmentation
    seg_subject = tio.Subject(seg=tio.LabelMap(tensor=segmentation[None, ...]))  # Add channel dimension
    
    # Create resize transform
    resize_transform = tio.Resize(target_size, image_interpolation='nearest')
    
    # Apply resize transform
    resized_seg_subject = resize_transform(seg_subject)
    resized_seg = resized_seg_subject['seg'].data

    # Save the output image
    if not outputPath:
        outputPath = os.path.splitext(imagePath)[0] + '-label' + os.path.splitext(imagePath)[1]
        
    tio.LabelMap(tensor=resized_seg.to(torch.uint8), affine=original_image.affine).save(outputPath)
    
    dice_score = None
    if labelPath:
        true_label = tio.LabelMap(labelPath)              
        true_label_data = true_label.data.squeeze().numpy()
        resized_seg_data = resized_seg.squeeze().numpy()        
        dice_score = get_dice_score(true_label_data,resized_seg_data)
        
                
    return dice_score


def main(useWholeImages=useWholeImages):
    print("useWholeImages: ",useWholeImages)
    if useWholeImages:
       trainWholeImage()
    else:
       trainPatchImage()

if __name__ == '__main__':
    
    sTm = time.time()
    
    if doTraining:
       print("useWholeImages: ", useWholeImages)
       main(useWholeImages=useWholeImages)     
    else:
       dice_score = doInference(inputImagePath, inputModelPath, labelPath=inputLabelPath, outputPath=outputSegPath)    
       print("dice_score : ", dice_score )
    
    print("Total time : ", time.time()-sTm, " seconds")