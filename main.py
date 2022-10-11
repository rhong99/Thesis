# training + preprocessing for AffectNet


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import models
import time
import preprocess
import shutil
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, sampler, Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from torchsummary import summary
from os import listdir
from os.path import isfile, join
from PIL import Image


def main():
    # check if device has cuda enabled/available
    # currently set to use single gpu (cuda:0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))

    # save name for trained pytorch model
    savename = 'model'

    # model settings
    # imbalanced = use of weights inversely proportional to class size
    # pretrained = using MS-Celeb-1M pretrained for ResNet18
    # openface = using openface au after CNN layers
    imbalanced = True
    pretrained = True
    openface = False

    # number of emotion classes to train for
    emotion_classes = 8

    # hyperparameters
    batch_size = 256
    epochs = 5
    lr = 0.0001

    # more settings
    # training = training model (vs. eval)
    # boosting = using boosted dataset
    # multiclass = multiclass training (vs. binary)
    training = True
    boosting = False
    multiclass = True

    # graph model
    plot_cm = True
    # save pytorch model
    save_model = True
    # show plots
    plot_graphs = True

    # file paths
    # unzipped but unprocessed AffectNet
    unprocessed_path = 'TODO'
    # output of preprocessing AffectNet
    processed_path = 'TODO'
    # output of boosting
    processed_boosted_path = 'TODO'

    # file paths for binary classification
    binary_anger = 'TODO'
    binary_contempt = 'TODO'
    binary_disgust = 'TODO'
    binary_fear = 'TODO'
    binary_happy = 'TODO'
    binary_neutral = 'TODO'
    binary_sad = 'TODO'
    binary_surprise = 'TODO'

    # list of emotion classes for binary classification
    binary_emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # set to True if you need to preprocess AffectNet
    needs_sorting = False
    # set to True if you need to boost dataset
    boost_dataset = False
    # set to True if you need to boost the binary dataset
    boost_binary = False

    # preprocess AffectNet
    # this process may take a while
    if needs_sorting:
        preprocess.sort_data_train(unprocessed_path + '/train', processed_path)
        preprocess.sort_data_val(unprocessed_path + '/val', processed_path)

    # boosting dataset
    if boost_dataset:
        # list of emotion classes to boost
        boost_emotions = ['TODO']
        preprocess.boost_data(processed_boosted_path, boost_emotions)

    # boosting binary dataset
    if boost_binary:
        # e.g. boost anger
        preprocess.boost_binary_data(binary_anger)

    # training
    # about 300s per epoch using a RTX 3090, uses about 11GB of VRAM with batch_size 256
    if training:
        # not using OpenFace AUs
        if not openface:
            # multiclass
            if multiclass:
                # data preprocessing
                transform = transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomApply([transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2))], p=0.7),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                transforms.RandomErasing()])
                transform_val = transforms.Compose([transforms.Resize((224, 224)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                if boosting:
                    train_images = torchvision.datasets.ImageFolder(root=processed_boosted_path + '/train', transform=transform)
                    val_images = torchvision.datasets.ImageFolder(root=processed_boosted_path + '/val', transform=transform_val)
                else:
                    train_images = torchvision.datasets.ImageFolder(root=processed_path + '/train', transform=transform)
                    val_images = torchvision.datasets.ImageFolder(root=processed_path + '/val', transform=transform_val)
                if imbalanced:
                    train_loader = DataLoader(train_images, batch_size=batch_size, sampler=ImbalancedSampler(train_images),
                                              shuffle=False, num_workers=4)
                    print('Imbalanced Dataset: Different Weights')
                else:
                    train_loader = DataLoader(train_images, batch_size=batch_size, shuffle=True, num_workers=4)
                val_loader = DataLoader(val_images, batch_size=batch_size, shuffle=False, num_workers=4)

                # declare model
                model = models.ResNet18(pretrained, emotion_classes).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

                # enable to see stats
                # summary(model, input_size=(3, 224, 224), batch_size=batch_size, device='cuda')

                # tracking values
                tacc = []
                tloss = []
                vacc = []
                vloss = []
                epoch = []
                iteracc = []
                init_time = time.time()

                # training loop
                for e in range(0, epochs):
                    running_loss = 0.0
                    tempacc = []

                    # batches
                    for i, data in enumerate(train_loader):
                        features, labels = data
                        features = features.to(device)
                        labels = labels.to(device)

                        # compute forward pass and backprop
                        optimizer.zero_grad()
                        outputs = model(features)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        # track stats
                        tempacc.append(count_predictions(outputs, labels))
                        running_loss += loss.item()
                        iteracc.append(tempacc[len(tempacc)-1])
                        if i % 100 == 0:
                            print('Epoch: {}  Iteration: {}'.format(e + 1, i))
                            print('Time Elapsed: {}'.format(time.time() - init_time))
                            print('Training Batch Accuracy: {}'.format(tempacc[len(tempacc)-1]))

                    # calculate validation stats
                    vtemploss, vtempacc = evaluate(model, val_loader, criterion, device, 3998)

                    # save stats
                    tacc.append(sum(tempacc) / len(tempacc))
                    tloss.append(running_loss / 287650)
                    vacc.append(vtempacc)
                    vloss.append(vtemploss)
                    epoch.append(e + 1)

                    # print stats
                    print('------------')
                    print('Epoch: {}'.format(e + 1))
                    print('Time Elapsed: {}'.format(time.time() - init_time))
                    print('Training Accuracy: {}  Training Loss: {}'.format(tacc[len(tacc) - 1], tloss[len(tloss) - 1]))
                    print('Validation Accuracy: {}  Validation Loss: {}'.format(vacc[len(vacc) - 1], vloss[len(vloss) - 1]))
                    print('------------')

                # compute class accuracy
                classacc = class_accuracies(model, val_loader, device)
                print('Validation Class Accuracies')
                print(classacc)

                # confusion matrix
                cm = conf_matrix(model, val_loader, criterion, device)
                print('Confusion Matrix')
                # print(cm)
                if plot_cm:
                    display = ConfusionMatrixDisplay(cm)
                    display.plot()
                    plt.savefig(savename + '_cm.png')
                    plt.show()

                # save model
                if save_model:
                    print('Model saved.')
                    torch.save(model, savename + '.pt')

                # plot accuracy and loss graphs
                if plot_graphs:
                    plot_acc(tacc, vacc, epoch, savename)
                    plot_iter_acc(iteracc, savename)
                    # plot_loss(tloss, vloss, epoch)

            # binary
            if not multiclass:
                # train a model for each emotion
                for binary_emotion in binary_emotions:
                    print('Running: {}'.format(binary_emotion))
                    binary_filepath = 'TODO' + binary_emotion
                    transform = transforms.Compose([transforms.Resize((224, 224)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomApply(
                                                        [transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2))],
                                                        p=0.7),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225]),
                                                    transforms.RandomErasing()])
                    transform_val = transforms.Compose([transforms.Resize((224, 224)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])])
                    train_images = torchvision.datasets.ImageFolder(root=binary_filepath + '/train', transform=transform)
                    val_images = torchvision.datasets.ImageFolder(root=binary_filepath + '/val', transform=transform_val)
                    if imbalanced:
                        train_loader = DataLoader(train_images, batch_size=batch_size, sampler=ImbalancedSampler(train_images), shuffle=False, num_workers=4)
                        print('Imbalanced Dataset: Different Weights')
                    else:
                        train_loader = DataLoader(train_images, batch_size=batch_size, shuffle=True, num_workers=4)
                    val_loader = DataLoader(val_images, batch_size=batch_size, shuffle=False, num_workers=4)

                    model = models.ResNet18(pretrained, 2).to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

                    # summary(model, input_size=(3, 224, 224), batch_size=batch_size, device='cuda')

                    tacc = []
                    tloss = []
                    vacc = []
                    vloss = []
                    epoch = []
                    iteracc = []

                    init_time = time.time()

                    for e in range(0, epochs):
                        running_loss = 0.0
                        tempacc = []

                        for i, data in enumerate(train_loader):
                            features, labels = data
                            features = features.to(device)
                            labels = labels.to(device)

                            optimizer.zero_grad()
                            outputs = model(features)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()

                            tempacc.append(count_predictions(outputs, labels))
                            running_loss += loss.item()

                            iteracc.append(tempacc[len(tempacc) - 1])

                            if i % 100 == 0:
                                print('Epoch: {}  Iteration: {}'.format(e + 1, i))
                                print('Time Elapsed: {}'.format(time.time() - init_time))
                                print('Training Batch Accuracy: {}'.format(tempacc[len(tempacc) - 1]))

                        vtemploss, vtempacc = evaluate(model, val_loader, criterion, device, 3998)

                        tacc.append(sum(tempacc) / len(tempacc))
                        tloss.append(running_loss / 287650)
                        vacc.append(vtempacc)
                        vloss.append(vtemploss)
                        epoch.append(e + 1)

                        print('------------')
                        print('Epoch: {}'.format(e + 1))
                        print('Time Elapsed: {}'.format(time.time() - init_time))
                        print('Training Accuracy: {}  Training Loss: {}'.format(tacc[len(tacc) - 1], tloss[len(tloss) - 1]))
                        print('Validation Accuracy: {}  Validation Loss: {}'.format(vacc[len(vacc) - 1], vloss[len(vloss) - 1]))
                        print('------------')

                    # classacc = class_accuracies(model, val_loader, device)
                    # print('Validation Class Accuracies')
                    # print(classacc)

                    cm = conf_matrix(model, val_loader, criterion, device)
                    print('Confusion Matrix')
                    # print(cm)
                    if plot_cm:
                        display = ConfusionMatrixDisplay(cm)
                        display.plot()
                        plt.savefig(savename + '_binary_' + binary_emotion + '_cm.png')
                        # plt.show()

                    if save_model:
                        print('Model saved.')
                        torch.save(model, savename + '_binary_' + binary_emotion + '.pt')

                    if plot_graphs:
                        plot_acc(tacc, vacc, epoch, savename + '_binary_' + binary_emotion)
                        plot_iter_acc(iteracc, savename + '_binary_' + binary_emotion)
                        # plot_loss(tloss, vloss, epoch)

        # using OpenFace AUs
        else:
            # multiclass
            if multiclass:
                transform = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((224, 224)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomApply(
                                                    [transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2))],
                                                    p=0.7),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225]),
                                                transforms.RandomErasing()])
                transform_val = transforms.Compose([transforms.ToPILImage(),
                                                    transforms.Resize((224, 224)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])
                train_images = OpenFaceDatasetAU('openface.csv', 'TODO', transform)
                val_images = OpenFaceDatasetAU('openface_val.csv', 'TODO', transform_val)
                train_loader = DataLoader(train_images, batch_size=batch_size, shuffle=True, num_workers=4)
                val_loader = DataLoader(val_images, batch_size=batch_size, shuffle=False, num_workers=4)

                model = models.ResNet18_openface_au(pretrained, emotion_classes).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

                # summary(model, input_size=(3, 224, 224), batch_size=batch_size, device='cuda')

                tacc = []
                tloss = []
                vacc = []
                vloss = []
                epoch = []
                iteracc = []

                init_time = time.time()

                for e in range(0, epochs):
                    running_loss = 0.0
                    tempacc = []

                    for i, data in enumerate(train_loader):
                        features, labels, openface_landmarks = data
                        features = features.to(device)
                        labels = labels.to(device)
                        openface_landmarks = openface_landmarks.to(device)

                        optimizer.zero_grad()
                        outputs = model(features, openface_landmarks)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        tempacc.append(count_predictions_openface(outputs, labels))
                        running_loss += loss.item()

                        iteracc.append(tempacc[len(tempacc) - 1])

                        if i % 100 == 0:
                            print('Epoch: {}  Iteration: {}'.format(e + 1, i))
                            print('Time Elapsed: {}'.format(time.time() - init_time))
                            print('Training Batch Accuracy: {}'.format(tempacc[len(tempacc) - 1]))

                    vtemploss, vtempacc = evaluate_openface(model, val_loader, criterion, device, 3998)

                    tacc.append(sum(tempacc) / len(tempacc))
                    tloss.append(running_loss / 287650)
                    vacc.append(vtempacc)
                    vloss.append(vtemploss)
                    epoch.append(e + 1)

                    print('------------')
                    print('Epoch: {}'.format(e + 1))
                    print('Time Elapsed: {}'.format(time.time() - init_time))
                    print('Training Accuracy: {}  Training Loss: {}'.format(tacc[len(tacc) - 1], tloss[len(tloss) - 1]))
                    print('Validation Accuracy: {}  Validation Loss: {}'.format(vacc[len(vacc) - 1],
                                                                                vloss[len(vloss) - 1]))
                    print('------------')

                classacc = class_accuracies_openface(model, val_loader, device)
                print('Validation Class Accuracies')
                print(classacc)

                cm = conf_matrix_openface(model, val_loader, criterion, device)
                print('Confusion Matrix')
                # print(cm)
                if plot_cm:
                    display = ConfusionMatrixDisplay(cm)
                    display.plot()
                    plt.savefig(savename + '_cm.png')
                    plt.show()

                if save_model:
                    print('Model saved.')
                    torch.save(model, savename + '.pt')

                if plot_graphs:
                    plot_acc(tacc, vacc, epoch, savename)
                    plot_iter_acc(iteracc, savename)
                    # plot_loss(tloss, vloss, epoch)

    if not training:
        load_image('TODO')


def count_predictions(outputs, labels):
    acc = 0
    for i in range(len(outputs)):
        if torch.argmax(outputs[i]) == labels[i]:
            acc += 1
    return acc / len(outputs)


def count_predictions_openface(outputs, labels):
    acc = 0
    for i in range(len(outputs)):
        if torch.argmax(outputs[i]) == torch.argmax(labels[i]):
            acc += 1
    return acc / len(outputs)


def evaluate(model, load, criterion, device, total):
    total_corr = 0
    total_loss = 0.0

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(load):
            feats, labels = batch
            feats = feats.to(device)
            labels = labels.to(device)

            predictions = model(feats.float())
            corr = torch.argmax(predictions, dim=1) == labels
            total_corr += int(corr.sum())
            loss = criterion(predictions, labels)
            total_loss += loss
        total_loss = total_loss / i

    model.train(True)

    return total_loss, float(total_corr) / total


def evaluate_openface(model, load, criterion, device, total):
    total_corr = 0
    total_loss = 0.0

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(load):
            feats, labels, openface_landmarks = batch
            feats = feats.to(device)
            labels = labels.to(device)
            openface_landmarks = openface_landmarks.to(device)

            predictions = model(feats.float(), openface_landmarks)
            corr = torch.argmax(predictions, dim=1) == torch.argmax(labels, dim=1)
            total_corr += int(corr.sum())
            loss = criterion(predictions, labels)
            total_loss += loss
        total_loss = total_loss / i

    model.train(True)

    return total_loss, float(total_corr) / total


def class_accuracies(model, load, device):
    # add more/less emotions based on classes used
    neutral = []
    happy = []
    sad = []
    surprise = []
    fear = []
    disgust = []
    anger = []
    contempt = []

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(load):
            feats, labels = batch
            feats = feats.to(device)
            labels = labels.to(device)
            predictions = model(feats.float())

            for j in range(0, len(labels)):
                label = labels[j].item()
                prediction = torch.argmax(predictions, dim=1)[j].item()

                if label == 0:
                    if label == prediction:
                        neutral.append(1)
                    else:
                        neutral.append(0)
                if label == 1:
                    if label == prediction:
                        happy.append(1)
                    else:
                        happy.append(0)
                if label == 2:
                    if label == prediction:
                        sad.append(1)
                    else:
                        sad.append(0)
                if label == 3:
                    if label == prediction:
                        surprise.append(1)
                    else:
                        surprise.append(0)
                if label == 4:
                    if label == prediction:
                        fear.append(1)
                    else:
                        fear.append(0)
                if label == 5:
                    if label == prediction:
                        disgust.append(1)
                    else:
                        disgust.append(0)
                if label == 6:
                    if label == prediction:
                        anger.append(1)
                    else:
                        anger.append(0)
                if label == 7:
                    if label == prediction:
                        contempt.append(1)
                    else:
                        contempt.append(0)

    model.train(True)

    return [sum(neutral) / len(neutral),
            sum(happy) / len(happy),
            sum(sad) / len(sad),
            sum(surprise) / len(surprise),
            sum(fear) / len(fear),
            sum(disgust) / len(disgust),
            sum(anger) / len(anger),
            sum(contempt) / len(contempt)]


def class_accuracies_openface(model, load, device):
    # add more/less emotions based on classes used
    neutral = []
    happy = []
    sad = []
    surprise = []
    fear = []
    disgust = []
    anger = []
    contempt = []

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(load):
            feats, labels, openface_landmarks = batch
            feats = feats.to(device)
            labels = labels.to(device)
            openface_landmarks = openface_landmarks.to(device)
            predictions = model(feats.float(), openface_landmarks)

            for j in range(0, len(labels)):
                label = torch.argmax(labels, dim=1)[j].item()
                prediction = torch.argmax(predictions, dim=1)[j].item()

                if label == 0:
                    if label == prediction:
                        neutral.append(1)
                    else:
                        neutral.append(0)
                if label == 1:
                    if label == prediction:
                        happy.append(1)
                    else:
                        happy.append(0)
                if label == 2:
                    if label == prediction:
                        sad.append(1)
                    else:
                        sad.append(0)
                if label == 3:
                    if label == prediction:
                        surprise.append(1)
                    else:
                        surprise.append(0)
                if label == 4:
                    if label == prediction:
                        fear.append(1)
                    else:
                        fear.append(0)
                if label == 5:
                    if label == prediction:
                        disgust.append(1)
                    else:
                        disgust.append(0)
                if label == 6:
                    if label == prediction:
                        anger.append(1)
                    else:
                        anger.append(0)
                if label == 7:
                    if label == prediction:
                        contempt.append(1)
                    else:
                        contempt.append(0)

    model.train(True)

    return [sum(neutral) / len(neutral),
            sum(happy) / len(happy),
            sum(sad) / len(sad),
            sum(surprise) / len(surprise),
            sum(fear) / len(fear),
            sum(disgust) / len(disgust),
            sum(anger) / len(anger),
            sum(contempt) / len(contempt)]


def plot_acc(t, v, steps, savename):
    tacc = np.asarray(t)
    vacc = np.asarray(v)
    step = np.asarray(steps)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(step, tacc, color='red', linewidth=0.8)
    ax.plot(step, vacc, color='blue', linewidth=0.8)
    ax.set(title='Accuracy vs. Epochs', ylabel='Accuracy', xlabel='Epochs')
    ax.legend(['Training', 'Validation'])
    plt.savefig(savename + '_accuracy.png')
    # plt.show()


def plot_iter_acc(iteracc, savename):
    iters = []
    for i in range(0, len(iteracc)):
        iters.append(i+1)
    iterations = np.asarray(iters)
    acc = np.asarray(iteracc)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(iterations, acc, color='blue', linewidth=0.8)
    ax.set(title='Batch Accuracy vs. Iterations', ylabel='Accuracy', xlabel='Iterations')
    plt.savefig(savename + '_batch_accuracy.png')
    # plt.show()


def plot_loss(t, v, steps):
    tloss = np.asarray(t)
    vloss = np.asarray(v)
    step = np.asarray(steps)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(step, tloss, color='red', linewidth=0.8)
    ax.plot(step, vloss, color='blue', linewidth=0.8)
    ax.set(title='Loss vs. Epochs', ylabel='Loss', xlabel='Epochs')
    ax.legend(['Training', 'Validation'])
    plt.savefig('loss.png')
    # plt.show()


def conf_matrix(model, val_loader, criterion, device):
    pred = []
    true = []

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            feats, label = batch
            feats = feats.to(device)
            label = label.to(device)
            prediction = model(feats.float())

            labels = label.tolist()
            predictions = (torch.argmax(prediction, dim=1)).tolist()
            for j in range(0, len(labels)):
                true.append(labels[j])
                pred.append(predictions[j])

    model.train()

    result = confusion_matrix(true, pred)

    return result


def conf_matrix_openface(model, val_loader, criterion, device):
    pred = []
    true = []

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            feats, label, openface_landmarks = batch
            feats = feats.to(device)
            label = label.to(device)
            openface_landmarks = openface_landmarks.to(device)
            prediction = model(feats.float(), openface_landmarks)

            labels = (torch.argmax(label, dim=1)).tolist()
            predictions = (torch.argmax(prediction, dim=1)).tolist()
            for j in range(0, len(labels)):
                true.append(labels[j])
                pred.append(predictions[j])

    model.train()

    result = confusion_matrix(true, pred)

    return result


def load_image(path):
    print('Evaluating')

    nets = ['binary_anger', 'binary_contempt', 'binary_disgust', 'binary_fear', 'binary_happy', 'binary_neutral', 'binary_sad', 'binary_surprise']

    # Checking for false negatives is true and false_neg/
    # Checking for false positives is false and false_pos/
    category = 'false'
    location = 'false_pos/'

    for net in nets:
        files = [f for f in listdir(path + net + '/val/' + category) if isfile(join(path + net + '/val/' + category, f))]

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = torch.load('pretrained_boosted_imbalanced_' + net + '.pt')
        model = model.to(device)
        model.eval()

        for i in range(0, len(files)):
            file = files[i]

            transform = transforms.Compose([transforms.ToTensor()])

            img = Image.open(path + net + '/val/' + category + '/' + file)
            im = transform(img)
            im = im.to(device)
            batch = torch.unsqueeze(im, 0)

            prediction = model(batch)
            pred = (torch.argmax(prediction, dim=1)).tolist()

            if pred[0] != 1:
                img.save(path + location + net + '/' + file)

        model.train()


class ImbalancedSampler(sampler.Sampler):
    def __init__(self, dataset):
        self.indices = list(range(len(dataset)))
        self.samples = len(self.indices)
        df = pd.DataFrame()
        df['labels'] = self.get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()
        labels_count = df['labels'].value_counts()
        weights = 1.0 / labels_count[df['labels']]
        self.weights = torch.DoubleTensor(weights.to_list())

    def get_labels(self, dataset):
        if isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.samples, replacement=True))

    def __len__(self):
        return self.samples


class OpenFaceDataset(Dataset):
    def __init__(self, csvfile, img_dir, transform=None):
        self.img_labels = pd.read_csv(csvfile)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_path = self.img_dir + str(int(self.img_labels.iloc[index]['file'])) + '.jpg'
        image = read_image(img_path)
        label = self.img_labels.iloc[index]['emotion_label']
        ohe_label = np.zeros(8)
        ohe_label[int(label)] = 1.0
        ohe_label = torch.from_numpy(ohe_label)
        df = self.img_labels.loc[[index]]
        df = df.drop(columns='file')
        df = df.drop(columns='emotion_label')
        df = torch.from_numpy(df.to_numpy()[0])
        if self.transform:
            image = self.transform(image)
        return image, ohe_label, df


class OpenFaceDatasetAU(Dataset):
    def __init__(self, csvfile, img_dir, transform=None):
        self.img_labels = pd.read_csv(csvfile)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_path = self.img_dir + str(int(self.img_labels.iloc[index]['file'])) + '.jpg'
        image = read_image(img_path)
        label = self.img_labels.iloc[index]['emotion_label']
        ohe_label = np.zeros(8)
        ohe_label[int(label)] = 1.0
        ohe_label = torch.from_numpy(ohe_label)
        df = self.img_labels.loc[[index]]
        df = df.drop(columns='file')
        df = df.drop(columns='emotion_label')
        df = df[[' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r',
                 ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r',
                 ' AU45_r']]
        df = torch.from_numpy(df.to_numpy()[0])
        if self.transform:
            image = self.transform(image)
        # sample = {'image': image, 'label': label, 'openface': df}
        return image, ohe_label, df


if __name__ == '__main__':
    main()

    # oftest = OpenFaceDataset('openface.csv', 'TODO')
    #
    # train_dataloader = DataLoader(oftest, batch_size=4, shuffle=True)
    # train_features, train_labels, train_ofdata = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # print(f"OF batch shape: {train_ofdata.size()}")

    # imgshow = train_features[0].squeeze()
    # labeling = train_labels[0]
    # plt.imshow(imgshow, cmap="gray")
    # plt.show()
    # print(f"Label: {labeling}")
    # print(train_ofdata[0])
