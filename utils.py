import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd

def add_hooks(model, safe_region_counter):
    for layer_name,layer in model.named_modules():
        if isinstance(layer, torch.nn.modules.activation.ReLU):
            safe_region_counter.hooks[layer_name] = layer.register_forward_pre_hook(
                safe_region_counter.get_activation(layer_name))

def run_counter(model, dataset, safe_region_counters,
        train = True, batch_size=128, loader_workers=4, adversarial = False):
   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for s in safe_region_counters:
        s.train = train
    
    dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=loader_workers)
    correct, correct_adv = 0, 0
    for batch_samples, target in tqdm.tqdm(dataloader):
        batch_samples, target = batch_samples.to(device), target.to(device)
        if adversarial:
            batch_samples.requires_grad = True
            for handler in safe_region_counters:
                handler.add = False
        output = model(batch_samples)
        if adversarial:
            for handler in safe_region_counters:
                handler.add = True
        init_pred = output.max(1, keepdim=True)[1].flatten() # get the index of the max log-probability
        correct += torch.sum(init_pred == target).item()
        #if init_pred.item() == target.item():
        #    correct += 1

        if adversarial:
            # If the initial prediction is wrong, dont bother attacking, just move on
            #if init_pred.item() != target.item():
            #    continue

            # Calculate the loss
            loss = F.nll_loss(output, target)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = batch_samples.grad.data
            
            # Call FGSM Attack
            perturbed_data = fgsm_attack(batch_samples, data_grad)

            # Re-classify the perturbed image
            output = model(perturbed_data)

            # Check for success
            final_pred = output.max(1, keepdim=True)[1].flatten() # get the index of the max log-probability
            
            correct_adv += torch.sum(final_pred == target).item()
            #if final_pred.item() == target.item():
            #    correct_adv += 1
    final_acc = correct/float(len(dataloader))
    final_acc_adv = correct_adv/float(len(dataloader))
    return final_acc, final_acc_adv

def add_safe_results(safe_region_counter, is_resnet=True):
    prob_safe_layer = []
    for layer_name, safe_counts in safe_region_counter.count_safe.items():
        unsafe_counts = safe_region_counter.count_notsafe[layer_name]
        if is_resnet and layer_name != 'relu':
            # torchvision implementation of resnet calls each relu in the basicblock layer twice
            num_safe1, num_safe2 = 0,0
            num_notsafe1, num_notsafe2 = 0,0
            assert len(safe_counts)%2 == 0
            for i in range(0,len(safe_counts),2):
                num_safe1 += safe_counts[i]
                num_safe2 += safe_counts[i+1]
                num_notsafe1 += unsafe_counts[i]
                num_notsafe2 += unsafe_counts[i+1]
            prob_safe_layer.append(num_safe1 / (num_safe1 + num_notsafe1))
            prob_safe_layer.append(num_safe2 / (num_safe2 + num_notsafe2))
        else:
            prob_safe_layer.append(sum(safe_counts) / (sum(safe_counts) + sum(unsafe_counts)))
    return prob_safe_layer

def quick_train(dataset,model, num_classes=None,
        batch_size=128, loader_workers=5,
        criterion=nn.CrossEntropyLoss(),
        num_epochs = 5):
    """ Naive quick training of a model"""
    if num_classes is not None:
        model.fc = torch.nn.Linear(model.fc.in_features,num_classes)
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=loader_workers)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs,labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Loss {}/{} = {}".format(epoch, num_epochs, 
            running_loss / len(dataset)))

def plot_prob_res(prob_safe_df, title):
    prob_safe_df['layer'] = range(prob_safe_df.shape[0])
    prob_safe_df = pd.melt(prob_safe_df, id_vars="layer", var_name="sampler", value_name="prob_safe")
    _, ax = plt.subplots(figsize=(10, 3))
    result_plot = sns.barplot(data = prob_safe_df,ax=ax, x = 'layer', y = 'prob_safe', hue = 'sampler')
    #result_plot.set(ylim=(0.5,1))
    result_plot.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1)
    result_plot.set_title(title)

def plot_minmax(all_handlers):
    params_df_min,params_df_max = pd.DataFrame(), pd.DataFrame()
    
    for k,v in all_handlers.items():
        min_arr, max_arr = [],[]
        if isinstance(v.min,dict):        
            for layer_name in v.min:
                mi, ma = v.get_bounds(layer_name)
                min_arr.append(mi)
                max_arr.append(ma)
        params_df_min[k] = min_arr
        params_df_max[k] = max_arr
        
    params_df_min['layer'] = range(params_df_min.shape[0])
    params_df_min = pd.melt(params_df_min, id_vars="layer", var_name="sampler", value_name="bounds")
    minplot = sns.lineplot(data = params_df_min, x = 'layer', y = 'bounds', hue = 'sampler')
    minplot.set_title('Minimum Bounds')

    plt.figure()
    params_df_max['layer'] = range(params_df_max.shape[0])
    params_df_max = pd.melt(params_df_max, id_vars="layer", var_name="sampler", value_name="bounds")
    maxplot = sns.lineplot(data = params_df_max, x = 'layer', y = 'bounds', hue = 'sampler')
    maxplot.set_title('Maximum Bounds')

def fgsm_attack(image, data_grad, epsilon = 0.2):
    # FGSM attack code - https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image
