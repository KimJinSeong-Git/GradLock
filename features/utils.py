import torch
import numpy as np
import math
import random
import train_CNN


def get_cls_info(model):
    list_layer_name = []
    list_layer_shape = []
    for (name, weights) in model.classifier.state_dict().items():
        tokens = name.split('.')
        if tokens[-1] == 'weight':
            list_layer_name.append(name)
            list_layer_shape.append(weights.shape)

    return list_layer_name[:-1], list_layer_shape[:-1]

def get_rows(nb_rows, inject_ratio=None):
    if inject_ratio == None:
        points = np.sin(np.linspace(0, 1, int(nb_rows*train_CNN.inject_ratio)))
    else:
        points = np.sin(np.linspace(0, 1, int(nb_rows*inject_ratio)))

    rows = np.unique((points * nb_rows).astype(int))

    return rows

def insert_data_to_model(model, data, layer_name, chunk_size, rows, row_idx, norm=0.01):
    is_next_layer = True

    data_flat = data.flatten()
    chunks = torch.split(data_flat, chunk_size)
    len_chunks = len(chunks)

    if len_chunks > (len(rows) - row_idx):
        is_next_layer = False
        return is_next_layer, -1
    
    with torch.no_grad():
        param = dict(model.classifier.named_parameters())[layer_name]
        for chunk in chunks:
            chunk = chunk * norm
            param[rows[row_idx], :chunk.shape[0]] = chunk
            row_idx += 1
            
    return is_next_layer, row_idx

def backward_hook(grad):
    nb_rows = grad.shape[0]
    rows = get_rows(nb_rows)

    mask = torch.ones_like(grad)
    mask[rows] = 0

    return grad * mask

def set_hook(model):
    list_layer_name, _ = get_cls_info(model)

    named_params = dict(model.classifier.named_parameters())

    for weight_name in list_layer_name:
        # backward hook
        param = named_params[weight_name]
        param.register_hook(backward_hook)

def gradlock(model, data, norm=0.01, inject_ratio=0.5):
    train_CNN.inject_ratio = inject_ratio
    set_hook(model)
    
    list_layer_name, list_layer_shape = get_cls_info(model)
    
    len_data = len(data)
    idx_list = list(range(0, len_data))
    random.shuffle(idx_list)

    data_idx = 0
    for layer_idx in range(len(list_layer_name)):
        is_next_layer = True
        layer_name = list_layer_name[layer_idx]
        layer_shape = list_layer_shape[layer_idx]
        rows = get_rows(layer_shape[0])
        chunk_size = layer_shape[1]
        row_idx = 0

        while is_next_layer:
            x, y = data[idx_list[data_idx]]
            is_next_layer, row_idx = insert_data_to_model(model, x, layer_name, chunk_size, rows, row_idx, norm)
            data_idx += 1

            if data_idx >= len_data:
                print(f'[system] done. ({data_idx} data inserted)')
                return
            
    print(f'[system] done. ({data_idx} data inserted)')

def extract_data(device, model, img_shape, inject_ratio, norm=0.01):
    list_layer_name, list_layer_shape = get_cls_info(model)
    c, h, w = img_shape
    img_size = c*h*w

    extracted_data = torch.empty((0, img_size)).to(device)
    for idx in range(len(list_layer_name)):
        layer_name = list_layer_name[idx]
        layer_shape = list_layer_shape[idx]

        nb_rows = layer_shape[0]
        rows = get_rows(nb_rows, inject_ratio)

        inserted_data = model.classifier.state_dict()[layer_name][rows]

        chunk_size = layer_shape[1]
        chunks_per_data = math.ceil(img_size/chunk_size)

        for i in range(0, len(inserted_data), chunks_per_data):
            sampled_inserted_data = inserted_data[i:i+chunks_per_data]
            flat_data = sampled_inserted_data.flatten()[:img_size].unsqueeze(dim=0)

            if extracted_data.shape[1] == flat_data.shape[1]:
                extracted_data = torch.cat([extracted_data, flat_data])

    extracted_data = extracted_data.reshape(extracted_data.shape[0], c, h, w)/norm

    return extracted_data