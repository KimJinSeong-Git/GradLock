from features import dataset, model, trainer, utils
import torch
import torch.nn as nn
import torch.optim as optim
import os
import yaml

inject_ratio = 0.0

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using '{device}' now.")

def load_config(config_path="default.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config["save_dir"] = f'./results/{config["prefix"]} {config["model"]["type"]}_{config["model"]["name"]}_{config["dataset"]["name"]}_{config["training"]["num_epochs"]}epochs_h{config["model"]["hidden_size"]}_d{config["model"]["layer_depth"]}_drop{config["model"]["drop_out"]}_r{config["attack"]["inject_ratio"]}'

    return config

def get_trainable_params(model):
    return [param for param in model.parameters() if param.requires_grad]

def train_model(config):
    # set model
    model_name = config["model"]["name"]
    drop_out = config["model"]["drop_out"]
    input_size = config["model"]["input_size"]
    hidden_size = config["model"]["hidden_size"]
    layer_depth = config["model"]["layer_depth"]
    num_classes = config["model"]["num_classes"]
    is_pretrained = config["model"]["is_pretrained"]

    target_model = model.CNN_Model(
        feature_model = model_name,
        drop_out = drop_out,
        input_size = input_size,
        hidden_size = hidden_size,
        layer_depth = layer_depth,
        nb_classes = num_classes,
        is_pretrained = is_pretrained
    ).to(device)

    # set training params
    if config["dataset"]["name"] == "CelebA":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer_name = config["training"]["optimizer"]
    learning_rate_feat = config["training"]["learning_rate_feat"]
    learning_rate_cls = config["training"]["learning_rate_cls"]
    weight_decay = config["training"]["weight_decay"]

    if optimizer_name == "adam":
        optimizer = optim.Adam([
            {'params': get_trainable_params(target_model.feature_extractor), 'lr': learning_rate_feat},  # 작은 LR
            {'params': target_model.classifier.parameters(), 'lr': learning_rate_cls}  # 큰 LR 유지
        ])
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(target_model.parameters(), lr=learning_rate_cls, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    num_epochs = config["training"]["num_epochs"]
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # load dataset
    path_train = config["dataset"]["path_train"]
    path_val = config["dataset"]["path_val"]
    bs = config["dataset"]["bs"]

    train_dataset = dataset.create_dataset(path_train, input_size)
    train_loader = dataset.create_dataloader(train_dataset, bs=bs, shuffle=True)

    val_dataset = dataset.create_dataset(path_val, input_size)
    val_loader = dataset.create_dataloader(val_dataset, bs=bs, shuffle=False)

    # attack
    norm = config["attack"]["norm"]
    inject_ratio = config["attack"]["inject_ratio"]

    # run training roof
    save_dir = config["save_dir"]
    utils.gradlock(target_model, train_dataset, norm, inject_ratio)

    cnn_trainer = trainer.CNN_Trainer(
        save_dir=save_dir,
        model=target_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    print(f"\n[ {model_name} Training on the {config['dataset']['name']} Dataset ]")
    cnn_trainer.run(num_epochs=num_epochs)
    print()

if __name__=='__main__':
    config = load_config("default.yaml")
    train_model(config)