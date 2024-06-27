# datasets
from .datasets.dataloaders import mnist_loader, cifar_loader, femnist_loader, svhn_loader, datasetloader_preload, shuffle, femnist_loader_iid
from .datasets.dataset_partition import partition_label_quantity, partition_label_distribution, partition_by_index, generate_global_dataset