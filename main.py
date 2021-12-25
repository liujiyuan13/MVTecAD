from torch.utils.data import DataLoader
from torchvision import transforms
from mvtec_ad import MVTecAD


def _convert_label(x):
    '''
    convert anomaly label. 0: normal; 1: anomaly.
    :param x (int): class label
    :return: 0 or 1
    '''
    return 0 if x == 0 else 1

if __name__ == '__main__':

    # define transforms
    transform = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor()])
    target_transform = transforms.Lambda(_convert_label)

    # load data
    mvtec = MVTecAD('data',
                    subset_name='bottle',
                    train=True,
                    transform=transform,
                    mask_transform=transform,
                    target_transform=target_transform,
                    download=True)

    # feed to data loader
    data_loader = DataLoader(mvtec,
                             batch_size=2,
                             shuffle=True,
                             num_workers=8,
                             pin_memory=True,
                             drop_last=True)

    # obtain in batch
    for idx, (image, mask, target) in enumerate(data_loader):
        print(idx, target)
