import torch

from data_loader import KMediconLoader
from torch.utils.data import DataLoader


def data_provider(args, flag):
    Data = KMediconLoader
    timeenc = 0 if args['embed'] != "timeF" else 1

    if flag == "test":
        shuffle_flag = False
        drop_last = True
        if args['task_name'] == "anomaly_detection" or args['task_name'] == "classification":
            batch_size = args['batch_size']
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args['freq']
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args['batch_size']  # bsz for train and valid
        freq = args['freq']

    if args['task_name'] == "anomaly_detection":
        drop_last = False
        data_set = Data(
            root_path=args['root_path'],
            win_size=args['seq_len'],
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args['num_workers'],
            drop_last=drop_last,
        )
        return data_set, data_loader
    elif args['task_name'] == "classification":
        drop_last = False
        data_set = Data(
            root_path=args['root_path'],
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args['num_workers'],
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(
                x, max_len=args['seq_len']
            ),  # only called when yeilding batches
        )
        return data_set, data_loader
    else:
        if args['data'] == "m4":
            drop_last = False
        data_set = Data(
            root_path=args['root_path'],
            data_path=args['data_path'],
            flag=flag,
            size=[args['seq_len'], args['label_len'], args['pred_len']],
            features=args['features'],
            target=args['target'],
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args['seasonal_patterns'],
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args['num_workers'],
            drop_last=drop_last,
        )
        return data_set, data_loader


def collate_fn(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [
        X.shape[0] for X in features
    ]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)

    X = torch.zeros(
        batch_size, max_len, features[0].shape[-1]
    )  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(
        torch.tensor(lengths, dtype=torch.int16), max_len=max_len
    )  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = (
        max_len or lengths.max_val()
    )  # trick works because of overloading of 'or' operator for non-boolean types
    return (
        torch.arange(0, max_len, device=lengths.device)
        .type_as(lengths)  # convert to same type as lengths tensor
        .repeat(batch_size, 1)  # (batch_size, max_len)
        .lt(lengths.unsqueeze(1))
    )