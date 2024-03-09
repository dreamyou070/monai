import os
from data.dataset import TrainDataset
import torch


def call_dataset(args) :

    # [1] set root data
    root_dir = os.path.join(args.data_path, f'{args.obj_name}/train')

    tokenizer = None
    if not args.on_desktop :
        from model.tokenizer import load_tokenizer
        tokenizer = load_tokenizer(args)
    print(f'root_dir = {root_dir}')
    dataset = TrainDataset(root_dir=root_dir,
                           anomaly_source_path=args.anomal_source_path,
                           resize_shape=[256,256],
                           tokenizer=tokenizer,
                           caption=args.trigger_word,
                           latent_res=args.latent_res,)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True)

    return dataloader