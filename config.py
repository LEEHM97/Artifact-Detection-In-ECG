CONFIG = {
    # Basic config
    'seed': 41,
    'is_training': 1,
    'model_id': 'NORM04',
    'task_name': 'classification',
    'model': 'Medformer',
    'monitor': 'CPI',
    
    # Data loader
    'data': 'K-Medicon',
    'root_path': './dataset/KMedicon/',
    'split_ratio': 0.8,
    
    # Forecasting task
    'seq_len': 96,
    'pred_len': 96,
    
    # Model define for baselines
    'enc_in': 7,
    'c_out': 7,
    'd_model': 128,
    'n_heads': 8,
    'e_layers': 6,
    'd_ff': 256,
    'factor': 1,
    'dropout': 0.1,
    'embed': 'timeF',
    'activation': 'gelu',
    'output_attention': False,
    'no_inter_attn': False,
    'patch_len_list': '2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32',
    'single_channel': False,
    'augmentations': 'jitter0.2,scale0.2,drop0.5',

    # Optimization
    'num_workers': 0,
    'itr': 1,
    'train_epochs': 100,
    'batch_size': 4,
    'patience': 5,
    'learning_rate': 1e-4,
    'des': 'Exp',
    'loss': 'MSE',
    'lradj': 'type1',
    'swa': True,
    
    # GPU
    'use_gpu': True,
    'gpu': 0,
    'use_multi_gpu': False,
    'devices': '0,1,2,3',
    
}