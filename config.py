CONFIG = {
    # Basic config
    'seed': 41,
    'is_training': 1,
    'model_id': 'K-Medicon',
    'task_name': 'classification',
    'model': 'Medformer',
    'monitor': 'vali_loss',
    
    # Data loader
    'data': 'K-Medicon',
    # 'root_path': './dataset/SingleLead/',
    'root_path': './dataset/',
    # 'features': 'M',    # options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate"
    # 'freq': 's',
    
    # Forecasting task
    'seq_len': 96,
    # 'label_len': 48,
    'pred_len': 96,
    
    # Model define for baselines
    # 'top_k': 5,
    # 'num_kernels': 6,
    'enc_in': 7,
    # 'dec_in': 7,
    'c_out': 7,
    'd_model': 128,
    'n_heads': 8,
    'e_layers': 6,
    # 'd_layers': 1,
    'd_ff': 256,
    # 'moving_avg': 25,
    'factor': 1,
    # 'distil': True,
    'dropout': 0.1,
    'embed': 'timeF',
    'activation': 'gelu',
    'output_attention': False,
    'no_inter_attn': False,
    # 'sampling_rate': 125,
    'patch_len_list': '2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32',
    'single_channel': True,
    'augmentations': 'jitter0.2,scale0.2,drop0.5',

    # Optimization
    'num_workers': 0,
    'itr': 1,
    'train_epochs': 100,
    'batch_size': 4,
    'patience': 30,
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
    
    # de-stationary projector params
    # 'p_hidden_dims': [128, 128],
    # 'p_hidden_layers': 2,
    
}