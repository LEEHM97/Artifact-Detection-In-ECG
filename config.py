CONFIG = {
    # Basic config
    'seed': 41,
    'is_training': 1,
    'model_id': 'ESB01',
    'task_name': 'classification',
    'model': 'Medformer',
    'monitor': 'vali_loss',
    
    # Data loader
    'data': 'K-Medicon',
    'root_path': './dataset/KMedicon/',
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
    'patch_len': 16,
    'single_channel': False,
    'augmentations': 'jitter0.2,scale0.2,drop0.5',

    # Optimization
    'num_workers': 0,
    'itr': 1,
    'train_epochs': 100,
    'batch_size': 2,
    'patience': 10,
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
    
    # Optimizers
    'lr': 1e-10,
    'max_lr': 1e-4,
    # 'initial_wd': 1e-4,
    # 'final_wd': 1e-3,
    'wd': 1e-2,
    'T_0': 100,
    'T_mult': 1,
    'T_up': 5,
    'gamma': 0.5,
    
    # Ensemble Model Path
    'path1': "./checkpoints/classification_ESB_Medformer_K-Medicon_bs4_sl96_lr0.0001_pl96_dm128_nh8_el6_df256_fc1_ebtimeF_Exp_seed41/checkpoint.pth",
    'path2': "./checkpoints/classification_ESB_Medformer_K-Medicon_bs4_sl2500_lr0.0001_pl0_dm128_nh8_el6_df256_fc1_ebtimeF_Exp_seed42/checkpoint.pth",
    'path3': "./checkpoints/classification_ESB_Medformer_K-Medicon_bs4_sl2500_lr0.0001_pl0_dm128_nh8_el6_df256_fc1_ebtimeF_Exp_seed43/checkpoint.pth",
    'path4': "./checkpoints/classification_ESB_Medformer_K-Medicon_bs4_sl2500_lr0.0001_pl0_dm128_nh8_el6_df256_fc1_ebtimeF_Exp_seed44/checkpoint.pth",
    'path5': "./checkpoints/classification_ESB_Medformer_K-Medicon_bs4_sl2500_lr0.0001_pl0_dm128_nh8_el6_df256_fc1_ebtimeF_Exp_seed45/checkpoint.pth",
    
}