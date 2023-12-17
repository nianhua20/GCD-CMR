import os
import argparse

from utils.functions import Storage


class ConfigRegression():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'gcd_cmr': self.__GCD_CMR
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if args.aligned else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(
            dict(
                vars(args),
                **dataArgs,
                **commonArgs,
                **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
            ))

    def __datasetCommonParams(self):
        root_dataset_dir = '/path/to/datastes'
        tmp = {
            'mosi': {
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                }
            },
            'mosei': {
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                }
            },
            'sims': {
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/features/unaligned_39.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (39, 400, 55),  # (text, audio, video)
                    'feature_dims': (768, 33, 709),  # (text, audio, video)
                    'train_samples': 1368,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                }
            }
        }
        return tmp

    def __GCD_CMR(self):
        tmp = {
            'commonParas': {
                'need_data_aligned': True,
                'need_model_aligned': True,
                'need_normalized': False,
                'use_bert': True,
                'use_finetune': True,
                'save_labels': False,
                'early_stop': 15,
                'update_epochs': 1
            },
            # dataset
            'datasetParas': {
                'mosi': {
                    # the batch_size of each epoch is update_epochs * batch_size
                    # 'batch_size': 64,
                    # 'epochs': 30,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 1e-3,
                    'learning_rate_video': 1e-4,
                    'learning_rate_other': 1e-3,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.001,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768,
                    'audio_out': 64,
                    'video_out': 128,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout': 0.1,
                    # res
                    'H': 3.0,
                    
                    # mi_net_lr
                    'mi_net_decay': 1e-4,
                    'mi_net_decay': 0.001,
                    
                },
                'mosei': {
                    # the batch_size of each epoch is update_epochs * batch_size
                    # 'batch_size': 64,
                    # 'epochs': 30,
                    'learning_rate_bert': 5e-6,
                    'learning_rate_audio': 1e-4,
                    'learning_rate_video': 1e-4,
                    'learning_rate_other': 1e-4,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.001,
                    'weight_decay_video': 0.001,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768,
                    'audio_out': 64,
                    'video_out': 128,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout': 0.1,
                    # res
                    'H': 3.0,

                    # mi_net_lr
                    'mi_net_decay': 1e-4,
                    'mi_net_decay': 0.001,
                },

            },
        }
        return tmp

    def get_config(self):
        return self.args
