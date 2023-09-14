import os
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import sys
from collections import defaultdict

import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSPseudolabeledSubset

sys.path.insert(1, os.path.join(os.getcwd(), 'examples'))
from utils import set_seed, Logger, BatchLogger, log_config, ParseKwargs, load, initialize_wandb, log_group_data, parse_bool, get_model_prefix, move_to
from train import train, evaluate, infer_predictions
from algorithms.initializer import initialize_algorithm, infer_d_out
from transforms import initialize_transform
from models.initializer import initialize_model
from configs.utils import populate_defaults
import configs.supported as supported

# python3 wilds_bert-base-uncased.py -d amazon --algorithm ERM --root_dir ./data

def main():
    
    ''' Arg defaults are filled in according to examples/argss/ '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('-d', '--dataset', choices=wilds.supported_datasets, required=True)
    parser.add_argument('--algorithm', required=True, choices=supported.algorithms)
    parser.add_argument('--root_dir', required=True,
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')

    # Dataset
    parser.add_argument('--split_scheme', help='Identifies how the train/val/test split is constructed. Choices are dataset-specific.')
    parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for dataset initialization passed as key1=value1 key2=value2')
    parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                        help='If true, tries to download the dataset if it does not exist in root_dir.')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.')
    parser.add_argument('--version', default=None, type=str, help='WILDS labeled dataset version number.')

    # Unlabeled Dataset
    parser.add_argument('--unlabeled_split', default=None, type=str, choices=wilds.unlabeled_splits,  help='Unlabeled split to use. Some datasets only have some splits available.')
    parser.add_argument('--unlabeled_version', default=None, type=str, help='WILDS unlabeled dataset version number.')
    parser.add_argument('--use_unlabeled_y', default=False, type=parse_bool, const=True, nargs='?', 
                        help='If true, unlabeled loaders will also the true labels for the unlabeled data. This is only available for some datasets. Used for "fully-labeled ERM experiments" in the paper. Correct functionality relies on CrossEntropyLoss using ignore_index=-100.')

    # Loaders
    parser.add_argument('--loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--unlabeled_loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--train_loader', choices=['standard', 'group'])
    parser.add_argument('--uniform_over_groups', type=parse_bool, const=True, nargs='?', help='If true, sample examples such that batches are uniform over groups.')
    parser.add_argument('--distinct_groups', type=parse_bool, const=True, nargs='?', help='If true, enforce groups sampled per batch are distinct.')
    parser.add_argument('--n_groups_per_batch', type=int)
    parser.add_argument('--unlabeled_n_groups_per_batch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--unlabeled_batch_size', type=int)
    parser.add_argument('--eval_loader', choices=['standard'], default='standard')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of batches to process before stepping optimizer and schedulers. If > 1, we simulate having a larger effective batch size (though batchnorm behaves differently).')

    # Model
    parser.add_argument('--model', choices=supported.models)
    parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for model initialization passed as key1=value1 key2=value2')
    parser.add_argument('--noisystudent_add_dropout', type=parse_bool, const=True, nargs='?', help='If true, adds a dropout layer to the student model of NoisyStudent.')
    parser.add_argument('--noisystudent_dropout_rate', type=float)
    parser.add_argument('--pretrained_model_path', default=None, type=str, help='Specify a path to pretrained model weights')
    parser.add_argument('--load_featurizer_only', default=False, type=parse_bool, const=True, nargs='?', help='If true, only loads the featurizer weights and not the classifier weights.')

    # NoisyStudent-specific loading
    parser.add_argument('--teacher_model_path', type=str, help='Path to NoisyStudent teacher model weights. If this is defined, pseudolabels will first be computed for unlabeled data before anything else runs.')

    # Transforms
    parser.add_argument('--transform', choices=supported.transforms)
    parser.add_argument('--additional_train_transform', choices=supported.additional_transforms, help='Optional data augmentations to layer on top of the default transforms.')
    parser.add_argument('--target_resolution', nargs='+', type=int, help='The input resolution that images will be resized to before being passed into the model. For example, use --target_resolution 224 224 for a standard ResNet.')
    parser.add_argument('--resize_scale', type=float)
    parser.add_argument('--max_token_length', type=int)
    parser.add_argument('--randaugment_n', type=int, help='Number of RandAugment transformations to apply.')

    # Objective
    parser.add_argument('--loss_function', choices=supported.losses)
    parser.add_argument('--loss_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for loss initialization passed as key1=value1 key2=value2')

    # Algorithm
    parser.add_argument('--groupby_fields', nargs='+')
    parser.add_argument('--group_dro_step_size', type=float)
    parser.add_argument('--coral_penalty_weight', type=float)
    parser.add_argument('--dann_penalty_weight', type=float)
    parser.add_argument('--dann_classifier_lr', type=float)
    parser.add_argument('--dann_featurizer_lr', type=float)
    parser.add_argument('--dann_discriminator_lr', type=float)
    parser.add_argument('--afn_penalty_weight', type=float)
    parser.add_argument('--safn_delta_r', type=float)
    parser.add_argument('--hafn_r', type=float)
    parser.add_argument('--use_hafn', default=False, type=parse_bool, const=True, nargs='?')
    parser.add_argument('--irm_lambda', type=float)
    parser.add_argument('--irm_penalty_anneal_iters', type=int)
    parser.add_argument('--self_training_lambda', type=float)
    parser.add_argument('--self_training_threshold', type=float)
    parser.add_argument('--pseudolabel_T2', type=float, help='Percentage of total iterations at which to end linear scheduling and hold lambda at the max value')
    parser.add_argument('--soft_pseudolabels', default=False, type=parse_bool, const=True, nargs='?')
    parser.add_argument('--algo_log_metric')
    parser.add_argument('--process_pseudolabels_function', choices=supported.process_pseudolabels_functions)

    # Model selection
    parser.add_argument('--val_metric')
    parser.add_argument('--val_metric_decreasing', type=parse_bool, const=True, nargs='?')

    # Optimization
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--optimizer', choices=supported.optimizers)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for optimizer initialization passed as key1=value1 key2=value2')

    # Scheduler
    parser.add_argument('--scheduler', choices=supported.schedulers)
    parser.add_argument('--scheduler_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for scheduler initialization passed as key1=value1 key2=value2')
    parser.add_argument('--scheduler_metric_split', choices=['train', 'val'], default='val')
    parser.add_argument('--scheduler_metric_name')

    # Evaluation
    parser.add_argument('--process_outputs_function', choices = supported.process_outputs_functions)
    parser.add_argument('--evaluate_all_splits', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--eval_splits', nargs='+', default=[])
    parser.add_argument('--eval_only', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--eval_epoch', default=None, type=int, help='If eval_only is set, then eval_epoch allows you to specify evaluating at a particular epoch. By default, it evaluates the best epoch by validation performance.')

    # Misc
    parser.add_argument('--device', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int)
    parser.add_argument('--save_best', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_last', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_pred', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--no_group_logging', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--progress_bar', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--resume', type=parse_bool, const=True, nargs='?', default=False, help='Whether to resume from the most recent saved model in the current log_dir.')

    # Weights & Biases
    parser.add_argument('--use_wandb', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--wandb_api_key_path', type=str,
                        help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.")
    parser.add_argument('--wandb_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for wandb.init() passed as key1=value1 key2=value2')

    args = parser.parse_args()
    args = populate_defaults(args)
    
    print(f"args: {args}")
    
    # Set device
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if len(args.device) > device_count:
            raise ValueError(f"Specified {len(args.device)} devices, but only {device_count} devices found.")

        args.use_data_parallel = len(args.device) > 1
        device_str = ",".join(map(str, args.device))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        args.device = torch.device("cuda")
    else:
        args.use_data_parallel = False
        args.device = torch.device("cpu")
    
    # Initialize logs
    if os.path.exists(args.log_dir) and args.resume:
        resume=True
        mode='a'
    elif os.path.exists(args.log_dir) and args.eval_only:
        resume=False
        mode='a'
    else:
        resume=False
        mode='w'

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = Logger(os.path.join(args.log_dir, 'log.txt'), mode)

    # Record args
    log_config(args, logger)

    # Set random seed
    set_seed(args.seed)

    # Data
    full_dataset = wilds.get_dataset(
        dataset=args.dataset,
        version=args.version,
        root_dir=args.root_dir,
        download=args.download,
        split_scheme=args.split_scheme,
        **args.dataset_kwargs)

    # Transforms & data augmentations for labeled dataset
    # To modify data augmentation, modify the following code block.
    # If you want to use transforms that modify both `x` and `y`,
    # set `do_transform_y` to True when initializing the `WILDSSubset` below.
    train_transform = initialize_transform(
        transform_name=args.transform,
        config=args,
        dataset=full_dataset,
        additional_transform_name=args.additional_train_transform,
        is_training=True)
    eval_transform = initialize_transform(
        transform_name=args.transform,
        config=args,
        dataset=full_dataset,
        is_training=False)

    # argsure unlabeled datasets
    unlabeled_dataset = None
    if args.unlabeled_split is not None:
        split = args.unlabeled_split
        full_unlabeled_dataset = wilds.get_dataset(
            dataset=args.dataset,
            version=args.unlabeled_version,
            root_dir=args.root_dir,
            download=args.download,
            unlabeled=True,
            **args.dataset_kwargs
        )
        train_grouper = CombinatorialGrouper(
            dataset=[full_dataset, full_unlabeled_dataset],
            groupby_fields=args.groupby_fields
        )

        # Transforms & data augmentations for unlabeled dataset
        if args.algorithm == "FixMatch":
            # For FixMatch, we need our loader to return batches in the form ((x_weak, x_strong), m)
            # We do this by initializing a special transform function
            unlabeled_train_transform = initialize_transform(
                args.transform, args, full_dataset, is_training=True, additional_transform_name="fixmatch"
            )
        else:
            # Otherwise, use the same data augmentations as the labeled data.
            unlabeled_train_transform = train_transform

        if args.algorithm == "NoisyStudent":
            # For Noisy Student, we need to first generate pseudolabels using the teacher
            # and then prep the unlabeled dataset to return these pseudolabels in __getitem__
            print("Inferring teacher pseudolabels for Noisy Student")
            assert args.teacher_model_path is not None
            if not args.teacher_model_path.endswith(".pth"):
                # Use the best model
                args.teacher_model_path = os.path.join(
                    args.teacher_model_path,  f"{args.dataset}_seed:{args.seed}_epoch:best_model.pth"
                )

            d_out = infer_d_out(full_dataset, args)
            teacher_model = initialize_model(args, d_out).to(args.device)
            load(teacher_model, args.teacher_model_path, device=args.device)
            # Infer teacher outputs on weakly augmented unlabeled examples in sequential order
            weak_transform = initialize_transform(
                transform_name=args.transform,
                args=args,
                dataset=full_dataset,
                is_training=True,
                additional_transform_name="weak"
            )
            unlabeled_split_dataset = full_unlabeled_dataset.get_subset(split, transform=weak_transform, frac=args.frac)
            sequential_loader = get_eval_loader(
                loader=args.eval_loader,
                dataset=unlabeled_split_dataset,
                grouper=train_grouper,
                batch_size=args.unlabeled_batch_size,
                **args.unlabeled_loader_kwargs
            )
            teacher_outputs = infer_predictions(teacher_model, sequential_loader, args)
            teacher_outputs = move_to(teacher_outputs, torch.device("cpu"))
            unlabeled_split_dataset = WILDSPseudolabeledSubset(
                reference_subset=unlabeled_split_dataset,
                pseudolabels=teacher_outputs,
                transform=unlabeled_train_transform,
                collate=full_dataset.collate,
            )
            teacher_model = teacher_model.to(torch.device("cpu"))
            del teacher_model
        else:
            unlabeled_split_dataset = full_unlabeled_dataset.get_subset(
                split, 
                transform=unlabeled_train_transform, 
                frac=args.frac, 
                load_y=args.use_unlabeled_y
            )

        unlabeled_dataset = {
            'split': split,
            'name': full_unlabeled_dataset.split_names[split],
            'dataset': unlabeled_split_dataset
        }
        unlabeled_dataset['loader'] = get_train_loader(
            loader=args.train_loader,
            dataset=unlabeled_dataset['dataset'],
            batch_size=args.unlabeled_batch_size,
            uniform_over_groups=args.uniform_over_groups,
            grouper=train_grouper,
            distinct_groups=args.distinct_groups,
            n_groups_per_batch=args.unlabeled_n_groups_per_batch,
            **args.unlabeled_loader_kwargs
        )
    else:
        train_grouper = CombinatorialGrouper(
            dataset=full_dataset,
            groupby_fields=args.groupby_fields
        )

    # argsure labeled torch datasets (WILDS dataset splits)
    datasets = defaultdict(dict)
    for split in full_dataset.split_dict.keys():
        if split=='train':
            transform = train_transform
            verbose = True
        elif split == 'val':
            transform = eval_transform
            verbose = True
        else:
            transform = eval_transform
            verbose = False
        # Get subset
        datasets[split]['dataset'] = full_dataset.get_subset(
            split,
            frac=args.frac,
            transform=transform)

        if split == 'train':
            datasets[split]['loader'] = get_train_loader(
                loader=args.train_loader,
                dataset=datasets[split]['dataset'],
                batch_size=args.batch_size,
                uniform_over_groups=args.uniform_over_groups,
                grouper=train_grouper,
                distinct_groups=args.distinct_groups,
                n_groups_per_batch=args.n_groups_per_batch,
                **args.loader_kwargs)
        else:
            datasets[split]['loader'] = get_eval_loader(
                loader=args.eval_loader,
                dataset=datasets[split]['dataset'],
                grouper=train_grouper,
                batch_size=args.batch_size,
                **args.loader_kwargs)

        # Set fields
        datasets[split]['split'] = split
        datasets[split]['name'] = full_dataset.split_names[split]
        datasets[split]['verbose'] = verbose

        # Loggers
        datasets[split]['eval_logger'] = BatchLogger(
            os.path.join(args.log_dir, f'{split}_eval.csv'), mode=mode, use_wandb=args.use_wandb
        )
        datasets[split]['algo_logger'] = BatchLogger(
            os.path.join(args.log_dir, f'{split}_algo.csv'), mode=mode, use_wandb=args.use_wandb
        )    
    


    from transformers import (
        BertConfig,
        BertTokenizer,
        BertForSequenceClassification,
        AdamW,
    )
    args.vanilla_training = True
    args.model_type = "bert"
    args.config_name = ""
    args.model_name_or_path = "bert-base-uncased"
    args.tokenizer_name = ""
    args.cache_dir = ""
    args.task_name = "amazon"
    args.do_lower_case = True
    args.weight_decay = 0.0
    MODEL_CLASSES = {
        "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
        # "roberta": (RobertaConfig, DeeRobertaForSequenceClassification, RobertaTokenizer),
        # "distilbert": (DistilBertConfig, DeeDistilBertForSequenceClassification, DistilBertTokenizer),
    }
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    if args.vanilla_training:
        vanilla_model_dict = {
            "bert": BertForSequenceClassification,
            # "roberta": RobertaForSequenceClassification,
            # "distilbert": DistilBertForSequenceClassification,
        }
        model_class = vanilla_model_dict[args.model_type]
        num_labels = 5

        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model = model_class.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        
        print(f"config {config}")
    
    # train(
    #     model=model,
    #     datasets=datasets,
    #     general_logger=logger,
    #     args=args,
    #     epoch_offset=epoch_offset,
    #     best_val_metric=best_val_metric,
    #     unlabeled_dataset=unlabeled_dataset,
    # )
    
    train(args, datasets["train"], model, tokenizer)


def train(args, train_dataset, model, tokenizer):
    train_dataloader = train_dataset["loader"]
    
    # select params whether ee or vanilla
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
            ],
            "weight_decay": args.weight_decay,
        }
    ]
    
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    for step, batch in enumerate(train_dataloader):
        model.train()
        print(type(batch), len(batch))
        print(batch[0].size())
        print(batch[1].size())
        print(batch[2].size())
        
        inputs = {"input_ids": batch[:, :, 0], "attention_mask": batch[:, :, 1], "token_type_ids": batch[:, :, 2]}
        
        outputs = model(**inputs)
        print(len(outputs))
        print(type(outputs[0]))
        loss.backward()
        
        exit(0)
        
    
    return

if __name__=='__main__':
    main()
