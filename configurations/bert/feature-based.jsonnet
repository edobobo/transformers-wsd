/* params */

local seed = 1213;

local target_namespace = 'senses';

local batch_size = 16;
local gradient_accumulation = 1;

/* encoder */

local encoder = {
    type: 'transformers',
    transformer_model: 'bert-base-cased',
    training_strategy: 'feature-based',
    use_last_n_layers: 4
};

local cached_encoder = {
    type: 'sense-mask-only-cached-encoder',
    backend_encoder: encoder
};

/* train and dev */

local train_path = '
    [
        ["data/datasets/train/SemCor/semcor", 1.0]
    ]
';

local dev_path = 'data/datasets/eval/WSD_Unified_Evaluation_Datasets/semeval2007/semeval2007';

/* config */

{
    dataset_reader: {
        type: 'multi',
        batch_size: batch_size,
        dataset_readers: [
            {
                type: 'raganato',
                encoder: encoder,
                wsd_instance_conversion_strategy: 'identity',
                multilabel_selection_strategy: 'always-first'
            }
        ]
    },
    validation_dataset_reader: {
        type: 'raganato',
        encoder: encoder,
        wsd_instance_conversion_strategy: 'identity',
        multilabel_selection_strategy: 'always-first',
    },
    train_data_path: train_path,
    validation_data_path: dev_path,
    model: {
        type: 'wsd',
        encoder: cached_encoder,
        classification_head: {
            type: 'linear',
            optimize_on_training: true
        },
        target_namespace: target_namespace
    },
    data_loader: {
        batch_size: batch_size,
        shuffle: true
    },
    validation_data_loader: {
        batch_size: batch_size
    },
    trainer: {
        type: 'gradient_descent',
        optimizer: {
            type: "adam",
            lr: 2e-5
        },
        num_epochs: 50,
        patience: 3,
        validation_metric: '+accuracy3',
        cuda_device: 0,
        num_gradient_accumulation_steps: gradient_accumulation,
        checkpointer: {
            num_serialized_models_to_keep: 3
        },
        epoch_callbacks: [
            {
                type: 'wandb',
                project_name: 'mulan',
                run_name: std.extVar('wandb_run_name')
            }
        ]
    },
    random_seed: seed,
    numpy_seed: seed,
    pytorch_seed: seed
}
