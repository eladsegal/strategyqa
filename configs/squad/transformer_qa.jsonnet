local pretrained_init = import "models/_snippets/pretrained_init.jsonnet";

{
    use_pretrained_init:: false,
    pretrained_model:: "roberta-base",
    seed:: 42,
    lr:: 3e-5,
    epochs:: 10,
    batch_size:: 8,
    num_gradient_accumulation_steps:: 1,
    training_max_instances:: null,
    validation_max_instances:: null,
    pickle_action:: "load",  # "load" (load if available, otherwise save) / "save" (just save) / null (do nothing)
    pickle_file_name:: "squad_v2_transformer_qa",
    pickle_save_even_when_max_instances:: false,
    archive_model_after_each_epoch:: false,
    model_type:: "transformer_qa_v2",
    dataset_reader_type:: "general_squad",
    mode:: "span_extraction",
    mode_config:: {},
    tokenizer_init_kwargs:: {
        "add_prefix_space": true, // needed in some cases in huggingface, but should be used carefully
    },
    tokenizer_call_kwargs:: {},
    "model": {
        "type": $.model_type,
        "pretrained_model": $.pretrained_model,
        "tokenizer_wrapper": $.dataset_reader.tokenizer_wrapper,
        "enable_no_answer": true,
    } + (if $.use_pretrained_init then pretrained_init else {}),
    "dataset_reader": {
        "type": $.dataset_reader_type,
        "tokenizer_wrapper": {
            "pretrained_model": $.pretrained_model,
            "init_kwargs": $.tokenizer_init_kwargs,
            "call_kwargs": $.tokenizer_call_kwargs
        },
        "is_training": true,
        "save_tokenizer": true,
        "max_instances": $.training_max_instances,
        "pickle": {
            "action": $.pickle_action,
            "file_name": $.pickle_file_name,
            "path": "../pickle",
            "save_even_when_max_instances": $.pickle_save_even_when_max_instances,
        },
    },
    "validation_dataset_reader": self.dataset_reader + {
        "is_training": false, 
        "save_tokenizer": false,
        "max_instances": $.validation_max_instances
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": $.batch_size
        }
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "weight_decay": 0.01,
            "lr": $.lr,
            "betas": [0.9, 0.98],
            "eps": 1e-8,
            "correct_bias": true
        },
        "learning_rate_scheduler": {
            "type": "polynomial_decay",
        },
        "num_epochs": $.epochs,
        "num_gradient_accumulation_steps": $.num_gradient_accumulation_steps,
        "checkpointer": {
            "num_serialized_models_to_keep": if $.archive_model_after_each_epoch then 2 else -1
        },
        "validation_metric": "+f1",
    },
    "train_data_path": "data/squad_v2/squad_v2_boolq_dataset_train.json",
    "validation_data_path": "data/squad_v2/squad_v2_boolq_dataset_dev.json",
    "random_seed": $.seed,
    "numpy_seed": $.seed,
    "pytorch_seed": $.seed
}