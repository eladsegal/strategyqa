local dropout = 0.1;
{
  pretrained_model:: "roberta-large",
  transformer_dim:: 1024,
  cls_is_last_token:: false,
  dataset_reader_type:: "strategy_qa_reader",
  paragraphs_source:: error "Must override paragraphs_source",
  num_epochs:: 15,
  batch_size_per_gpu:: 2,
  num_gradient_accumulation_steps:: 16, // batch size is batch_size_per_gpu * num_gradient_accumulation_steps * NUM_OF_GPUS
  pickle_action:: null,  # "load" (load if available, otherwise save) / "save" (just save) / null (do nothing)
  pickle_file_name:: if $.pickle_action != null then error "Must override pickle_file_name" else null,
  pickle_save_even_when_max_instances:: false,
  "dataset_reader": {
    "type": $.dataset_reader_type,
    "save_tokenizer": true,
    "tokenizer_wrapper": {
      "pretrained_model": $.pretrained_model,
      "call_kwargs": {
        "truncation": "only_first",
        "return_offsets_mapping": true,
        "return_special_tokens_mask": true,
      }
    },
    "is_training": true,
    "paragraphs_source": $.paragraphs_source,
    "pickle": {
        "action": $.pickle_action,
        "file_name": $.pickle_file_name,
        "path": "../pickle",
        "save_even_when_max_instances": $.pickle_save_even_when_max_instances,
    },
  },
  "validation_dataset_reader": $.dataset_reader + {
    "save_tokenizer": false,
    "is_training": false,
  },
  "train_data_path": "data/strategyqa/train.json",
  "validation_data_path": "data/strategyqa/dev.json",
  "evaluate_on_test": true,
  "model": {
    "type": "hf_classifier",
    "pretrained_model":  $.pretrained_model,
    "tokenizer_wrapper": $.dataset_reader.tokenizer_wrapper,
    "num_labels": 2,
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : $.batch_size_per_gpu
    }
  },
  "trainer": {
    "num_epochs": $.num_epochs,
    "cuda_device" : -1,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 1e-5,
      "weight_decay": 0.1,
      "eps": 1e-6,
      "betas": [0.9, 0.98],
    },
    "num_gradient_accumulation_steps": $.num_gradient_accumulation_steps
  },
  "random_seed": 42,
  "numpy_seed": 42,
  "pytorch_seed": 42,
}
