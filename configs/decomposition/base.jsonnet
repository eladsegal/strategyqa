{
  pretrained_model:: "facebook/bart-large",
  dataset_reader_type:: "strategy_qa_decomposition_reader",
  num_epochs:: 15,
  batch_size_per_gpu:: 2,
  num_gradient_accumulation_steps:: 16, // batch size is batch_size_per_gpu * num_gradient_accumulation_steps * NUM_OF_GPUS
  archive_model_after_each_epoch:: false,
  "dataset_reader": {
    "type": $.dataset_reader_type,
    "tokenizer_wrapper": {
      "pretrained_model": $.pretrained_model,
      "init_kwargs": {
          "add_prefix_space": true,
      },
    },
    "save_tokenizer": true,
    "is_training": true,
  },
  "validation_dataset_reader": $.dataset_reader + {
    "save_tokenizer": false,
    "is_training": false,
  },
  "train_data_path": "data/strategy_qa/train.json",
  "validation_data_path": "data/strategy_qa/dev.json",
  "model": {
    "type": "gen",
    "pretrained_model":  $.pretrained_model,
    "tokenizer_wrapper": $.dataset_reader.tokenizer_wrapper,
    "metrics": {
        "bleu": {
            "type": "bleu",
        },
        "rouge": {
            "type": "rouge",
        },
        "sari": {
            "type": "sari",
            "is_main": true,
        },
    },
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
    "optimizer": {
        "type": "huggingface_adamw",
        "lr": 3e-5,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "correct_bias": true
    },
    "learning_rate_scheduler": {
        "type": "polynomial_decay",
    },
    "num_gradient_accumulation_steps": $.num_gradient_accumulation_steps,
    "checkpointer": {
        "num_serialized_models_to_keep": if $.archive_model_after_each_epoch then 2 else -1
    },
    "validation_metric": "+SARI",
  },
  "random_seed": 42,
  "numpy_seed": 42,
  "pytorch_seed": 42,
}
