local dropout = 0.1;
{
  pretrained_model:: "roberta-large",
  dataset_reader_type:: "boolean_qa_reader",
  num_epochs:: 10,
  batch_size_per_gpu:: 2,
  with_context:: error "Must override with_context",
  num_gradient_accumulation_steps:: 16, // batch size is batch_size_per_gpu * num_gradient_accumulation_steps * NUM_OF_GPUS
  "dataset_reader": {
    "type": $.dataset_reader_type,
    "save_tokenizer": true,
    "tokenizer_wrapper": {
      "pretrained_model": $.pretrained_model,
      "call_kwargs": {
        "truncation": "only_first",
      }
    },
    "is_training": true,
    "with_context": $.with_context,
  },
  "validation_dataset_reader": $.dataset_reader + {
    "save_tokenizer": false,
    "is_training": false,
  },
  "train_data_path": error "Must override train_data_path",
  "validation_data_path": error "Must override validation_data_path",
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
