local config = import "base.jsonnet";

config {
  with_context:: true,
  "train_data_path": "data/boolq/train.jsonl",
  "validation_data_path": "data/boolq/dev.jsonl",
  "dataset_reader"+: {
    "context_key": "passage"
  },
  "model"+: {
    "initializer": {
      "regexes": [
        [".*",
          {
            "type": "pretrained",
            "weights_file_path": std.extVar("weights"),
            "parameter_name_overrides": {}
          }
        ]
      ]
    }
  },
  "trainer"+: {
    "checkpointer": {
        "num_serialized_models_to_keep": -1
    },
  },
}