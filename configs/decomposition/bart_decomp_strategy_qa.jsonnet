local config = import "base.jsonnet";

config {
  dataset_reader_type:: "strategy_decomposition_reader",
  "train_data_path": "data/strategyqa/train.json",
  "validation_data_path": "data/strategyqa/dev.json",
}
