local config = import "base.jsonnet";

config {
  num_epochs:: 1,
  batch_size_per_gpu:: 8,
  num_gradient_accumulation_steps:: 4,
  with_context:: false,
  "train_data_path": "data/twentyquestions/twentyquestions-train.jsonl",
  "validation_data_path": "data/twentyquestions/twentyquestions-dev.jsonl",
  "dataset_reader"+: {
    "answer_key": "majority",
    "is_twenty_questions": true,
  },
  "model"+: {
    "transformer_weights_path": std.extVar("DROP_WEIGHTS"),
  },
  "trainer"+: {
    "checkpointer": {
        "num_serialized_models_to_keep": -1
    },
  },
}