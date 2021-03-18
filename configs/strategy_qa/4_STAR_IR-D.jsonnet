local config = import "STAR_base.jsonnet";

config {
      paragraphs_source:: "IR-D",
      "dataset_reader"+: {
            "generated_decompositions_paths": [
                  "data/strategyqa/generated/bart_decomp_dev_predictions.jsonl",
            ]
      },
      "model"+: {
          "initializer": {}
      },
      "train_data_path": null,
}
