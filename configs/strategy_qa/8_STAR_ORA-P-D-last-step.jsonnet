local config = import "STAR_base.jsonnet";

config {
    epochs:: 10,
    paragraphs_source:: "ORA-P",
    "dataset_reader"+: {
        "answer_last_decomposition_step": true,
        "skip_if_context_missing": false,
    },
    "train_data_path": "data/strategyqa/generated/transformer_qa_ORA-P_train_no_placeholders.json",
    "validation_data_path": "data/strategyqa/generated/transformer_qa_ORA-P_dev_no_placeholders.json",
}