local config = import "STAR_base.jsonnet";

config {
    epochs:: 10,
    paragraphs_source:: "ORA-P",
    "dataset_reader"+: {
        "answer_last_decomposition_step": true,
        "skip_if_context_missing": false,
    },
}