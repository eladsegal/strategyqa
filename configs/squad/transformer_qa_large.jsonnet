local config = import "transformer_qa.jsonnet";

config {
    pretrained_model:: "roberta-large",
    epochs:: 3,
    lr:: 1e-5,
    batch_size:: 2,
}