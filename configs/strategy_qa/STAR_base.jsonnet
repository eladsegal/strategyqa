local config = import "base.jsonnet";

config {
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
  }
}