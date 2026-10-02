"""Valid and invalid fixtures for the DFlash Open Spec Config schema.

Each fixture is the example config with one change applied.

Run with: pip install "jsonschema>=4.18" pytest && pytest tests
"""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

SPEC_DIR = Path(__file__).resolve().parents[1] / "src" / "open-spec-config"
SCHEMA = json.loads((SPEC_DIR / "schema" / "dflash.schema.json").read_text())
EXAMPLE = json.loads((SPEC_DIR / "examples" / "dflash.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)

SC = "speculative_config"
ARCH_FIELDS = [
    k
    for k in EXAMPLE
    if k
    not in (
        "open_spec_config_version",
        "architectures",
        SC,
    )
]


def set_(path, value):
    def apply(cfg):
        *parents, key = path
        for p in parents:
            cfg = cfg[p]
        cfg[key] = value

    return apply


def del_(*paths):
    def apply(cfg):
        for path in paths:
            node = cfg
            *parents, key = path
            for p in parents:
                node = node[p]
            del node[key]

    return apply


VALID = {
    "example": lambda cfg: None,
    "minimal": del_(
        [SC, "verifier"],
        [SC, "draft_vocab_size"],
        [SC, "training_framework"],
        [SC, "training_framework_version"],
        *[[k] for k in ARCH_FIELDS],
    ),
    "one_indexed_target_layers": set_([SC, "target_layer_start_idx"], 1),
    "sample_from_anchor": set_([SC, "sample_from_anchor"], True),
    "framework_without_version": del_([SC, "training_framework_version"]),
    "extra_architecture_field": set_(["some_new_model_field"], 123),
}

INVALID = {
    "empty_object": lambda cfg: cfg.clear(),
    "missing_version": del_(["open_spec_config_version"]),
    "wrong_version": set_(["open_spec_config_version"], "1.0.0"),
    "missing_architectures": del_(["architectures"]),
    "empty_architectures": set_(["architectures"], []),
    "other_architecture": set_(["architectures"], ["DsparkDraftModel"]),
    "extra_architecture": set_(["architectures"], ["DflashDraftModel", "Qwen3ForCausalLM"]),
    "empty_training_framework": set_([SC, "training_framework"], ""),
    "empty_training_framework_version": set_([SC, "training_framework_version"], ""),
    "missing_speculative_config": del_([SC]),
    "unknown_method": set_([SC, "method"], "eagle3"),
    "zero_speculative_tokens": set_([SC, "speculative_tokens"], 0),
    "negative_speculative_tokens": set_([SC, "speculative_tokens"], -1),
    "fractional_speculative_tokens": set_([SC, "speculative_tokens"], 7.5),
    "empty_verifier": set_([SC, "verifier"], ""),
    "empty_target_layer_ids": set_([SC, "target_layer_ids"], []),
    "negative_target_layer_id": set_([SC, "target_layer_ids"], [-1, 2]),
    "duplicate_target_layer_ids": set_([SC, "target_layer_ids"], [2, 2]),
    "bad_target_layer_start_idx": set_([SC, "target_layer_start_idx"], 2),
    "zero_draft_vocab_size": set_([SC, "draft_vocab_size"], 0),
    "negative_mask_token_id": set_([SC, "mask_token_id"], -1),
    "string_sample_from_anchor": set_([SC, "sample_from_anchor"], "false"),
    "missing_sample_from_anchor": del_([SC, "sample_from_anchor"]),
    "missing_mask_token_id": del_([SC, "mask_token_id"]),
    "missing_sliding_window_non_causal": del_([SC, "sliding_window_non_causal"]),
    "unknown_speculative_field": set_([SC, "block_size"], 16),
    "nested_dflash_config": set_([SC, "dflash_config"], {"mask_token_id": 0}),
    "old_speculative_library_name": set_(
        [SC, "speculative_library"], "vllm-project/speculators"
    ),
}


def test_schema_is_valid():
    Draft202012Validator.check_schema(SCHEMA)


@pytest.mark.parametrize("change", VALID.values(), ids=VALID.keys())
def test_valid(change):
    cfg = copy.deepcopy(EXAMPLE)
    change(cfg)
    errors = [e.message for e in VALIDATOR.iter_errors(cfg)]
    assert not errors, errors


@pytest.mark.parametrize("change", INVALID.values(), ids=INVALID.keys())
def test_invalid(change):
    cfg = copy.deepcopy(EXAMPLE)
    change(cfg)
    assert not VALIDATOR.is_valid(cfg)
