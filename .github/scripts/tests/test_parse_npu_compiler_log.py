# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess  # nosec B404
import sys
import textwrap
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "parse_npu_compiler_log.py"


def run_parser(tmp_path: Path, *, platform: str, framework: str, test_type: str, log_text: str, output_name: str = "metrics.json"):
    log_path = tmp_path / f"{test_type}.log"
    log_path.write_text(textwrap.dedent(log_text).strip() + "\n", encoding="utf-8")
    output_path = tmp_path / output_name
    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            str(SCRIPT),
            "--platform",
            platform,
            "--framework",
            framework,
            "--test-type",
            test_type,
            "--input",
            str(log_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    return result, output_path


def test_duplicate_model_names_are_preserved_per_test_type(tmp_path):
    convert_result, output_path = run_parser(
        tmp_path,
        platform="3720",
        framework="tensorflow",
        test_type="convert_model",
        log_text="""
        tests/model_hub_tests/tensorflow/test_tf_convert_model.py::TestTFConvertModel::test_precommit[NPU-imagenet/resnet_v2_50/feature_vector] PASSED
        Compilation memory usage: Peak 123.0 KB
        Compile net time: 45.0 ms
        """,
    )
    assert convert_result.returncode == 0, convert_result.stderr

    read_result, _ = run_parser(
        tmp_path,
        platform="3720",
        framework="tensorflow",
        test_type="read_model",
        log_text="""
        tests/model_hub_tests/tensorflow/test_tf_read_model.py::TestTFReadModel::test_precommit[NPU-imagenet/resnet_v2_50/feature_vector] PASSED
        Compilation memory usage: Peak 456.0 KB
        Compile net time: 78.0 ms
        """,
    )
    assert read_result.returncode == 0, read_result.stderr

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "3720": {
            "tensorflow": {
                "convert_model": {
                    "NPU-imagenet/resnet_v2_50/feature_vector": {
                        "compilation_memory_usage_kb": 123.0,
                        "compile_net_time_ms": 45.0,
                    }
                },
                "read_model": {
                    "NPU-imagenet/resnet_v2_50/feature_vector": {
                        "compilation_memory_usage_kb": 456.0,
                        "compile_net_time_ms": 78.0,
                    }
                },
            }
        }
    }


def test_rerun_updates_only_selected_test_type_and_preserves_other_entries(tmp_path):
    output_path = tmp_path / "metrics.json"
    output_path.write_text(
        json.dumps(
            {
                "3720": {
                    "tensorflow": {
                        "convert_model": {
                            "NPU-model-a": {
                                "compilation_memory_usage_kb": 10.0,
                                "compile_net_time_ms": 11.0,
                            }
                        },
                        "read_model": {
                            "NPU-model-a": {
                                "compilation_memory_usage_kb": 20.0,
                                "compile_net_time_ms": 21.0,
                            }
                        },
                    },
                    "jax": {
                        "jax": {
                            "NPU-jax-model": {
                                "compilation_memory_usage_kb": 30.0,
                                "compile_net_time_ms": 31.0,
                            }
                        }
                    },
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result, _ = run_parser(
        tmp_path,
        platform="3720",
        framework="tensorflow",
        test_type="convert_model",
        log_text="""
        tests/model_hub_tests/tensorflow/test_tf_convert_model.py::TestTFConvertModel::test_precommit[NPU-model-a] PASSED
        Compilation memory usage: Peak 110.0 KB
        Compile net time: 111.0 ms
        tests/model_hub_tests/tensorflow/test_tf_convert_model.py::TestTFConvertModel::test_precommit[NPU-model-b] PASSED
        Compilation memory usage: Peak 210.0 KB
        Compile net time: 211.0 ms
        """,
    )
    assert result.returncode == 0, result.stderr

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "3720": {
            "tensorflow": {
                "convert_model": {
                    "NPU-model-a": {
                        "compilation_memory_usage_kb": 110.0,
                        "compile_net_time_ms": 111.0,
                    },
                    "NPU-model-b": {
                        "compilation_memory_usage_kb": 210.0,
                        "compile_net_time_ms": 211.0,
                    },
                },
                "read_model": {
                    "NPU-model-a": {
                        "compilation_memory_usage_kb": 20.0,
                        "compile_net_time_ms": 21.0,
                    }
                },
            },
            "jax": {
                "jax": {
                    "NPU-jax-model": {
                        "compilation_memory_usage_kb": 30.0,
                        "compile_net_time_ms": 31.0,
                    }
                }
            },
        }
    }


def test_metric_extraction_handles_ansi_and_alternative_time_patterns(tmp_path):
    result, output_path = run_parser(
        tmp_path,
        platform="4000",
        framework="pytorch",
        test_type="pt_groupA",
        log_text="""
        \x1b[32mtests/model_hub_tests/pytorch/test_timm.py::TestTimm::test_precommit[NPU-resnet18]\x1b[0m PASSED
        Compilation memory usage: Peak 2048 KB
        Compile model took 15.5 ms
        """,
    )
    assert result.returncode == 0, result.stderr

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "4000": {
            "pytorch": {
                "pt_groupA": {
                    "NPU-resnet18": {
                        "compilation_memory_usage_kb": 2048.0,
                        "compile_net_time_ms": 15.5,
                    }
                }
            }
        }
    }


def test_legacy_framework_schema_is_rejected_deliberately(tmp_path):
    output_path = tmp_path / "metrics.json"
    output_path.write_text(
        json.dumps(
            {
                "3720": {
                    "tensorflow": {
                        "NPU-model-a": {
                            "compilation_memory_usage_kb": 10.0,
                            "compile_net_time_ms": 11.0,
                        }
                    }
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result, _ = run_parser(
        tmp_path,
        platform="3720",
        framework="tensorflow",
        test_type="convert_model",
        log_text="""
        tests/model_hub_tests/tensorflow/test_tf_convert_model.py::TestTFConvertModel::test_precommit[NPU-model-a] PASSED
        Compilation memory usage: Peak 110.0 KB
        Compile net time: 111.0 ms
        """,
    )

    assert result.returncode != 0
    assert "unexpected test-type bucket" in result.stderr


def test_empty_namespaced_bucket_is_preserved_and_reused(tmp_path):
    output_path = tmp_path / "metrics.json"
    output_path.write_text(
        json.dumps(
            {
                "3720": {
                    "tensorflow": {
                        "convert_model": {},
                    }
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result, _ = run_parser(
        tmp_path,
        platform="3720",
        framework="tensorflow",
        test_type="read_model",
        log_text="""
        tests/model_hub_tests/tensorflow/test_tf_read_model.py::TestTFReadModel::test_precommit[NPU-model-a] PASSED
        Compilation memory usage: Peak 110.0 KB
        Compile net time: 111.0 ms
        """,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "3720": {
            "tensorflow": {
                "convert_model": {},
                "read_model": {
                    "NPU-model-a": {
                        "compilation_memory_usage_kb": 110.0,
                        "compile_net_time_ms": 111.0,
                    }
                },
            }
        }
    }
