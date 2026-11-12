# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CPU-only CLI regressions. Run with SIT_BINARY pointing at a built SIT."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import openvino as ov
from openvino import opset13 as ops


def heads(score=0.9, link=0.9, level=2):
    scores = np.zeros((1, 1, 3, 6), dtype=np.float32)
    scores[0, 0, 1, 1:3] = score
    deltas = np.zeros((1, 8, 3, 6), dtype=np.float32)
    deltas[:, 0::2] = np.array([-0.25, 0.25, 0.25, -0.25])[None, :, None, None]
    deltas[:, 1::2] = np.array([-0.25, -0.25, 0.25, 0.25])[None, :, None, None]
    return {
        f"scores_hori_fpn{level}": scores,
        f"scores_vert_fpn{level}": np.zeros_like(scores),
        f"bbox_deltas_hori_fpn{level}": deltas,
        f"bbox_deltas_vert_fpn{level}": deltas.copy(),
        f"link_scores_hori_fpn{level}": np.full_like(deltas, link),
        f"link_scores_vert_fpn{level}": np.full_like(deltas, link),
    }


class PixelLinkCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = Path(os.environ["SIT_BINARY"]).resolve(strict=True)
        if os.environ.get("SIT_TEST_OUTPUT_DIR"):
            cls.root = Path(os.environ["SIT_TEST_OUTPUT_DIR"]).resolve()
            cls.root.mkdir(parents=True, exist_ok=False)
        else:
            cls.temporary = tempfile.TemporaryDirectory(prefix="sit-pixellink-")
            cls.addClassCleanup(cls.temporary.cleanup)
            cls.root = Path(cls.temporary.name)

    def setUp(self):
        self.case = self.root / self.id().split(".")[-1]
        self.case.mkdir()
        self.invocations = 0

    def run_sit(self, values, flags=(), references=None):
        self.invocations += 1
        folder = self.case / str(self.invocations)
        folder.mkdir()
        parameter = ops.parameter([1], np.float32, name="data")
        outputs = []
        for name, value in values.items():
            output = ops.add(ops.constant(value), parameter)
            output.set_friendly_name(name)
            output.output(0).get_tensor().set_names({name})
            outputs.append(output)
        model = ov.Model(outputs, [parameter], "sit_regression")
        network = folder / "model.xml"
        ov.save_model(model, network, compress_to_fp16=False)
        input_path = folder / "input.bin"
        np.zeros(1, dtype=np.float32).tofile(input_path)
        ref_paths = []
        for index, name in enumerate(values):
            path = folder / f"ref_{index}.bin"
            (references or values)[name].tofile(path)
            ref_paths.append(str(path))
        command = [
            str(self.binary), "--network", str(network), "--device", "CPU",
            "--input", str(input_path), "--img_as_bin", "--ip", "FP32", "--op", "FP32",
            "--run_test", "--mode", "map", "--map_threshold", "0.99",
            "--ref_results", ",".join(ref_paths),
            *flags,
        ]
        result = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=60)
        (folder / "command.txt").write_text("\n".join(command) + "\n")
        (folder / "sit.log").write_text(result.stdout + result.stderr)
        return result

    def assert_pass(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("100.0000%", result.stdout)
        self.assertIn("PASSED", result.stdout)

    def assert_failure(self, result, message):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(message, result.stdout + result.stderr)

    def test_poc_boxes_and_nondefault_dimensions(self):
        result = self.run_sit(heads(), ["--pixellink_image_size", "30,60"])
        self.assert_pass(result)
        self.assertIn("GT: 1", result.stdout)

    def test_score_rounding_matches_numpy(self):
        score = np.float32(0.5505)
        self.assertEqual(float(np.round(score, 3)), float(np.float32(0.55)))
        result = self.run_sit(heads(score=score))
        self.assert_failure(result, "No predictions found")
        self.assertNotIn("Could not determine detection format", result.stdout)

    def test_cross_scale_nms(self):
        values = heads()
        second = {}
        for name, value in heads(level=3).items():
            second[name] = np.zeros((1, value.shape[1], 6, 12), dtype=np.float32)
        second["scores_hori_fpn3"][0, 0, 2, 2:4] = 0.9
        second["link_scores_hori_fpn3"].fill(0.9)
        for corner, (x, y) in enumerate(((0.75, 0.75), (2.25, 0.75), (2.25, 1.25), (0.75, 1.25))):
            second["bbox_deltas_hori_fpn3"][0, 2 * corner, 2, 2:4] = 2 * x - np.array([2, 3])
            second["bbox_deltas_hori_fpn3"][0, 2 * corner + 1, 2, 2:4] = 2 * y - 2
        values.update(second)
        suppressed = self.run_sit(values)
        self.assert_pass(suppressed)
        self.assertIn("GT: 1", suppressed.stdout)
        unsuppressed = self.run_sit(values, ["--pixellink_nms_iou", "1"])
        self.assert_pass(unsuppressed)
        self.assertIn("GT: 2", unsuppressed.stdout)

    def test_empty_and_weak_links_are_recognized(self):
        for values in (heads(score=0), heads(link=0.5)):
            with self.subTest():
                result = self.run_sit(values)
                self.assert_failure(result, "No predictions found")
                self.assertNotIn("Could not determine detection format", result.stdout)

    def test_optional_closing_can_join_components(self):
        values = heads()
        score = values["scores_hori_fpn2"]
        score.fill(0)
        score[0, 0, 1, 1:3] = 0.9
        score[0, 0, 1, 4:6] = 0.9
        separate = self.run_sit(values)
        self.assert_pass(separate)
        self.assertIn("GT: 2", separate.stdout)
        joined = self.run_sit(values, ["--pixellink_morph_close"])
        self.assert_pass(joined)
        self.assertIn("GT: 1", joined.stdout)

    def test_malformed_heads_fail_explicitly(self):
        missing = heads()
        del missing["bbox_deltas_vert_fpn2"]
        duplicate = heads()
        duplicate["other_scores_hori_fpn2"] = duplicate["scores_hori_fpn2"]
        wrong_channels = heads()
        wrong_channels["link_scores_hori_fpn2"] = np.zeros((1, 4, 3, 6), np.float32)
        nonfinite = heads()
        nonfinite["bbox_deltas_hori_fpn2"][0, 0, 1, 1] = np.nan
        for values, message in (
            (missing, "incomplete six-head"),
            (duplicate, "duplicate score head"),
            (wrong_channels, "unexpected channel count"),
            (nonfinite, "non-finite value"),
        ):
            with self.subTest(message=message):
                self.assert_failure(self.run_sit(values), message)

    def test_invalid_decoder_flags_fail(self):
        for flag, value in (
            ("pixellink_image_size", "30"),
            ("pixellink_image_size", "30,60junk"),
            ("pixellink_image_size", "0,60"),
            ("pixellink_image_size", "30,-1"),
            ("pixellink_image_size", "99999999999999999999,60"),
            ("pixellink_text_threshold", "nan"),
            ("pixellink_link_threshold", "1.1"),
            ("pixellink_nms_iou", "-0.1"),
            ("pixellink_text_threshold", "scores_hori_fpn2:0.5"),
        ):
            with self.subTest(flag=flag, value=value):
                self.assert_failure(self.run_sit(heads(), [f"--{flag}", value]), f"--{flag}")

    def test_existing_detection_formats_ignore_pixellink_flags(self):
        boxes = np.array([[[0.5, 0.5, 0.5, 0.5]]], np.float32)
        logits = np.array([[[20.0, 0.0]]], np.float32)
        formats = (
            {"pred_boxes": boxes, "logits": logits},
            {"boxes": boxes, "classes": logits},
            {"detections": np.array([[[1, 2, 10, 20, 0.9, 0]]], np.float32)},
        )
        for values in formats:
            with self.subTest(format=list(values)):
                self.assert_pass(self.run_sit(values, ["--pixellink_image_size", "invalid"]))

    def test_changed_boxes_fail_map_gate(self):
        actual = heads()
        reference = heads()
        reference["bbox_deltas_hori_fpn2"][:, 0::2] += 10
        result = self.run_sit(actual, references=reference)
        self.assert_failure(result, "FAILED")
        self.assertIn("mAP@0.5000 = 0.0000%", result.stdout)


if __name__ == "__main__":
    unittest.main()
