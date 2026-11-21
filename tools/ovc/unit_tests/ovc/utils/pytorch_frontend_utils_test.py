# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace

from openvino.tools.ovc.cli_parser import _InputCutInfo, input_to_input_cut_info
from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import extract_input_info_from_example
from unit_tests.ovc.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry


class FakeTensor:
    # Stands in for a torch.Tensor: only .shape, .ndim and .dtype are read by
    # extract_input_info_from_example, and "int" is a key pt_to_ov_type_map already
    # maps without needing torch installed.
    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = "int"


class FakeInputModel:
    def __init__(self, example_input):
        self._example_input = example_input
        self._input_signature = None


class TestExtractInputInfoFromExample(UnitTestWithMockedTelemetry):
    def test_dict_example_input_matched_by_name_not_position(self):
        # example_input's dict order ("attention_mask" first) does not match the
        # order the user listed in 'input' ("input_ids" first). Matching by
        # position would swap the two shapes.
        args = Namespace()
        args.input_model = FakeInputModel({
            "attention_mask": FakeTensor((1, 4)),
            "input_ids": FakeTensor((1, 8)),
        })
        args.example_input = None
        args.dynamo = True
        args.input = input_to_input_cut_info(["input_ids", "attention_mask"])

        extract_input_info_from_example(args, args.input)

        shapes_by_name = {inp.name: inp.shape for inp in args.input}
        self.assertEqual(list(shapes_by_name["input_ids"]), [1, 8])
        self.assertEqual(list(shapes_by_name["attention_mask"]), [1, 4])

    def test_unnamed_example_input_still_matches_positionally(self):
        # No names anywhere, and the descriptors carry no shape either, so this
        # actually exercises copying example_input's shape by position, not just
        # echoing a shape the descriptor already had.
        args = Namespace()
        args.input_model = FakeInputModel((FakeTensor((1, 4)), FakeTensor((1, 8))))
        args.example_input = None
        args.dynamo = True
        args.input = [_InputCutInfo(None, None, None, None), _InputCutInfo(None, None, None, None)]

        extract_input_info_from_example(args, args.input)

        self.assertEqual(list(args.input[0].shape), [1, 4])
        self.assertEqual(list(args.input[1].shape), [1, 8])

    def test_unmatched_name_leaves_shape_unset_instead_of_guessing_by_position(self):
        # Both sides are named, but the name sets don't line up (the user mistyped
        # "attn_mask"). Falling back to position here would silently attach
        # "input_ids"'s shape to "attn_mask", so it must stay unset instead.
        args = Namespace()
        args.input_model = FakeInputModel({
            "attention_mask": FakeTensor((1, 4)),
            "input_ids": FakeTensor((1, 8)),
        })
        args.example_input = None
        args.dynamo = True
        args.input = input_to_input_cut_info(["input_ids", "attn_mask"])

        extract_input_info_from_example(args, args.input)

        shapes_by_name = {inp.name: inp.shape for inp in args.input}
        self.assertEqual(list(shapes_by_name["input_ids"]), [1, 8])
        self.assertIsNone(shapes_by_name["attn_mask"])

    def test_named_input_matched_by_name_even_off_its_own_position(self):
        # example_input mixes a plain tensor ("input_ids") with a nested tuple
        # ("past_key_values"): flatten_inputs() drops the name on the nested one,
        # so input_arg ends up [unnamed leaf, ("input_ids", shape)] in that order.
        # The user lists input_ids first, landing it on input_arg's unnamed slot.
        # Matching must still find input_ids by name instead of grabbing whatever
        # sits at that position.
        args = Namespace()
        args.input_model = FakeInputModel({
            "past_key_values": (FakeTensor((1, 2)),),
            "input_ids": FakeTensor((1, 8)),
        })
        args.example_input = None
        args.dynamo = True
        args.input = input_to_input_cut_info(["input_ids", "unused_name"])

        extract_input_info_from_example(args, args.input)

        shapes_by_name = {inp.name: inp.shape for inp in args.input}
        self.assertEqual(list(shapes_by_name["input_ids"]), [1, 8])

    def test_extract_input_info_from_example_matches_named_inputs_by_name(self):
        args = Namespace()
        args.input_model = FakeInputModel({
            "a": FakeTensor((1, 3, 224, 224)),
            "b": FakeTensor((1,)),
        })
        args.example_input = None
        args.dynamo = True
        args.input = input_to_input_cut_info(["b", "a"])

        extract_input_info_from_example(args, args.input)

        shapes_by_name = {inp.name: inp.shape for inp in args.input}
        self.assertEqual(list(shapes_by_name["a"]), [1, 3, 224, 224])
        self.assertEqual(list(shapes_by_name["b"]), [1])
