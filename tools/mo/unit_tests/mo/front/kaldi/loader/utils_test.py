# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import struct
import unittest

from openvino.tools.mo.front.kaldi.loader.utils import end_of_nnet_tag, end_of_component_tag, get_bool, get_uint16, get_uint32, \
    get_uint64, read_binary_bool_token, read_binary_integer32_token, read_binary_integer64_token, read_string, \
    read_binary_float_token, find_next_tag, find_next_component, find_end_of_component, get_parameters, \
    collect_until_token_and_read, get_args_for_specifier
from openvino.tools.mo.utils.error import Error


class TestKaldiUtilsLoading(unittest.TestCase):
    bool_fmt = '?'
    uint16_fmt = 'H'
    uint32_fmt = 'I'
    uint64_fmt = 'q'
    float32_fmt = 'f'

    @staticmethod
    def bytesio_from(buffer):
        return io.BytesIO(buffer)

    @staticmethod
    def pack_value(value, fmt):
        return struct.pack(fmt, value)

    def test_check_common_tags(self):
        self.assertEqual(end_of_nnet_tag, '</Nnet>')
        self.assertEqual(end_of_component_tag, '<!EndOfComponent>')

    def test_check_results_getting_function(self):
        self.assertTrue(get_bool(self.pack_value(True, self.bool_fmt)))
        self.assertFalse(get_bool(self.pack_value(False, self.bool_fmt)))
        self.assertEqual(get_uint16(self.pack_value(16, self.uint16_fmt)), 16)
        self.assertEqual(get_uint32(self.pack_value(32, self.uint32_fmt)), 32)
        self.assertEqual(get_uint64(self.pack_value(64, self.uint64_fmt)), 64)

    def test_read_binary_bool_token(self):
        true_value = self.bytesio_from(self.pack_value(True, self.bool_fmt))
        false_value = self.bytesio_from(self.pack_value(False, self.bool_fmt))
        self.assertTrue(read_binary_bool_token(true_value))
        self.assertFalse(read_binary_bool_token(false_value))

    def test_read_binary_integer32_token(self):
        stream = self.bytesio_from(self.pack_value(4, 'B') + self.pack_value(32, self.uint32_fmt))
        self.assertEqual(read_binary_integer32_token(stream), 32)

    def test_read_binary_integer64_token(self):
        stream = self.bytesio_from(self.pack_value(8, 'B') + self.pack_value(64, self.uint64_fmt))
        self.assertEqual(read_binary_integer64_token(stream), 64)

    def test_read_binary_float_token(self):
        stream = self.bytesio_from(self.pack_value(4, 'B') + self.pack_value(0.001, self.float32_fmt))
        self.assertAlmostEqual(read_binary_float_token(stream), 0.001)

    def test_read_string_token(self):
        stream = self.bytesio_from(b"opgru3.renorm <NormalizeComponent> ")
        self.assertEqual(read_string(stream), b"opgru3.renorm")

    def test_find_next_tag(self):
        test_token = b'<TestToken>'
        self.assertEqual(find_next_tag(self.bytesio_from(test_token)), test_token.decode('ascii'))
        fake_token = b'<FakeBegin' + test_token
        self.assertEqual(find_next_tag(self.bytesio_from(fake_token)), test_token.decode('ascii'))

    def test_find_next_tag_raise_error(self):
        test_token = b'some bytes'
        self.assertRaises(Error, find_next_tag, self.bytesio_from(test_token))

    def test_find_next_component(self):
        component = b'<LstmProjectedStreams>'
        test_file = b'<Nnet>somefakeinfo<another>info' + component + b'<tag><!EndOfComponent></Nnet>'
        self.assertEqual(find_next_component(self.bytesio_from(test_file)), component.decode('ascii').lower()[1:-1])

    def test_find_next_component_eoc(self):
        component = b'<LstmProjectedStreams>'
        test_file = b'<!EndOfComponent>' + component + b'<tag><!EndOfComponent></Nnet>'
        self.assertEqual(find_next_component(self.bytesio_from(test_file)), component.decode('ascii').lower()[1:-1])

    def test_find_next_component_end_of_nnet(self):
        test_file = b'<Nnet>somefakeinfo<another>info<tag><!EndOfComponent></Nnet>'
        self.assertRaises(Error, find_next_component, self.bytesio_from(test_file))

    def test_find_end_of_component(self):
        component = '<AffineComponent>'
        test_file = b'somefakeinfo<another>info<tag>' + bytes(end_of_component_tag, 'ascii') + b'</Nnet>'
        end_tag, position = find_end_of_component(self.bytesio_from(test_file), component.lower()[1:-1])
        self.assertEqual(end_tag, end_of_component_tag)
        self.assertEqual(position, test_file.decode('ascii').index(end_of_component_tag) + len(end_of_component_tag))

    def test_get_pb(self):
        component = '<AffineComponent>'
        test_file = b'somefakeinfo<another>info<tag>' + bytes(end_of_component_tag, 'ascii') + b'</Nnet>'
        end_tag, end_position = find_end_of_component(self.bytesio_from(test_file), component[1:-1].lower())
        pb = get_parameters(self.bytesio_from(test_file), 0, end_position)

    def test_collect_until_token_and_read(self):
        tag = b'<InputDim>'
        test_file = b'<ComponentName> opgru3.renorm <NormalizeComponent> <InputDim> ' + self.pack_value(4, 'B') + \
                    self.pack_value(256, 'I') + b' <TargetRms> ' + self.pack_value(4, 'B') + \
                    self.pack_value(0.5, 'f') + b' <AddLogStddev> F</NormalizeComponent>'
        value = collect_until_token_and_read(self.bytesio_from(test_file), tag)
        self.assertEqual(value, 256)

    def test_get_args_for_specifier(self):
        string = b"(Offset(input, -2), Offset(input, -1), input, Offset(input, 1), Offset(input, 2))"
        args = get_args_for_specifier(string)
        ref = [b"Offset(input, -2)", b"Offset(input, -1)", b"input", b"Offset(input, 1)", b"Offset(input, 2)"]
        self.assertEqual(args, ref)

    def test_get_args_for_specifier_2(self):
        string = b"(Offset(input, -2), input, Offset(Offset(input, -1), 1))"
        args = get_args_for_specifier(string)
        ref = [b"Offset(input, -2)", b"input", b"Offset(Offset(input, -1), 1)"]
        self.assertEqual(args, ref)

    def test_get_args_for_specifier_3(self):
        string = b"(Offset(input, 1), Offset(input, 2))"
        args = get_args_for_specifier(string)
        ref = [b"Offset(input, 1)", b"Offset(input, 2)"]
        self.assertEqual(args, ref)
