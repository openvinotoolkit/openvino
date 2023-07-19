# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest

import numpy as np
from defusedxml.common import EntitiesForbidden

import openvino.tools.mo.utils.ir_reader.extenders.convert_extender
from openvino.tools.mo.middle.passes.convert_data_type import destination_type_to_np_data_type
from openvino.tools.mo.middle.passes.infer import type_infer
from openvino.tools.mo.utils.graph import Node
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from openvino.tools.mo.utils.ir_reader.extender import Extender
from openvino.tools.mo.utils.ir_reader.restore_graph import restore_graph_from_ir, save_restored_graph


class TestIRReader(unittest.TestCase):
    def test_read_xml_incorrect(self):
        incorrect_xml = b'<?xml version="1.0"?>\n' \
                        b'<!DOCTYPE lolz [\n' \
                        b' <!ENTITY lol "lol">\n' \
                        b' <!ELEMENT lolz (#PCDATA)>\n' \
                        b' <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">\n' \
                        b' <!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">\n' \
                        b' <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">\n' \
                        b' <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">\n' \
                        b' <!ENTITY lol5 "&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;">\n' \
                        b' <!ENTITY lol6 "&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;">\n' \
                        b' <!ENTITY lol7 "&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;">\n' \
                        b' <!ENTITY lol8 "&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;">\n' \
                        b' <!ENTITY lol9 "&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;">\n' \
                        b']>\n' \
                        b'<lolz>&lol9;</lolz>'

        incorrect_xml_file = tempfile.NamedTemporaryFile(delete=False)
        incorrect_xml_file.write(incorrect_xml)
        incorrect_xml_file.close()
        self.assertRaises(EntitiesForbidden, restore_graph_from_ir, incorrect_xml_file.name)
        os.remove(incorrect_xml_file.name)

    def test_read_untrusted_IR(self):
        untrusted_xml = b'<?xml version="1.0"?>\n' \
                        b'<!DOCTYPE foo [\n' \
                        b'<!ELEMENT foo ANY>\n' \
                        b'<!ENTITY xxe SYSTEM "file:///c:/boot.ini">\n' \
                        b']>\n' \
                        b'<foo>&xxe;</foo>\n'

        untrusted_xml_file = tempfile.NamedTemporaryFile(delete=False)
        untrusted_xml_file.write(untrusted_xml)
        untrusted_xml_file.close()
        self.assertRaises(EntitiesForbidden, restore_graph_from_ir, untrusted_xml_file.name)
        os.remove(untrusted_xml_file.name)

    def test_read_malformed_IR(self):
        ir_front = b'<?xml version="1.0"?>' \
                   b'<net name="test" version="11">' \
                   b'	<layers>' \
                   b'		<layer id="0" name="parameter" type="Parameter" version="opset1">' \
                   b'			<data shape="1, 3, 22, 22" element_type="f32" />' \
                   b'			<output>' \
                   b'				<port id="0" precision="FP32" names="parameter">' \
                   b'					<dim>1</dim>' \
                   b'					<dim>3</dim>' \
                   b'					<dim>22</dim>' \
                   b'					<dim>22</dim>' \
                   b'				</port>' \
                   b'			</output>' \
                   b'		</layer>' \

        ir_front_malformed = b'<?xml version="1.0"?>' \
                             b'<net name="test" version="11">' \
                             b'	<layers>' \
                             b'		<layer id="0" name="parameter" type="Parameter" version="opset1">' \
                             b'			<data shape="1, 3, 22, 22" element_type="f32" />' \
                             b'			<output>' \
                             b'				<port id="boot.ini" precision="FP32" names="parameter">' \
                             b'					<dim>1</dim>' \
                             b'					<dim>3</dim>' \
                             b'					<dim>22</dim>' \
                             b'					<dim>22</dim>' \
                             b'				</port>' \
                             b'			</output>' \
                             b'		</layer>' \

        ir_end = b'		<layer id="1" name="Relu_4" type="ReLU" version="opset1">' \
                 b'			<input>' \
                 b'				<port id="0" precision="FP32">' \
                 b'					<dim>1</dim>' \
                 b'					<dim>3</dim>' \
                 b'					<dim>22</dim>' \
                 b'					<dim>22</dim>' \
                 b'				</port>' \
                 b'			</input>' \
                 b'			<output>' \
                 b'				<port id="1" precision="FP32">' \
                 b'					<dim>1</dim>' \
                 b'					<dim>3</dim>' \
                 b'					<dim>22</dim>' \
                 b'					<dim>22</dim>' \
                 b'				</port>' \
                 b'			</output>' \
                 b'		</layer>' \
                 b'		<layer id="2" name="result" type="Result" version="opset1">' \
                 b'			<input>' \
                 b'				<port id="0" precision="FP32">' \
                 b'					<dim>1</dim>' \
                 b'					<dim>3</dim>' \
                 b'					<dim>22</dim>' \
                 b'					<dim>22</dim>' \
                 b'				</port>' \
                 b'			</input>' \
                 b'		</layer>' \
                 b'	</layers>' \
                 b'	<edges>' \
                 b'		<edge from-layer="0" from-port="0" to-layer="1" to-port="0" />' \
                 b'		<edge from-layer="1" from-port="1" to-layer="2" to-port="0" />' \
                 b'	</edges>' \
                 b'</net>' \

        normal_ir_ir = ir_front + ir_end
        normal_ir_file = tempfile.NamedTemporaryFile(delete=False)
        normal_ir_file.write(normal_ir_ir)
        normal_ir_file.close()
        # we must expect no exceptions
        restore_graph_from_ir(normal_ir_file.name)
        os.remove(normal_ir_file.name)

        # expect that IR Reader complains on IR with malformed port id
        malformed_ir = ir_front_malformed + ir_end
        malformed_ir_file = tempfile.NamedTemporaryFile(delete=False)
        malformed_ir_file.write(malformed_ir)
        malformed_ir_file.close()
        self.assertRaises(ValueError, restore_graph_from_ir, malformed_ir_file.name)
        os.remove(malformed_ir_file.name)


class PatchedConvert_extender(Extender):
    """
    Original ConvertExtender contains setting 'stop_value_propagation', and because axis value goes to the Gather
    through Convert during shape_infer axis turns out to be None and shape_infer fails.
    For purposes of this unit-test we patch extender so that it will not add 'stop_value_propagation' attr.
    Outside the unit-test Convert_extender is left unchanged because inserting 'stop_value_propagation'
    is needed in other cases for CompressQuantizeWeights.
    See description of openvino/tools/mo/utils/ir_reader/extenders/convert_extender.py
    """
    op = 'Convert'

    @staticmethod
    def extend(op: Node):
        op['dst_type'] = destination_type_to_np_data_type(op.destination_type)


class TestIRSerializeAndRestore(unittest.TestCase):
    test_ir_xml = """<?xml version="1.0"?>
    <net name="test_ir" version="11">
    <layers>
        <layer id="0" name="input_1" type="Parameter" version="opset1">
            <data shape="1,128" element_type="f32" />
            <output>
                <port id="0" precision="FP32" names="input_1">
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="input_2" type="Parameter" version="opset1">
            <data shape="10" element_type="i32" />
            <output>
                <port id="0" precision="I32" names="input_2">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="gather_axis" type="Const" version="opset1">
            <data element_type="i32" shape="1" offset="0" size="4" />
            <output>
                <port id="0" precision="I32">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="501" name="gather" type="Gather" version="opset8">
        <data batch_dims="0" />
        <input>
            <port id="0" precision="FP32">
                <dim>1</dim>
                <dim>128</dim>
            </port>
            <port id="1" precision="I32">
                <dim>10</dim>
            </port>
            <port id="2" precision="I32">
                <dim>1</dim>
            </port>
        </input>
        <output>
            <port id="3" precision="FP32" names="gather">
                <dim>1</dim>
                <dim>10</dim>
            </port>
        </output>
        </layer>
        <layer id="590" name="result" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </input>
        </layer>
    </layers>
        
    <edges>
        <edge from-layer="0" from-port="0" to-layer="501" to-port="0" />
        <edge from-layer="1" from-port="0" to-layer="501" to-port="1" />
        <edge from-layer="3" from-port="0" to-layer="501" to-port="2" />
        <edge from-layer="501" from-port="3" to-layer="590" to-port="0" />
    </edges>
    </net>
    """

    def test_save_and_restore(self):
        original_xml_file = tempfile.NamedTemporaryFile(delete=False)
        original_xml_file.write(bytes(self.test_ir_xml, 'utf-8'))
        original_xml_file.close()

        axis_const_blob = np.array([1], dtype=np.int32)
        original_bin_file = tempfile.NamedTemporaryFile(mode='wb', delete=False)
        axis_const_blob.tofile(original_bin_file)
        original_bin_file.close()

        graph_orig, _ = restore_graph_from_ir(original_xml_file.name, original_bin_file.name)
        type_infer(graph_orig)
        os.remove(original_xml_file.name)
        os.remove(original_bin_file.name)

        restored_ir_dir = tempfile.TemporaryDirectory()

        save_restored_graph(graph_orig.copy(), restored_ir_dir.name, {})
        restored_xml_name = restored_ir_dir.name + '/test_ir.xml'
        restored_bin_name = restored_ir_dir.name + '/test_ir.bin'

        # Gather is listed in convert_inputs_of_specific_ops as 'Gather': {2: 'int64'}, but
        # no additional converts will be inserted, because input is int32
        graph_restored, _ = restore_graph_from_ir(restored_xml_name, restored_bin_name)
        os.remove(restored_xml_name)
        os.remove(restored_bin_name)
        os.remove(restored_xml_name.replace('xml', 'mapping'))
        os.removedirs(restored_ir_dir.name)

        flag, msg = compare_graphs(graph_orig, graph_restored, 'result', 'gather/sink_port_0')
        self.assertTrue(flag, msg)

    test_ir_xml_with_i8 = """<?xml version="1.0"?>
    <net name="test_ir" version="11">
    <layers>
        <layer id="0" name="input_1" type="Parameter" version="opset1">
            <data shape="1,128" element_type="f32" />
            <output>
                <port id="0" precision="FP32" names="input_1">
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="input_2" type="Parameter" version="opset1">
            <data shape="10" element_type="i32" />
            <output>
                <port id="0" precision="I32" names="input_2">
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="gather_axis" type="Const" version="opset1">
            <data element_type="i8" shape="1" offset="0" size="1" />
            <output>
                <port id="0" precision="I8">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="501" name="gather" type="Gather" version="opset8">
        <data batch_dims="0" />
        <input>
            <port id="0" precision="FP32">
                <dim>1</dim>
                <dim>128</dim>
            </port>
            <port id="1" precision="I32">
                <dim>10</dim>
            </port>
            <port id="2" precision="I32">
                <dim>1</dim>
            </port>
        </input>
        <output>
            <port id="3" precision="FP32" names="gather">
                <dim>1</dim>
                <dim>10</dim>
            </port>
        </output>
        </layer>
        <layer id="590" name="result" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </input>
        </layer>
    </layers>
        
    <edges>
        <edge from-layer="0" from-port="0" to-layer="501" to-port="0" />
        <edge from-layer="1" from-port="0" to-layer="501" to-port="1" />
        <edge from-layer="3" from-port="0" to-layer="501" to-port="2" />
        <edge from-layer="501" from-port="3" to-layer="590" to-port="0" />
    </edges>
    </net>
    """

    test_ir_xml_with_convert = """<?xml version="1.0"?>
    <net name="test_ir" version="11">
    <layers>
        <layer id="0" name="input_1" type="Parameter" version="opset1">
            <data shape="1,128" element_type="f32" />
            <output>
                <port id="0" precision="FP32" names="input_1">
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="input_2" type="Parameter" version="opset1">
            <data shape="10" element_type="i32" />
            <output>
                <port id="0" precision="I32" names="input_2">
                    <dim>10</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="gather_axis" type="Const" version="opset1">
            <data element_type="i8" shape="1" offset="0" size="1" />
            <output>
                <port id="0" precision="I8">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="583" name="convert" type="Convert" version="opset1">
            <data destination_type="i64" />
            <input>
                <port id="0" precision="I8">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="I64">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="501" name="gather" type="Gather" version="opset8">
        <data batch_dims="0" />
        <input>
            <port id="0" precision="FP32">
                <dim>1</dim>
                <dim>128</dim>
            </port>
            <port id="1" precision="I32">
                <dim>10</dim>
            </port>
            <port id="2" precision="I32">
                <dim>1</dim>
            </port>
        </input>
        <output>
            <port id="3" precision="FP32" names="gather">
                <dim>1</dim>
                <dim>10</dim>
            </port>
        </output>
        </layer>
        <layer id="590" name="result" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>10</dim>
                </port>
            </input>
        </layer>
    </layers>
    
    <edges>
        <edge from-layer="0" from-port="0" to-layer="501" to-port="0" />
        <edge from-layer="1" from-port="0" to-layer="501" to-port="1" />
        <edge from-layer="3" from-port="0" to-layer="583" to-port="0" />
        <edge from-layer="583" from-port="1" to-layer="501" to-port="2" />
        <edge from-layer="501" from-port="3" to-layer="590" to-port="0" />
    </edges>
    </net>
    """

    def test_save_and_restore_with_converts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = tmp_dir + os.sep
            original_xml_file = tempfile.NamedTemporaryFile(prefix=tmp_dir_path, delete=False)
            original_xml_file.write(bytes(self.test_ir_xml_with_i8, 'utf-8'))
            original_xml_file.close()

            gather_axis_blob = np.array([1], dtype=np.int8)
            original_bin_file = tempfile.NamedTemporaryFile(prefix=tmp_dir_path, mode='wb', delete=False)
            gather_axis_blob.tofile(original_bin_file)
            original_bin_file.close()

            graph_orig, _ = restore_graph_from_ir(original_xml_file.name, original_bin_file.name)
            type_infer(graph_orig)

            save_restored_graph(graph_orig.copy(), tmp_dir_path, {})

            ir_file_with_convert = tempfile.NamedTemporaryFile(prefix=tmp_dir_path, delete=False)
            ir_file_with_convert.write(bytes(self.test_ir_xml_with_convert, 'utf-8'))
            ir_file_with_convert.close()

            from openvino.tools.mo.utils.ir_reader.extender import Extender

            if 'Convert' in Extender.registered_ops:
                Extender.registered_ops['Convert'] = PatchedConvert_extender

            graph_with_convert, _ = restore_graph_from_ir(ir_file_with_convert.name, original_bin_file.name)
            type_infer(graph_with_convert)

            if 'Convert' in Extender.registered_ops:
                Extender.registered_ops['Convert'] = openvino.tools.mo.utils.ir_reader.extenders.convert_extender.Convert_extender

            restored_xml_file = tmp_dir_path + 'test_ir.xml'
            restored_bin_file = tmp_dir_path + 'test_ir.bin'

            # Gather is listed in convert_inputs_of_specific_ops as 'Gather': {2: 'int64'},
            # converts from int8 to int64 will be inserted
            graph_restored, _ = restore_graph_from_ir(restored_xml_file, restored_bin_file)

            flag, msg = compare_graphs(graph_orig, graph_restored, 'result', 'gather/sink_port_0')
            self.assertTrue(flag, msg)
