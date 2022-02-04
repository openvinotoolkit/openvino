# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest
from defusedxml.common import EntitiesForbidden
from openvino.tools.mo.utils.ir_reader.restore_graph import restore_graph_from_ir


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
