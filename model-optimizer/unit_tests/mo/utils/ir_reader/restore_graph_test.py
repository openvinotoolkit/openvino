# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
import tempfile

from mo.utils.ir_reader.restore_graph import restore_graph_from_ir
from defusedxml.common import EntitiesForbidden


class TestIRReader(unittest.TestCase):
    def setUp(self):
        self.xml_bomb = b'<?xml version="1.0"?>\n' \
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

    def test_read_xml_bomb(self):
        bomb_file = tempfile.NamedTemporaryFile(delete=False)
        bomb_file.write(self.xml_bomb)
        bomb_file.close()
        self.assertRaises(EntitiesForbidden, restore_graph_from_ir, bomb_file.name)
        os.remove(bomb_file.name)
