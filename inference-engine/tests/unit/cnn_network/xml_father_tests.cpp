// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "xml_father.hpp"

using namespace testing;
class XMLFatherF : public ::testing::Test {
public:
    XMLFather x = XMLFather::make_without_schema();
};

TEST_F(XMLFatherF, canCreateValidXmlNode) {
    ASSERT_STREQ("<net></net>\n", x.node("net").c_str());
}

TEST_F(XMLFatherF, canCreateValidNodeWithName) {
    ASSERT_STREQ("<net name=\"myname\"></net>\n", x.node("net").attr("name", "myname").c_str());
}


TEST_F(XMLFatherF, canCreateValidXmlNodeWithContent) {
    ASSERT_STREQ("<net><model>10</model></net>\n", x.node("net").node("model", 10).c_str());
}

TEST_F(XMLFatherF, canCreateValidXmlNodeWithAdvancedContent) {
    ASSERT_STREQ("<net><model>10 10 12</model></net>\n", x.node("net").node("model", 10, 10, 12).c_str());
}

TEST_F(XMLFatherF, canCreateLevel2Hierarchy) {
    ASSERT_STREQ("<net><net2><model>10</model></net2></net>\n", x.node("net").node("net2").node("model", 10).c_str());
}

TEST_F(XMLFatherF, canContinueAfterTrivialNode) {
    ASSERT_STREQ("<net><net2><model>10</model><model2>20</model2></net2></net>\n",
                 x.node("net").node("net2").node("model", 10).node("model2", 20).c_str());
}

TEST_F(XMLFatherF, canContinueAfterNodeWithSubnodes) {
    ASSERT_STREQ("<net><net2><model>10</model></net2><net4><model4>1</model4></net4></net>\n",
                 x.node("net")
                    .node("net2")
                        .node("model", 10)
                    .close()
                    .node("net4")
                        .node("model4", 1)
                 .c_str());
}
