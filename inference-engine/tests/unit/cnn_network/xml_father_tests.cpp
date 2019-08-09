// Copyright (C) 2018-2019 Intel Corporation
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
    std::string actual(x.node("net").c_str());
    actual.erase(std::remove(actual.begin(), actual.end(), '\n'), actual.end());
    ASSERT_EQ(std::string("<net></net>"), actual);
}

TEST_F(XMLFatherF, canCreateValidXmlNodeWithName) {
    std::string actual(x.node("net").attr("name", "myname").c_str());
    actual.erase(std::remove(actual.begin(), actual.end(), '\n'), actual.end());
    ASSERT_EQ(std::string("<net name=\"myname\"></net>"), actual);
}


TEST_F(XMLFatherF, canCreateValidXmlNodeWithContent) {
    std::string actual(x.node("net").node("model", 10).c_str());
    actual.erase(std::remove(actual.begin(), actual.end(), '\n'), actual.end());
    ASSERT_EQ(std::string("<net><model>10</model></net>"), actual);
}

TEST_F(XMLFatherF, canCreateValidXmlNodeWithAdvancedContent) {
    std::string actual(x.node("net").node("model", 10, 10, 12).c_str());
    actual.erase(std::remove(actual.begin(), actual.end(), '\n'), actual.end());
    ASSERT_EQ(std::string("<net><model>10 10 12</model></net>"), actual);
}

TEST_F(XMLFatherF, canCreateLevel2Hierarchy) {
    std::string actual(x.node("net").node("net2").node("model", 10).c_str());
    actual.erase(std::remove(actual.begin(), actual.end(), '\n'), actual.end());
    ASSERT_EQ(std::string("<net><net2><model>10</model></net2></net>"), actual);
}

TEST_F(XMLFatherF, canContinueAfterTrivialNode) {
    std::string actual(x.node("net").node("net2").node("model", 10).node("model2", 20).c_str());
    actual.erase(std::remove(actual.begin(), actual.end(), '\n'), actual.end());
    ASSERT_EQ(std::string("<net><net2><model>10</model><model2>20</model2></net2></net>"), actual);
}

TEST_F(XMLFatherF, canContinueAfterNodeWithSubnodes) {
    std::string actual(x.node("net")
                    .node("net2")
                    .node("model", 10)
                    .close()
                    .node("net4")
                    .node("model4", 1)
                    .c_str());
    actual.erase(std::remove(actual.begin(), actual.end(), '\n'), actual.end());
    ASSERT_EQ(std::string("<net><net2><model>10</model></net2><net4><model4>1</model4></net4></net>"), actual);
}
