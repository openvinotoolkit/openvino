// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#define MT_BATCH 1
#define MT_CHANNELS 3
#define MT_DEPTH 2
#define MT_HEIGHT 1
#define MT_WIDTH 2
#define LAYER_COUNT 1
#include "parser_tests_base.hpp"


#include "parser_tests_base.hpp"

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace testing;

class V3FormatParserTest : public FormatParserTest {
};

TEST_F(V3FormatParserTest, DISABLED_canNotParseEmptyElementwiseNode) {
    string content = BEGIN_NET_V3()
        .initlayerInOut("e", "Eltwise", 1, 1, 2)
        .node("data").attr("operation", "").close()
        .close()
            END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V3FormatParserTest, DISABLED_canNotParseMissedElementwiseNodeType) {
    string content = BEGIN_NET_V3()
        .initlayerInOut("e", "Eltwise", 1, 1, 2)
        .close()
            END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V3FormatParserTest, cannotParseUnknownEltwiseOperation) {
    string content = BEGIN_NET_V3()
        .initlayerInOut("e", "Eltwise", 1, 1, 2)
        .node("data").attr("operation", "unknown").close()
        .close()
            END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V3FormatParserTest, canParseProdInElementwiseNode) {
    string content = BEGIN_NET_V3()
        .initlayerInOut("e", "Eltwise", 1, 1, 2)
        .node("data").attr("operation", "prod").close()
        .close()
            END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
    CNNLayerPtr ewise;
    ASSERT_EQ(OK, net->getLayerByName("e", ewise, nullptr));
    auto *eltwise = dynamic_cast<EltwiseLayer *>(ewise.get());
    ASSERT_NE(nullptr, eltwise);
    ASSERT_EQ(eltwise->_operation, EltwiseLayer::Prod);
}

TEST_F(V3FormatParserTest, canParseMulInElementwiseNode) {
    string content = BEGIN_NET_V3()
        .initlayerInOut("e", "Eltwise", 1, 1, 2)
        .node("data").attr("operation", "mul").close()
        .close()
            END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
    CNNLayerPtr ewise;
    ASSERT_EQ(OK, net->getLayerByName("e", ewise, nullptr));
    auto *eltwise = dynamic_cast<EltwiseLayer *>(ewise.get());
    ASSERT_NE(nullptr, eltwise);
    ASSERT_EQ(eltwise->_operation, EltwiseLayer::Prod);
}

TEST_F(V3FormatParserTest, canParse5Dinput) {
    string content = xml().node("net").attr("name", "Only_input_5D").attr("version", 3)
            .initInputlayer5D("data", 0, 0);

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V3FormatParserTest, DISABLE_conv3DInvalidKernel) {
    string content = xml().node("net").attr("name", "5d_net").attr("version", 3)
            .initConv5DlayerInOut("3D_conv", 0, 1, 64, "", "0,0,0", "0,0,0", "1,1,1", "1,1,1", 0, 0)
            .close();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}
