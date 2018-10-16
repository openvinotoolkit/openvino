// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#define MT_BATCH 1
#define MT_CHANNELS 3
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

TEST_F(V3FormatParserTest, canNotParseEmptyElementwiseNode) {
    string content = BEGIN_NET_V3()
        .initlayerInOut("e", "Eltwise", 1, 1, 2)
        .node("data").attr("operation", "").close()
        .close()
            END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V3FormatParserTest, canNotParseMissedElementwiseNodeType) {
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

TEST_F(V3FormatParserTest, DISABLED_canParseProdInElementwiseNode) {
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


TEST_F(V3FormatParserTest, DISABLED_canParseMulInElementwiseNode) {
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