// Copyright (C) 2018-2019 Intel Corporation
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

class V2ParserPublicSegments: public InferenceEngine::details::FormatParser {
public:
    const std::map<std::string, LayerParseParameters>& getLayerParseParameters() {
        return layersParseInfo;
    }
};

TEST_F(V3FormatParserTest, LargeWeights) {
    std::string model = R"V0G0N(
<net name="PVANET" version="3" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>1024</dim>
                </port>
            </output>
        </layer>
		<layer id="1" name="MatMul" precision="FP32" type="FullyConnected">
			<data out-size="800000"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1024</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>800000</dim>
				</port>
			</output>
			<blobs>
				<weights offset="891492352" size="3276800000"/>
				<biases offset="4168292352" size="3200000"/>
			</blobs>
		</layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>)V0G0N";

    parse(model);

    auto params = ((V2ParserPublicSegments *)parser.get())->getLayerParseParameters();
    ASSERT_NE(params.end(), params.find("MatMul"));
    ASSERT_EQ(891492352, params["MatMul"].blobs["weights"].start);
    ASSERT_EQ(3276800000, params["MatMul"].blobs["weights"].size);
    ASSERT_EQ(4168292352, params["MatMul"].blobs["biases"].start);
    ASSERT_EQ(3200000, params["MatMul"].blobs["biases"].size);
}

TEST_F(V3FormatParserTest, IncorrectWeights) {
    std::string model = R"V0G0N(
<net name="PVANET" version="3" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>1024</dim>
                </port>
            </output>
        </layer>
		<layer id="1" name="MatMul" precision="FP32" type="FullyConnected">
			<data out-size="800000"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1024</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>800000</dim>
				</port>
			</output>
			<blobs>
				<weights offset="891492352" size="-64"/>
				<biases offset="4168292352" size="3200000"/>
			</blobs>
		</layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>)V0G0N";

    assertParseFail(model);
}