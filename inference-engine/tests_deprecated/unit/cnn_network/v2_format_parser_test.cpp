// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cpp/ie_cnn_network.h>
#include <gtest/gtest.h>
#include <string>
#include "mean_image.h"

#define MT_BATCH 1
#define MT_CHANNELS 3
#define MT_HEIGHT 1
#define MT_WIDTH 2
#define LAYER_COUNT 3

#include "parser_tests_base.hpp"
#include "common_test_utils/xml_net_builder/xml_father.hpp"

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class V2FormatParserTest : public FormatParserTest {
};


TEST_F (V2FormatParserTest, invalidXml_ShouldThrow) {
    string content = CommonTestUtils::XMLFather()
            .node("net")
            .attr("name", "AlexNet").attr("version", 2);

    ASSERT_THROW(parse(content), InferenceEngine::details::InferenceEngineException);
}

TEST_F (V2FormatParserTest, canParseDims) {
    // <input name="data"><dim>10</dim><dim>3</dim><dim>227</dim><dim>227</dim></input>
    string content = xml().node("net").attr("name", "AlexNet").attr("version", 2)
            .node("layers")
            .initInputlayer("data", 0, 0)
            .initPowerlayerInOutV2("power1", 1, 1, 2)
            .initlayerInV2("power2", 2, 3)
            .newnode("edges")
            .initedge(0, 0, 1, 1)
            .initedge(1, 2, 2, 3);

     ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
}

TEST_F(V2FormatParserTest, failWhenNoDims) {
    string content = xml().node("net").attr("name", "AlexNet").attr("version", 2)
            .node("layer").attr("type", "Input").attr("name", "data").attr("id", 0);

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, failOnZeroDim) {
    string content = xml().node("net").attr("name", "AlexNet").attr("version", 2)
            .node("layer").attr("type", "Input").attr("name", "data").attr("id", 0)
                .node("output")
                    .node("port").attr("id", 0)
                      .node("dim", 0)
                    .close()
                .close()
            .close()
            .close();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}
TEST_F(V2FormatParserTest, canParseMeanImageValues) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2()
            .node("channel").attr("id", "0").node("mean").attr("value", "104.5").close()
            .newnode("channel").attr("id", "1").node("mean").attr("value", "117.8").close()
            .newnode("channel").attr("id", "2").node("mean").attr("value", "123");

     ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
	auto & pp = getFirstInput()->getPreProcess();

    ASSERT_EQ(pp.getMeanVariant(), MEAN_VALUE);
    ASSERT_EQ(MT_CHANNELS, pp.getNumberOfChannels());
    InferenceEngine::PreProcessChannel::Ptr preProcessChannel;
    ASSERT_FLOAT_EQ(104.5f, pp[0]->meanValue);
    ASSERT_FLOAT_EQ(117.8f, pp[1]->meanValue);
    ASSERT_FLOAT_EQ(123.f, pp[2]->meanValue);

    ASSERT_EQ(nullptr, pp[0]->meanData);
    ASSERT_EQ(nullptr, pp[1]->meanData);
    ASSERT_EQ(nullptr, pp[2]->meanData);
}

TEST_F(V2FormatParserTest, canParseScaleValuesOnly) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2()
            .node("channel").attr("id", "0").node("scale").attr("value", "104.5").close()
            .newnode("channel").attr("id", "1").node("scale").attr("value", "117.8").close()
            .newnode("channel").attr("id", "2").node("scale").attr("value", "123");
     ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
	auto & pp = getFirstInput()->getPreProcess();
    ASSERT_EQ(pp.getMeanVariant(), NONE);
}

TEST_F(V2FormatParserTest, failIfOneOfMeanImageIdsMissed) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2()
            .node("channel").attr("id", "1").node("mean").attr("value", "117.8").close()
            .newnode("channel").attr("id", "2").node("mean").attr("value", "123");

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, failIfValueAttributeIsNotValid) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2()
            .node("channel").attr("id", "0").node("mean").attr("value1", "104.5").close()
            .newnode("channel").attr("id", "1").node("mean").attr("value", "117.8").close()
            .newnode("channel").attr("id", "2").node("mean").attr("value", "123");

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, failIfMeanValueIsNotSpecified) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2()
            .node("channel").attr("id", "0").node("mean").attr("value", "").close();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, failIfMeanValueNotSpecifiedInPreProcessing) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2()
            .node("channel").attr("id", "0").node("mean").attr("value", "104.5").close()
            .newnode("channel").attr("id", "1").node("mean").attr("value", "117.8").close()
            .newnode("channel").attr("id", "2").node("mean1").attr("value", "123");

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, failIfIdLessThanZero) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2()
            .node("channel").attr("id", "-1").node("mean").attr("value", "104.5").close();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, failIfIdNotInteger) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2()
            .node("channel").attr("id", "0").node("mean").attr("value", "104.5").close()
            .newnode("channel").attr("id", "1").node("mean").attr("value", "117.8").close()
            .newnode("channel").attr("id", "2_2").node("mean").attr("value", "123").close();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, failIfValueNotFloat) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2()
            .node("channel").attr("id", "0").node("mean").attr("value", "104,5").close()
            .newnode("channel").attr("id", "1").node("mean").attr("value", "117.8").close()
            .newnode("channel").attr("id", "2").node("mean").attr("value", "123").close();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, failIfIdMoreThanNumChannels) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2()
            .node("channel").attr("id", "4").node("mean").attr("value", "104.5").close();
    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, failIfIdIsDuplicated) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2()
            .node("channel").attr("id", "0").node("mean").attr("value", "104.5").close()
            .newnode("channel").attr("id", "0").node("mean").attr("value", "117.8").close()
            .newnode("channel").attr("id", "2").node("mean").attr("value", "123").close();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, failParseMeanImageWithoutSpecifyingPrecision) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2()
            .node("channel").attr("id", "0").node("mean").attr("offset", "0").attr("size", "5").close();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, failIfOneOfMeanImageIfMeanNotSpecified) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2()
            .node("channel").attr("id", "0").close();

     ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
	auto & pp = getFirstInput()->getPreProcess();
    ASSERT_EQ(pp.getMeanVariant(), NONE);
}

TEST_F(V2FormatParserTest, failIfOffsetValueMissing) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2().attr("mean-precision", "FP32")
            .node("channel").attr("id", "0").node("mean").attr("offset", "").attr("size", "5").close();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, failIfSizeValueMissing) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2().attr("mean-precision", "FP32")
            .node("channel").attr("id", "0").node("mean").attr("offset", "1").attr("size", "").close();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, throwsIfSizeOfMeanElementsMismatchWithExpected) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2().attr("mean-precision", "FP32")
            .node("channel").attr("id", "0").node("mean")
            .attr("offset", "0").attr("size", "2").close()
            .newnode("channel").attr("id", "1").node("mean")
            .attr("offset", "2").attr("size", "2").close()
            .newnode("channel").attr("id", "2").node("mean")
            .attr("offset", "4").attr("size", "2").close();

	ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, canHandleQ78MeanValues) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2().attr("mean-precision", "Q78")
            .node("channel").attr("id", "0").node("mean")
            .attr("offset", "0").attr("size", "4").close()
            .newnode("channel").attr("id", "1").node("mean")
            .attr("offset", "4").attr("size", "4").close()
            .newnode("channel").attr("id", "2").node("mean")
            .attr("offset", "8").attr("size", "4").close();

     ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
	auto & pp = getFirstInput()->getPreProcess();
    ASSERT_EQ(pp.getMeanVariant(), MEAN_IMAGE);
    auto binBlob = makeBinBlobForMeanTest<short>();
    assertSetWeightsSucceed(binBlob);
}

TEST_F(V2FormatParserTest, canParseBinFileWithMeanImageUINT8Values) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2().attr("mean-precision", "U8")
            .node("channel").attr("id", "0").node("mean")
            .attr("offset", "0").attr("size", "2").close()
            .newnode("channel").attr("id", "1").node("mean")
            .attr("offset", "2").attr("size", "2").close()
            .newnode("channel").attr("id", "2").node("mean")
            .attr("offset", "4").attr("size", "2").close();

     ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
	auto & pp = getFirstInput()->getPreProcess();
    ASSERT_EQ(pp.getMeanVariant(), MEAN_IMAGE);
    auto binBlob = makeBinBlobForMeanTest<uint8_t>();
    assertSetWeightsSucceed(binBlob);
    assertMeanImagePerChannelCorrect<uint8_t>();
    assertMeanImageCorrect<uint8_t>();
}

TEST_F(V2FormatParserTest, canParseBinFileWithMeanImageI16Values) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2().attr("mean-precision", "I16")
            .node("channel").attr("id", "0").node("mean")
            .attr("offset", "0").attr("size", "4").close()
            .newnode("channel").attr("id", "1").node("mean")
            .attr("offset", "4").attr("size", "4").close()
            .newnode("channel").attr("id", "2").node("mean")
            .attr("offset", "8").attr("size", "4").close();

     ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
	auto & pp = getFirstInput()->getPreProcess();
    ASSERT_EQ(pp.getMeanVariant(), MEAN_IMAGE);
    auto binBlob = makeBinBlobForMeanTest<short>();
    assertSetWeightsSucceed(binBlob);
    assertMeanImagePerChannelCorrect<short>();
    assertMeanImageCorrect<short>();
}

TEST_F(V2FormatParserTest, canParseBinFileWithMeanImageFloatValues) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2().attr("mean-precision", "FP32")
            .node("channel").attr("id", "0").node("mean")
            .attr("offset", "0").attr("size", "8").close()
            .newnode("channel").attr("id", "1").node("mean")
            .attr("offset", "8").attr("size", "8").close()
            .newnode("channel").attr("id", "2").node("mean")
            .attr("offset", "16").attr("size", "8").close();

     ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
	auto & pp = getFirstInput()->getPreProcess();
    ASSERT_EQ(pp.getMeanVariant(), MEAN_IMAGE);
    auto binBlob = makeBinBlobForMeanTest<float>();
    assertSetWeightsSucceed(binBlob);
    assertMeanImagePerChannelCorrect<float>();
    assertMeanImageCorrect<float>();
}

TEST_F(V2FormatParserTest, throwIfSizeDoesNotMatchExpectedMeanSize) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2().attr("mean-precision", "FP32")
            .node("channel").attr("id", "0").node("mean")
            .attr("offset", "0").attr("size", "9").close()
            .newnode("channel").attr("id", "1").node("mean")
            .attr("offset", "8").attr("size", "8").close()
            .newnode("channel").attr("id", "2").node("mean")
            .attr("offset", "16").attr("size", "8").close();

	ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, failIfSizeExceedBinaryFileSize) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2().attr("mean-precision", "FP32")
            .node("channel").attr("id", "0").node("mean")
            .attr("offset", "0").attr("size", "8").close()
            .newnode("channel").attr("id", "1").node("mean")
            .attr("offset", "8").attr("size", "8").close()
            .newnode("channel").attr("id", "2").node("mean")
            .attr("offset", "1600").attr("size", "8").close();

     ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
	auto & pp = getFirstInput()->getPreProcess();
    ASSERT_EQ(pp.getMeanVariant(), MEAN_IMAGE);
    auto binBlob = makeBinBlobForMeanTest<float>();
    assertSetWeightsFail(binBlob);
}

TEST_F(V2FormatParserTest, failIfMixedAttributesAreSet) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2().attr("mean-precision", "FP32")
            .node("channel").attr("id", "0").node("mean")
            .attr("value", "0").close()
            .newnode("channel").attr("id", "1").node("mean")
            .attr("offset", "8").attr("size", "8").close()
            .newnode("channel").attr("id", "2").node("mean")
            .attr("offset", "16").attr("size", "8").close();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, parseSucceedIfMixedButAllValuesSet) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2().attr("mean-precision", "FP32")
            .node("channel").attr("id", "0").node("mean")
            .attr("value", "0").close()
            .newnode("channel").attr("id", "1").node("mean")
            .attr("value", "0").attr("offset", "8").attr("size", "8").close()
            .newnode("channel").attr("id", "2").node("mean")
            .attr("value", "0").attr("offset", "16").attr("size", "8").close();

     ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
	auto & pp = getFirstInput()->getPreProcess();
    ASSERT_EQ(pp.getMeanVariant(), MEAN_VALUE);
}


TEST_F(V2FormatParserTest, parseTileLayer) {
    string content = BEGIN_NET()
        .initlayerInOut("tile", "Tile", 1, 1, 2)
            .node("data").attr("axis",3).attr("tiles", 88).close()
        .close()
    END_NET();


     ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
    auto lp = getLayer<TileLayer>("tile");
    ASSERT_TRUE(lp);

    ASSERT_EQ(lp->axis, 3);
    ASSERT_EQ(lp->tiles, 88);
}


TEST_F(V2FormatParserTest, checkPreProcessWithRefName) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2()
            .attr("mean-precision", "FP32")
            .attr("reference-layer-name", "data")
        .node("channel").attr("id", "0").node("mean")
        .attr("offset", "0").attr("size", "8").close()
        .newnode("channel").attr("id", "1").node("mean")
        .attr("offset", "8").attr("size", "8").close()
        .newnode("channel").attr("id", "2").node("mean")
        .attr("offset", "16").attr("size", "8").close();

     ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
    auto & pp = getFirstInput()->getPreProcess();
    ASSERT_EQ(pp.getMeanVariant(), MEAN_IMAGE);
    auto binBlob = makeBinBlobForMeanTest<float>();
    assertSetWeightsSucceed(binBlob);
    assertMeanImagePerChannelCorrect<float>();
    assertMeanImageCorrect<float>();
}

TEST_F(V2FormatParserTest, failWhenPreProcessNameMissing) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2()
        .attr("mean-precision", "FP32")
        .attr("reference-layer-name", "foo")
        .node("channel").attr("id", "0").node("mean")
        .attr("offset", "0").attr("size", "8").close()
        .newnode("channel").attr("id", "1").node("mean")
        .attr("offset", "8").attr("size", "8").close()
        .newnode("channel").attr("id", "2").node("mean")
        .attr("offset", "16").attr("size", "8").close();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, cannotParseUnknownEltwiseOperation) {
    string content = BEGIN_NET()
        .initlayerInOut("e", "Eltwise", 1, 1, 2)
        .node("data").attr("operation", "unknown").close()
        .close()
         END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, asserOnUnknownEltwiseOperation) {
    string content = BEGIN_NET()
        .initlayerInOut("e", "Eltwise", 1, 1, 2)
        .node("data").attr("operation", "unknown").close()
        .close()
        END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}


TEST_F(V2FormatParserTest, canParseEmptyElementwiseNodeAsSum) {
    string content = BEGIN_NET()
        .initlayerInOut("e", "Eltwise", 1, 1, 2)
        .node("data").attr("operation", "").close()
        .close()
        END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
    CNNLayerPtr ewise;
    ASSERT_EQ(OK, net->getLayerByName("e", ewise, nullptr));
    auto *eltwise = dynamic_cast<EltwiseLayer *>(ewise.get());
    ASSERT_NE(nullptr, eltwise);
    ASSERT_EQ(eltwise->_operation, EltwiseLayer::Sum);
}

TEST_F(V2FormatParserTest, canParseEmptyElementwiseNodeAsSumAmazonIR) {
    string content = BEGIN_NET()
        .initlayerInOut("e", "Eltwise", 1, 1, 2)
        .node("elementwise_data").attr("operation", "").close()
        .close()
            END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
    CNNLayerPtr ewise;
    ASSERT_EQ(OK, net->getLayerByName("e", ewise, nullptr));
    auto *eltwise = dynamic_cast<EltwiseLayer *>(ewise.get());
    ASSERT_NE(nullptr, eltwise);
    ASSERT_EQ(eltwise->_operation, EltwiseLayer::Sum);
}

TEST_F(V2FormatParserTest, canParseMissedElementwiseOperationNodeAsSum) {
    string content = BEGIN_NET()
        .initlayerInOut("e", "Eltwise", 1, 1, 2)
        .node("data").close()
        .close()
            END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
    CNNLayerPtr ewise;
    ASSERT_EQ(OK, net->getLayerByName("e", ewise, nullptr));
    auto *eltwise = dynamic_cast<EltwiseLayer *>(ewise.get());
    ASSERT_NE(nullptr, eltwise);
    ASSERT_EQ(eltwise->_operation, EltwiseLayer::Sum);
}

TEST_F(V2FormatParserTest, canParseMissedElementwiseDataNodeAsSum) {
    string content = BEGIN_NET()
        .initlayerInOut("e", "Eltwise", 1, 1, 2)
        .close()
    END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
    CNNLayerPtr ewise;
    ASSERT_EQ(OK, net->getLayerByName("e", ewise, nullptr));
    auto *eltwise = dynamic_cast<EltwiseLayer *>(ewise.get());
    ASSERT_NE(nullptr, eltwise);
    ASSERT_EQ(eltwise->_operation, EltwiseLayer::Sum);
}

TEST_F(V2FormatParserTest, canParseProdInElementwiseNode) {
    string content = BEGIN_NET()
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

TEST_F(V2FormatParserTest, canParseMulInElementwiseNode) {
    string content = BEGIN_NET()
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

TEST_F(V2FormatParserTest, canParseSumInElementwiseNode) {
    string content = BEGIN_NET()
        .initlayerInOut("e", "Eltwise", 1, 1, 2)
        .node("data").attr("operation", "sum").close()
        .close()
    END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
    CNNLayerPtr ewise;
    ASSERT_EQ(OK, net->getLayerByName("e", ewise, nullptr));
    auto *eltwise = dynamic_cast<EltwiseLayer *>(ewise.get());
    ASSERT_NE(nullptr, eltwise);
    ASSERT_EQ(eltwise->_operation, EltwiseLayer::Sum);
}

TEST_F(V2FormatParserTest, parsesNumberOfLayersCorrectly) {
    string content = MAKE_ALEXNET_FOR_MEAN_TESTS_V2();

    ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));
    CNNNetwork network(net);
    ASSERT_EQ(network.layerCount(), LAYER_COUNT);
}

TEST_F(V2FormatParserTest, canThrowExceptionIfNoType) {

    string content = BEGIN_NET()
        .initlayerInOut("a", "Activation", 1, 1, 2)
        .node("data1").attr("type", "tanH").close()
        .close()
            END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, canThrowExceptionIfMultipleTypes) {

    string content = BEGIN_NET()
        .initlayerInOut("a", "Activation", 1, 1, 2)
        .node("data").attr("type", "tanH").attr("type", "sigmoid").close()
        .close()
            END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseFail(content));
}

TEST_F(V2FormatParserTest, canConvertActivationLayerAsTanH) {

    string content = BEGIN_NET()
        .initlayerInOut("a", "Activation", 1, 1, 2)
        .node("data").attr("type", "tanH").close()
        .close()
            END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));

    CNNLayerPtr tanh;
    ASSERT_EQ(OK, net->getLayerByName("a", tanh, nullptr));
    ASSERT_STREQ(tanh->type.c_str(), "tanh");
    ASSERT_EQ(tanh->params.find("type"), tanh->params.end());
}

TEST_F(V2FormatParserTest, canConvertActivationLayerAsELU) {

    string content = BEGIN_NET()
        .initlayerInOut("a", "Activation", 1, 1, 2)
        .node("data").attr("type", "elu").attr("alpha", "0.1").close()
        .close()
            END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));

    CNNLayerPtr elu;
    ASSERT_EQ(OK, net->getLayerByName("a", elu, nullptr));
    ASSERT_STREQ(elu->type.c_str(), "elu");
    ASSERT_FLOAT_EQ(elu->GetParamAsFloat("alpha"), 0.1);
    ASSERT_EQ(elu->params.find("type"), elu->params.end());
}

TEST_F(V2FormatParserTest, canConvertActivationLayerAsRelu) {

    string content = BEGIN_NET()
        .initlayerInOut("a", "Activation", 1, 1, 2)
        .node("data").attr("type", "relu").attr("negative_slope", "0.1").close()
        .close()
            END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));

    CNNLayerPtr relu;
    ASSERT_EQ(OK, net->getLayerByName("a", relu, nullptr));
    ASSERT_STREQ(relu->type.c_str(), "relu");

    auto *reluLayer = dynamic_cast<ReLULayer *>(relu.get());
    ASSERT_NE(nullptr, reluLayer);

    ASSERT_FLOAT_EQ(reluLayer->negative_slope, 0.1);
    ASSERT_EQ(reluLayer->params.find("type"), reluLayer->params.end());
}

TEST_F(V2FormatParserTest, canConvertActivationLayerAsPRelu) {

    string content = BEGIN_NET()
        .initlayerInOut("a", "Activation", 1, 1, 2)
        .node("data").attr("type", "pRelu").attr("channel_shared", "1").close()
        .close()
            END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));

    CNNLayerPtr layer;
    ASSERT_EQ(OK, net->getLayerByName("a", layer, nullptr));
    ASSERT_STREQ(layer->type.c_str(), "prelu");

    auto *preluLayer = dynamic_cast<PReLULayer *>(layer.get());
    ASSERT_NE(nullptr, preluLayer);

    ASSERT_EQ(preluLayer->_channel_shared, 1);
    ASSERT_EQ(preluLayer->params.find("type"), preluLayer->params.end());
}

TEST_F(V2FormatParserTest, canConvertActivationLayerAsSigmoid) {

    string content = BEGIN_NET()
        .initlayerInOut("a", "Activation", 1, 1, 2)
        .node("data").attr("type", "sigmoid").close()
        .close()
            END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));

    CNNLayerPtr sigmoid;
    ASSERT_EQ(OK, net->getLayerByName("a", sigmoid, nullptr));
    ASSERT_STREQ(sigmoid->type.c_str(), "sigmoid");
    ASSERT_EQ(sigmoid->params.find("type"), sigmoid->params.end());
}

TEST_F(V2FormatParserTest, canConvertActivationLayerAsClamp) {

    string content = BEGIN_NET()
        .initlayerInOut("a", "Activation", 1, 1, 2)
        .node("data").attr("type", "clamp").attr("max","5").attr("min","-5").close()
        .close()
            END_NET();

    ASSERT_NO_FATAL_FAILURE(assertParseSucceed(content));

    CNNLayerPtr layer;
    ASSERT_EQ(OK, net->getLayerByName("a", layer, nullptr));
    ASSERT_STREQ(layer->type.c_str(), "clamp");
    auto  clamp = dynamic_cast<ClampLayer*>(layer.get());
    ASSERT_NE(clamp, nullptr);

    ASSERT_EQ(clamp->min_value, -5);
    ASSERT_EQ(clamp->max_value, 5);
    ASSERT_EQ(clamp->params.find("type"), clamp->params.end());
}
