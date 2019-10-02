// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

#include <shape_infer/mock_reshaper_launcher.hpp>
#include <shape_infer/mock_ishape_infer_impl.hpp>
#include <shape_infer/ie_reshape_io_controllers.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace ShapeInfer;
using namespace ::testing;

class OutputControllerTest : public ::testing::Test {
public:

    static const std::string TEST_NAME;
    DataPtr notEmptyData = std::make_shared<Data>(TEST_NAME, Precision::UNSPECIFIED, Layout::C);
    SizeVector inDims{1};
public:
    CNNLayerPtr createLayer(const std::string& name) {
        LayerParams params;
        params.name = name;
        return std::make_shared<CNNLayer>(params);
    }
};

const std::string OutputControllerTest::TEST_NAME = "TEST_NAME";

TEST_F(OutputControllerTest, failedToCreateWithEmptyOutData) {
    std::vector<DataPtr> inData;
    EXPECT_THROW(OutputController({}, TEST_NAME), InferenceEngineException);
}

TEST_F(OutputControllerTest, failedToCreateWithNullOutData) {
    EXPECT_THROW(OutputController({nullptr}, TEST_NAME), InferenceEngineException);
}

TEST_F(OutputControllerTest, canCreateOutputController) {
    ASSERT_NO_THROW(OutputController({notEmptyData}, TEST_NAME));
}

TEST_F(OutputControllerTest, canGetChanges) {
    OutputController controller({notEmptyData}, TEST_NAME);
    std::vector<SizeVector> shapes;
    ASSERT_NO_THROW(shapes = controller.getShapes(false));
    ASSERT_EQ(1, shapes.size());
}

TEST_F(OutputControllerTest, canSetShapes) {
    OutputController controller({notEmptyData}, TEST_NAME);
    auto shapes = {inDims, inDims};
    ASSERT_NO_THROW(controller.setShapes(shapes));
    ASSERT_EQ(shapes.size(), controller.getShapes(false).size());
}

TEST_F(OutputControllerTest, noThrowOnGetWithExcessShapes) {
    OutputController controller({notEmptyData}, TEST_NAME);
    ASSERT_NO_THROW(controller.setShapes({inDims, inDims}));
    ASSERT_FALSE(controller.getShapes(false).empty());
}

TEST_F(OutputControllerTest, throwOnPropagateWithNotEnoughShapes) {
    OutputController controller({notEmptyData, notEmptyData}, TEST_NAME);
    controller.setShapes({inDims});
    ASSERT_THROW(controller.propagateShapes({}), InferenceEngineException);
}

TEST_F(OutputControllerTest, throwOnPropagateWithExcessShapes) {
    OutputController controller({notEmptyData}, TEST_NAME);
    controller.setShapes({inDims, inDims});
    ASSERT_THROW(controller.propagateShapes({}), InferenceEngineException);
}

TEST_F(OutputControllerTest, throwOnPropagateWithEmptyLaunchers) {
    OutputController controller({notEmptyData}, TEST_NAME);
    notEmptyData->inputTo = {{{}, createLayer(TEST_NAME)}};
    controller.setShapes({inDims});
    ASSERT_THROW(controller.propagateShapes({}), InferenceEngineException);
}

TEST_F(OutputControllerTest, throwOnPropagateWithoutProperLauncher) {
    OutputController controller({notEmptyData}, TEST_NAME);
    notEmptyData->inputTo = {{{}, createLayer(TEST_NAME + "another")}};
    controller.setShapes({inDims});
    auto launcher = std::make_shared<MockReshapeLauncher>();
    EXPECT_CALL(*launcher.get(), getLayerName()).WillOnce(Return(TEST_NAME));
    ASSERT_THROW(controller.propagateShapes({launcher}), InferenceEngineException);
}

TEST_F(OutputControllerTest, canPropagateShapes) {
    OutputController controller({notEmptyData}, TEST_NAME);
    notEmptyData->inputTo = {{{}, createLayer(TEST_NAME)}};
    controller.setShapes({inDims});
    auto launcher = std::make_shared<MockReshapeLauncher>();
    EXPECT_CALL(*launcher.get(), setShapeByName(inDims, TEST_NAME));
    EXPECT_CALL(*launcher.get(), getLayerName()).WillOnce(Return(TEST_NAME));
    controller.propagateShapes({launcher});
}

TEST_F(OutputControllerTest, throwOnApplyWithNotEnoughShapes) {
    OutputController controller({notEmptyData, notEmptyData}, TEST_NAME);
    controller.setShapes({inDims});
    ASSERT_THROW(controller.applyChanges(), InferenceEngineException);
}

TEST_F(OutputControllerTest, throwOnApplyWithExcessShapes) {
    OutputController controller({notEmptyData}, TEST_NAME);
    auto shapes = {inDims, inDims};
    controller.setShapes(shapes);
    ASSERT_THROW(controller.applyChanges(), InferenceEngineException);
}

TEST_F(OutputControllerTest, canApplyChanges) {
    OutputController controller({notEmptyData}, TEST_NAME);
    controller.setShapes({inDims});
    ASSERT_NO_THROW(controller.applyChanges());
}

TEST_F(OutputControllerTest, canResetShapes) {
    OutputController controller({notEmptyData}, TEST_NAME);
    controller.setShapes({inDims});
    ASSERT_NO_THROW(controller.reset());
    ASSERT_TRUE(controller.getShapes(false).begin()->empty());
}
