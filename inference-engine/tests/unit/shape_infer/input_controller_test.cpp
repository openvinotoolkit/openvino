// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

#include <shape_infer/mock_ishape_infer_impl.hpp>
#include <shape_infer/ie_reshape_io_controllers.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace ShapeInfer;
using namespace ::testing;

class InputControllerTest : public ::testing::Test {
public:
    static const std::string TEST_NAME;
    DataPtr notEmptyData = std::make_shared<Data>(TEST_NAME, Precision::UNSPECIFIED, Layout::C);
    SizeVector inDims{1};
};

const std::string InputControllerTest::TEST_NAME = "TEST_NAME";

TEST_F(InputControllerTest, failedToCreateWithEmptyInsData) {
    EXPECT_THROW(InputController({}, TEST_NAME), InferenceEngineException);
}

TEST_F(InputControllerTest, failedToCreateWithNullData) {
    EXPECT_THROW(InputController({nullptr}, TEST_NAME), InferenceEngineException);
}

TEST_F(InputControllerTest, canCreateInputController) {
    ASSERT_NO_THROW(InputController({notEmptyData}, TEST_NAME));
}

TEST_F(InputControllerTest, canPushShapes) {
    InputController controller({notEmptyData}, TEST_NAME);
    ASSERT_NO_THROW(controller.setShapeByName(inDims, TEST_NAME));
}

TEST_F(InputControllerTest, throwOnGetWithNotEnoughShapes) {
    InputController controller({notEmptyData, notEmptyData}, TEST_NAME);
    controller.setShapeByName(inDims, TEST_NAME);
    ASSERT_THROW(controller.getShapes(true), InferenceEngineException);
}

TEST_F(InputControllerTest, canGetWithNotEnoughShapes) {
    InputController controller({notEmptyData, notEmptyData}, TEST_NAME);
    controller.setShapeByName(inDims, TEST_NAME);
    controller.getShapes(false);
}

TEST_F(InputControllerTest, canGetChanges) {
    InputController controller({notEmptyData}, TEST_NAME);
    controller.setShapeByName(inDims, TEST_NAME);
    ASSERT_NO_THROW(controller.getShapes(true));
}

TEST_F(InputControllerTest, throwOnApplyWithNotEnoughShapes) {
    InputController controller({notEmptyData, notEmptyData}, TEST_NAME);
    controller.setShapeByName(inDims, TEST_NAME);
    ASSERT_THROW(controller.applyChanges(), InferenceEngineException);
}

TEST_F(InputControllerTest, canApplyChanges) {
    InputController controller({notEmptyData}, TEST_NAME);
    controller.setShapeByName(inDims, TEST_NAME);
    ASSERT_NO_THROW(controller.applyChanges());
}

TEST_F(InputControllerTest, canResetShapes) {
    InputController controller({notEmptyData}, TEST_NAME);
    controller.setShapeByName(inDims, TEST_NAME);
    ASSERT_FALSE(controller.getShapes(true).empty());
    ASSERT_NO_THROW(controller.reset());
    ASSERT_THROW(controller.getShapes(true), InferenceEngineException);
}
