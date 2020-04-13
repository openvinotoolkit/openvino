// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_layer_validators.hpp>
#include <shape_infer/built_in_shape_infer_general_test.hpp>
#include <memory>
#include <ie_data.h>

#include "layer_builder.h"
#include "shapes.h"
using namespace InferenceEngine;
using namespace InferenceEngine::details;

TEST_P(CNNLayerValidationTests, checkValidParams) {

    assertThat(type)->setParams(valid_params);
    auto layer = getLayer();
    LayerValidator::Ptr validator = LayerValidators::getInstance()->getValidator(type);

    ASSERT_NO_THROW(validator->parseParams(layer.get()));
    ASSERT_NO_THROW(validator->checkParams(layer.get()));
}

TEST_P(CNNLayerValidationTests, checkInvalidParams) {

    assertThat(type);
    int numberOfParams = getNumOfParams();
    LayerValidator::Ptr validator = LayerValidators::getInstance()->getValidator(type);
    auto layer_ = getLayer();
    for (int i = 0; i < numberOfParams; ++i) {
        layer->setParams(!valid_params);
        ASSERT_THROW(validator->parseParams(layer_.get()), InferenceEngineException);
        ASSERT_THROW(validator->checkParams(layer_.get()), InferenceEngineException);
    }
}

TEST_P(CNNLayerValidationTests, checkInvalidInputShapes) {
    LayerValidator::Ptr validator = LayerValidators::getInstance()->getValidator(type);
    std::vector<DataPtr> spData;
    assertThat(type)->setShapes(spData, !valid_input);

    auto layer_ = getLayer();
    InOutDims shapes;
    InferenceEngine::details::getInOutShapes(layer_.get(), shapes);
    ASSERT_THROW(validator->checkShapes(layer_.get(), shapes.inDims), InferenceEngineException);
}

TEST_P(CNNLayerValidationTests, checkValidShapes) {

    std::vector<DataPtr> spData;
    assertThat(type)->setShapes(spData, valid_input);
    auto layer = getLayer();
    LayerValidator::Ptr validator = LayerValidators::getInstance()->getValidator(type);
    InOutDims shapes;
    InferenceEngine::details::getInOutShapes(layer.get(), shapes);
    ASSERT_NO_THROW(validator->checkShapes(layer.get(), shapes.inDims));
}

INSTANTIATE_TEST_CASE_P(
        InstantiationName, CNNLayerValidationTests,
        ::testing::Values(
                "Convolution"
                ,"Deconvolution"
                ,"DetectionOutput"
        )
);
