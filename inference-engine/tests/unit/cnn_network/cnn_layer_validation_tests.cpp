/*
* INTEL CONFIDENTIAL
* Copyright (C) 2018-2019 Intel Corporation.
*
* The source code contained or described herein and all documents
* related to the source code ("Material") are owned by Intel Corporation
* or its suppliers or licensors. Title to the Material remains with
* Intel Corporation or its suppliers and licensors. The Material may
* contain trade secrets and proprietary and confidential information
* of Intel Corporation and its suppliers and licensors, and is protected
* by worldwide copyright and trade secret laws and treaty provisions.
* No part of the Material may be used, copied, reproduced, modified,
* published, uploaded, posted, transmitted, distributed, or disclosed
* in any way without Intel's prior express written permission.
*
* No license under any patent, copyright, trade secret or other
* intellectual property right is granted to or conferred upon you by
* disclosure or delivery of the Materials, either expressly, by implication,
* inducement, estoppel or otherwise. Any license under such intellectual
* property rights must be express and approved by Intel in writing.
*
* Include any supplier copyright notices as supplier requires Intel to use.
*
* Include supplier trademarks or logos as supplier requires Intel to use,
* preceded by an asterisk. An asterisked footnote can be added as follows:
* *Third Party trademarks are the property of their respective owners.
*
* Unless otherwise agreed by Intel in writing, you may not remove or alter
* this notice or any other notice embedded in Materials by Intel or Intel's
* suppliers or licensors in any way.
*/
#include <gtest/gtest.h>
#include <xml_net_builder.hpp>
#include <inference_engine/cnn_network_impl.hpp>
#include <inference_engine/ie_format_parser.h>
#include <inference_engine/ie_layer_validators.hpp>
#include <xml_helper.hpp>
#include <../shape_infer/built_in_shape_infer_general_test.hpp>
#include <memory>
#include <../include/ie_data.h>

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
