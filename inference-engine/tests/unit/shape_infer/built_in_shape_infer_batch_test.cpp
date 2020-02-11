// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <shape_infer/built-in/ie_built_in_holder.hpp>
#include <xml_net_builder.hpp>
#include <cnn_network_impl.hpp>
#include <ie_format_parser.h>
#include <xml_helper.hpp>
#include <shape_infer/ie_reshaper.hpp>
#include "built_in_shape_infer_general_test.hpp"

using namespace InferenceEngine;
using namespace ShapeInfer;

class BuiltInShapeInferImplTestBatch : public BuiltInShapeInferImplTest {};

TEST_P(BuiltInShapeInferImplTestBatch, batch) {
    auto cnnNetworkImplPtr = buildSingleLayerNetwork<3>(type, inOutShapes, &layerParams.data, layerDataName);
    auto reshaper = std::make_shared<Reshaper>(*cnnNetworkImplPtr);

    if (canInfer) {
        StatusCode sts = cnnNetworkImplPtr->setBatchSizeReshape(BATCH, &resp);
        ASSERT_EQ((int)OK, sts) << resp.msg;
        checkNetworkInOut(*cnnNetworkImplPtr, newInOutShapes);
    } else {
        sts = cnnNetworkImplPtr->setBatchSizeReshape(BATCH, &resp);
        ASSERT_EQ(GENERAL_ERROR, sts) << resp.msg;
    }
}

// TBD: instantiate
