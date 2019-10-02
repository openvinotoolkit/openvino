// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <inference_engine/shape_infer/built-in/ie_built_in_holder.hpp>
#include <xml_net_builder.hpp>
#include <inference_engine/cnn_network_impl.hpp>
#include <inference_engine/ie_format_parser.h>
#include <xml_helper.hpp>
#include <inference_engine/shape_infer/ie_reshaper.hpp>
#include "built_in_shape_infer_general_test.hpp"

using namespace InferenceEngine;
using namespace ShapeInfer;

class BuiltInShapeInferImplFakeTest : public BuiltInShapeInferImplTest {
};

TEST_P(BuiltInShapeInferImplFakeTest, reshaper) {
    auto cnnNetworkImplPtr = buildSingleLayerNetwork<3>(type, inOutShapes, &layerParams.data, layerDataName);
    auto reshaper = std::make_shared<Reshaper>(*cnnNetworkImplPtr);
    auto inputShapes = setInputShapes(*cnnNetworkImplPtr, newInOutShapes.inDims);

    if (canInfer) {
        reshaper->run(inputShapes);
        checkNetworkInOut(*cnnNetworkImplPtr, newInOutShapes);
    } else {
        ASSERT_THROW(reshaper->run(inputShapes), InferenceEngine::details::InferenceEngineException);
    }
}

//TODO: use static variables for dimensions and parameters!!
//TODO: think about shorter instantiation

INSTANTIATE_TEST_CASE_P(
        BuiltInImplsFake2, BuiltInShapeInferImplFakeTest,
        ::testing::Values(
                ::testing::make_tuple(LayerType("NOT_KNOWN"),
                                      InOutShapes({{{1, 2, 3, 4}, {1, 2}, {1, 2, 3}},
                                                   {{2, 1},       {2, 1}, {2, 1}}}),
                                      NewInOutShapes({{{1, 2, 3, 4}, {1, 2}, {1, 2, 3}},
                                                      {{2, 1},       {2, 1}, {2, 1}}}),
                                      MapParams(MapStrStr()),
                                      LayerDataName("data"),
                                      CanInfer(true)),
                ::testing::make_tuple(LayerType("NOT_KNOWN"),
                                      InOutShapes({{{1, 2, 3, 4}},
                                                   {{2, 1}}}),
                                      NewInOutShapes({{{BATCH, 2, 3, 4}},
                                                      {{BATCH, 1}}}),
                                      MapParams(MapStrStr()),
                                      LayerDataName("data"),
                                      CanInfer(false)),
                ::testing::make_tuple(LayerType("NOT_KNOWN"),
                                      InOutShapes({{{1, 2, 3, 4}, {1, 2}},
                                                   {{2, 1},       {2, 1}, {2, 1}}}),
                                      NewInOutShapes({{{1, 2, 3, 4}, {BATCH, 2}},
                                                      {{2, 1},       {2,     1}, {2, 1}}}),
                                      MapParams(MapStrStr()),
                                      LayerDataName("data"),
                                      CanInfer(false)),
                ::testing::make_tuple(LayerType("NOT_KNOWN"),
                                      InOutShapes({{{1, 2, 3, 4}},
                                                   {{2, 1}, {2, 1}, {2, 1}}}),
                                      NewInOutShapes({{{BATCH, 2, 3, 4}},
                                                      {{BATCH, 1}, {BATCH, 1}, {BATCH, 1}}}),
                                      MapParams(MapStrStr()),
                                      LayerDataName("data"),
                                      CanInfer(false)),
                ::testing::make_tuple(LayerType("NOT_KNOWN"),
                                      InOutShapes({{{1, 2, 3, 4}},
                                                   {{2, 1}, {2, 1}, {2, 1}}}),
                                      NewInOutShapes({{{1, BATCH, 3, 4}},
                                                      {{2, 1}, {2, 1}, {2, 1}}}),
                                      MapParams(MapStrStr()),
                                      LayerDataName("data"),
                                      CanInfer(false)))
);
