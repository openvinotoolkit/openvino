// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <extension/ext_list.hpp>
#include <xml_net_builder.hpp>
#include <inference_engine/cnn_network_impl.hpp>
#include <inference_engine/shape_infer/ie_reshaper.hpp>
#include <cpp/ie_cnn_net_reader.h>
#include <test_model_path.hpp>
#include <inference_engine/debug.h>
#include <extension/ext_list.hpp>
#include "built_in_shape_infer_general_test.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace ShapeInfer;

class CPUExtShapeInferTests : public BuiltInShapeInferImplTest {
protected:
    void SetUp() override {
        BuiltInShapeInferImplTest::SetUp();
        holder = std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>();
    }
};

TEST_P(CPUExtShapeInferTests, impl) {
    auto impl = getShapeInferImpl(type);
    ASSERT_NE(nullptr, impl);
    ASSERT_NO_THROW(sts = impl->inferShapes(newInOutShapes.inDims, layerParams.data, blobs, outShapes, &resp));

    if (canInfer) {
        ASSERT_EQ(int(OK), sts) << resp.msg;
        ASSERT_EQ(newInOutShapes.outDims, outShapes);
    } else {
        ASSERT_EQ(GENERAL_ERROR, sts) << resp.msg;
    }
}

TEST_P(CPUExtShapeInferTests, reshaper) {
    auto cnnNetworkImplPtr = buildSingleLayerNetwork(type, inOutShapes, &layerParams.data, layerDataName);
    auto reshaper = std::make_shared<Reshaper>(*cnnNetworkImplPtr);
    auto inputShapes = setInputShapes(*cnnNetworkImplPtr.get(), newInOutShapes.inDims);
    reshaper->AddExtension(holder);

    if (canInfer) {
        reshaper->run(inputShapes);
        checkNetworkInOut(*cnnNetworkImplPtr, newInOutShapes);
    } else {
        ASSERT_THROW(reshaper->run(inputShapes), InferenceEngine::details::InferenceEngineException);
    }
}

INSTANTIATE_TEST_CASE_P(
        CPUExtGeneralImpls, CPUExtShapeInferTests,
        ::testing::Values(
                ::testing::make_tuple(LayerType("SpatialTransformer"),
                                      InOutShapes({{{1, 6, 5, 5}, {1, 3}},
                                                   {{1, 6, 5, 5}}}),
                                      NewInOutShapes({{{2, 6, 5, 6}, {1, 3}},
                                                      {{2, 6, 5, 6}}}),
                                      MapParams(MapStrStr()),
                                      LayerDataName("data"),
                                      CanInfer(true))
        )
);
