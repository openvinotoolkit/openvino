// Copyright (C) 2018 Intel Corporation
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

class BuiltInShapeInferPoolImplTest
        : public BuiltInShapeInferTestWithParam<std::tuple<InOutShapes, kernel, stride, pad, pool_type, exclude_pad, auto_pad, NewInOutShapes, padrb>> {
protected:
    void SetUp() override {
        BuiltInShapeInferCommon::SetUp();
        auto params = GetParam();
        inOutShapes = std::get<0>(params);
        kernel = std::get<1>(params);
        stride = std::get<2>(params);
        pad = std::get<3>(params);
        pool_type = std::get<4>(params);
        exclude_pad = std::get<5>(params);
        auto_pad = std::get<6>(params);
        newInOutShapes = std::get<7>(params);
        padrb = std::get<8>(params);
    }

    std::map<std::string, std::string> getMapParams() {
        std::map<std::string, std::string> params{
                {"kernel-x",    std::to_string(kernel.x)},
                {"kernel-y",    std::to_string(kernel.y)},
                {"stride-x",    std::to_string(stride.x)},
                {"stride-y",    std::to_string(stride.y)},
                {"pad-x",       std::to_string(pad.x)},
                {"pad-y",       std::to_string(pad.y)},
                {"pool-method", pool_type},
                {"exclude-pad", exclude_pad ? "false" : "true"},
        };
        if (!auto_pad.empty()) params["auto_pad"] = auto_pad;
        if (padrb.x) params["pad-r"] = std::to_string(padrb.x);
        if (padrb.y) params["pad-b"] = std::to_string(padrb.y);
        return params;
    }

    std::map<std::string, std::string> getMapParams_IRv3() {
        std::map<std::string, std::string> params = {
                {"kernel",      kernel.toSeparetedRow(",")},
                {"strides",     stride.toSeparetedRow(",")},
                {"pads_begin",  pad.toSeparetedRow(",")},
                {"pool-method", pool_type},
                {"exclude-pad", exclude_pad ? "false" : "true"}
        };
        if (!auto_pad.empty()) params["auto_pad"] = auto_pad;
        if (padrb.x != 0 && padrb.y != 0) {
            params["pads_end"] = padrb.toSeparetedRow(",");
        }
        return params;
    }

protected:
    std::string type = "Pooling";
    testing::InOutData inOutShapes;
    testing::InOutData newInOutShapes;
    param_size kernel;
    param_size stride;
    param_size pad;
    std::string pool_type;
    bool exclude_pad;
    std::string auto_pad;
    param_size padrb;
};

TEST_P(BuiltInShapeInferPoolImplTest, body) {
    auto impl = getShapeInferImpl(type);
    ASSERT_NE(nullptr, impl);
    ASSERT_NO_THROW(sts = impl->inferShapes(inOutShapes.inDims, getMapParams(), blobs, outShapes, &resp));
    ASSERT_EQ(int(OK), sts) << resp.msg;
    ASSERT_EQ(inOutShapes.outDims, outShapes);
}

TEST_P(BuiltInShapeInferPoolImplTest, reshaper) {
    auto layerParams = getMapParams();
    auto cnnNetworkImplPtr = buildSingleLayerNetwork<2>(type, inOutShapes, &layerParams, "pooling_data");
    auto reshaper = std::make_shared<Reshaper>(*cnnNetworkImplPtr);
    auto inputShapes = setInputShapes(*cnnNetworkImplPtr, newInOutShapes.inDims);
    reshaper->run(inputShapes);
    checkNetworkInOut(*cnnNetworkImplPtr, newInOutShapes);
}

TEST_P(BuiltInShapeInferPoolImplTest, batch) {
    auto layerParams = getMapParams();
    auto cnnNetworkImplPtr = buildSingleLayerNetwork<2>(type, inOutShapes, &layerParams, "pooling_data");
    auto reshaper = std::make_shared<Reshaper>(*cnnNetworkImplPtr);
    sts = cnnNetworkImplPtr->setBatchSize(BATCH, &resp);
    ASSERT_EQ((int)OK, sts) << resp.msg;
    inOutShapes.inDims[0][0] = inOutShapes.outDims[0][0] = BATCH;
    checkNetworkInOut(*cnnNetworkImplPtr, inOutShapes);
}

TEST_P(BuiltInShapeInferPoolImplTest, body_IRv3) {
    auto impl = getShapeInferImpl(type);
    ASSERT_NE(nullptr, impl);
    ASSERT_NO_THROW(sts = impl->inferShapes(inOutShapes.inDims, getMapParams_IRv3(), blobs, outShapes, &resp));
    ASSERT_EQ(int(OK), sts) << resp.msg;
    ASSERT_EQ(inOutShapes.outDims, outShapes);
}

TEST_P(BuiltInShapeInferPoolImplTest, reshaper_IRv3) {
    auto layerParams = getMapParams_IRv3();
    auto cnnNetworkImplPtr = buildSingleLayerNetwork<3>(type, inOutShapes, &layerParams, "pooling_data");
    auto reshaper = std::make_shared<Reshaper>(*cnnNetworkImplPtr);
    auto inputShapes = setInputShapes(*cnnNetworkImplPtr, newInOutShapes.inDims);
    reshaper->run(inputShapes);
    checkNetworkInOut(*cnnNetworkImplPtr, newInOutShapes);
}

TEST_P(BuiltInShapeInferPoolImplTest, batch_IRv3) {
    auto layerParams = getMapParams_IRv3();
    auto cnnNetworkImplPtr = buildSingleLayerNetwork<3>(type, inOutShapes, &layerParams, "pooling_data");
    auto reshaper = std::make_shared<Reshaper>(*cnnNetworkImplPtr);
    sts = cnnNetworkImplPtr->setBatchSize(BATCH, &resp);
    ASSERT_EQ((int)OK, sts) << resp.msg;
    inOutShapes.inDims[0][0] = inOutShapes.outDims[0][0] = BATCH;
    checkNetworkInOut(*cnnNetworkImplPtr, inOutShapes);
}

INSTANTIATE_TEST_CASE_P(
        BuiltInImpls, BuiltInShapeInferPoolImplTest,
        ::testing::Values(
                // fixate pad
                ::testing::make_tuple(InOutShapes({{{4, 3, 228, 228}},
                                                   {{4, 3, 229, 115}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 1}), pool_type("max"), exclude_pad(true), auto_pad(""),
                                      NewInOutShapes({{{1, 3, 228, 228}},
                                                      {{1, 3, 229, 115}}}), padrb({0, 0})),
                // fixate pad + right/bottom
                ::testing::make_tuple(InOutShapes({{{4, 3, 228, 228}},
                                                   {{4, 3, 229, 115}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 1}), pool_type("max"), exclude_pad(true), auto_pad(""),
                                      NewInOutShapes({{{1, 3, 228, 228}},
                                                      {{1, 3, 229, 115}}}), padrb({3, 2})),
                // valid + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3, 228, 228}},
                                                   {{4, 3, 227, 113}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), pool_type("max"), exclude_pad(true), auto_pad("valid"),
                                      NewInOutShapes({{{1, 3, 228, 228}},
                                                      {{1, 3, 227, 113}}}), padrb({0, 0})),
                // valid + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3, 228, 228}},
                                                   {{4, 3, 227, 113}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 4}), pool_type("max"), exclude_pad(true), auto_pad("valid"),
                                      NewInOutShapes({{{1, 3, 228, 228}},
                                                      {{1, 3, 227, 113}}}), padrb({2, 1})),
                // same_upper + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3, 227, 227}},
                                                   {{4, 3, 227, 114}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), pool_type("max"), exclude_pad(true), auto_pad("same_upper"),
                                      NewInOutShapes({{{1, 3, 227, 227}},
                                                      {{1, 3, 227, 114}}}), padrb({0, 0})),
                // same_upper + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3, 227, 227}},
                                                   {{4, 3, 227, 114}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 4}), pool_type("max"), exclude_pad(true), auto_pad("same_upper"),
                                      NewInOutShapes({{{1, 3, 227, 227}},
                                                      {{1, 3, 227, 114}}}), padrb({0, 0})),
                // same_lower + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3, 227, 227}},
                                                   {{4, 3, 227, 113}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), pool_type("max"), exclude_pad(true), auto_pad("same_lower"),
                                      NewInOutShapes({{{1, 3, 227, 227}},
                                                      {{1, 3, 227, 113}}}), padrb({0, 0})),
                // same_lower + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3, 227, 227}},
                                                   {{4, 3, 227, 113}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 4}), pool_type("max"), exclude_pad(true), auto_pad("same_lower"),
                                      NewInOutShapes({{{1, 3, 227, 227}},
                                                      {{1, 3, 227, 113}}}), padrb({0, 0}))
        )
);
