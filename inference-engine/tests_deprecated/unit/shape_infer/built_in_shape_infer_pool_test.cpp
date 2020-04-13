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

class BuiltInShapeInferPoolImplTest
        : public BuiltInShapeInferTestWithParam<std::tuple<InOutShapes, kernel, stride, pad, pool_type, exclude_pad, auto_pad, NewInOutShapes, pad_end>> {
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
        pad_end = std::get<8>(params);
    }

    std::map<std::string, std::string> getMapParams() {
        std::map<std::string, std::string> params = {
                {"kernel",      kernel.toSeparetedRow(",")},
                {"strides",     stride.toSeparetedRow(",")},
                {"pads_begin",  pad.toSeparetedRow(",")},
                {"pool-method", pool_type},
                {"exclude-pad", exclude_pad ? "false" : "true"}
        };
        if (!auto_pad.empty()) params["auto_pad"] = auto_pad;
        if (!pad_end.empty()) params["pads_end"] = pad_end.toSeparetedRow(",");
        return params;
    }

protected:
    std::string type = "Pooling";
    testing::InOutShapes inOutShapes;
    testing::InOutShapes newInOutShapes;
    param_size kernel;
    param_size stride;
    param_size pad;
    std::string pool_type;
    bool exclude_pad;
    std::string auto_pad;
    param_size pad_end;
};

TEST_P(BuiltInShapeInferPoolImplTest, body) {
    auto impl = getShapeInferImpl(type);
    ASSERT_NE(nullptr, impl);
    ASSERT_NO_THROW(sts = impl->inferShapes(getBlobs(inOutShapes.inDims), getMapParams(), blobs, outShapes, &resp));
    ASSERT_EQ(int(OK), sts) << resp.msg;
    ASSERT_EQ(inOutShapes.outDims, outShapes);
}

TEST_P(BuiltInShapeInferPoolImplTest, reshaper) {
    auto layerParams = getMapParams();
    auto cnnNetworkImplPtr = buildSingleLayerNetwork<4>(type, inOutShapes, &layerParams, "pooling_data");
    auto reshaper = std::make_shared<Reshaper>(*cnnNetworkImplPtr);
    auto inputShapes = setInputShapes(*cnnNetworkImplPtr, newInOutShapes.inDims);
    reshaper->run(inputShapes);
    checkNetworkInOut(*cnnNetworkImplPtr, newInOutShapes);
}

TEST_P(BuiltInShapeInferPoolImplTest, batch) {
    auto layerParams = getMapParams();
    auto cnnNetworkImplPtr = buildSingleLayerNetwork<4>(type, inOutShapes, &layerParams, "pooling_data");
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
                                                      {{1, 3, 229, 115}}}), pad_end()),
                // fixate pad + right/bottom
                ::testing::make_tuple(InOutShapes({{{4, 3, 228, 228}},
                                                   {{4, 3, 229, 115}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 1}), pool_type("max"), exclude_pad(true), auto_pad(""),
                                      NewInOutShapes({{{1, 3, 228, 228}},
                                                      {{1, 3, 229, 115}}}), pad_end({3, 2})),
                // valid + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3, 228, 228}},
                                                   {{4, 3, 227, 113}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), pool_type("max"), exclude_pad(true), auto_pad("valid"),
                                      NewInOutShapes({{{1, 3, 228, 228}},
                                                      {{1, 3, 227, 113}}}), pad_end()),
                // valid + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3, 228, 228}},
                                                   {{4, 3, 227, 113}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 4}), pool_type("max"), exclude_pad(true), auto_pad("valid"),
                                      NewInOutShapes({{{1, 3, 228, 228}},
                                                      {{1, 3, 227, 113}}}), pad_end({2, 1})),
                // same_upper + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3, 227, 227}},
                                                   {{4, 3, 227, 114}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), pool_type("max"), exclude_pad(true), auto_pad("same_upper"),
                                      NewInOutShapes({{{1, 3, 227, 227}},
                                                      {{1, 3, 227, 114}}}), pad_end()),
                // same_upper + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3, 227, 227}},
                                                   {{4, 3, 227, 114}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 4}), pool_type("max"), exclude_pad(true), auto_pad("same_upper"),
                                      NewInOutShapes({{{1, 3, 227, 227}},
                                                      {{1, 3, 227, 114}}}), pad_end({0, 0})),
                // same_lower + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3, 227, 227}},
                                                   {{4, 3, 227, 113}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), pool_type("max"), exclude_pad(true), auto_pad("same_lower"),
                                      NewInOutShapes({{{1, 3, 227, 227}},
                                                      {{1, 3, 227, 113}}}), pad_end({0, 0})),
                // same_lower + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3, 227, 227}},
                                                   {{4, 3, 227, 113}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 4}), pool_type("max"), exclude_pad(true), auto_pad("same_lower"),
                                      NewInOutShapes({{{1, 3, 227, 227}},
                                                      {{1, 3, 227, 113}}}), pad_end({0, 0})),
                // 5D tensors
                // fixate pad
                ::testing::make_tuple(InOutShapes({{{4, 3, 16, 128, 130}},
                                                   {{4, 3, 17, 129, 66}}}), kernel({4, 2, 2}), stride({2, 1, 1}),
                                      pad({2, 1, 1}), pool_type("max"), exclude_pad(true), auto_pad(""),
                                      NewInOutShapes({{{1, 3, 16, 128, 130}},
                                                      {{1, 3, 17, 129, 66}}}), pad_end()),
                // valid + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3, 16, 128, 130}},
                                                   {{4, 3, 15, 127, 64}}}), kernel({4, 2, 2}), stride({2, 1, 1}),
                                      pad({0, 0, 0}), pool_type("max"), exclude_pad(true), auto_pad("valid"),
                                      NewInOutShapes({{{1, 3, 16, 128, 130}},
                                                      {{1, 3, 15, 127, 64}}}), pad_end()),
                // same_upper + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3, 16, 128, 130}},
                                                   {{4, 3, 16, 128, 65}}}), kernel({4, 2, 2}), stride({2, 1, 1}),
                                      pad({0, 0, 0}), pool_type("max"), exclude_pad(true), auto_pad("same_upper"),
                                      NewInOutShapes({{{1, 3, 16, 128, 130}},
                                                      {{1, 3, 16, 128, 65}}}), pad_end())
        )
);
