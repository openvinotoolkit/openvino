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

class BuiltInShapeInferConvImplTest
        : public BuiltInShapeInferTestWithParam<std::tuple<InOutShapes, kernel, stride, pad, auto_pad, out_channels, group, dilation_factor, NewInOutShapes, CanInfer, pad_end, IsTransposed>> {
protected:
    void SetUp() override {
        BuiltInShapeInferCommon::SetUp();
        auto params = GetParam();
        inOutShapes = std::get<0>(params);
        kernel = std::get<1>(params);
        stride = std::get<2>(params);
        pad = std::get<3>(params);
        auto_pad = std::get<4>(params);
        out_channels = std::get<5>(params);
        group = std::get<6>(params);
        dilation_factor = std::get<7>(params);
        newInOutShapes = std::get<8>(params);
        canInfer = std::get<9>(params);
        pad_end = std::get<10>(params);
        isTransposed = std::get<11>(params);
        if (isTransposed) {
            type = "Deconvolution";
            dataName = "deconvolution_data";
        }
    }

    std::map<std::string, std::string> getMapParams() {
        std::map<std::string, std::string> params = {
                {"kernel",     kernel.toSeparetedRow(",")},
                {"strides",    stride.toSeparetedRow(",")},
                {"pads_begin", pad.toSeparetedRow(",")},
                {"output",     std::to_string(out_channels)},
                {"group",      std::to_string(group)},
                {"dilations",  dilation_factor.toSeparetedRow(",")}
        };
        if (!auto_pad.empty()) params["auto_pad"] = auto_pad;
        if (!pad_end.empty()) params["pads_end"] = pad_end.toSeparetedRow(",");
        return params;
    }

protected:
    std::string type = "Convolution";
    std::string dataName = "convolution_data";
    testing::InOutShapes inOutShapes;
    testing::InOutShapes newInOutShapes;
    param_size kernel{};
    param_size stride{};
    param_size pad{};
    param_size pad_end{};
    param_size dilation_factor{};
    std::string auto_pad;
    unsigned out_channels{};
    unsigned group{};
    bool canInfer;
    bool isTransposed;
};


TEST_P(BuiltInShapeInferConvImplTest, impl) {
    auto impl = getShapeInferImpl(type);
    ASSERT_NE(nullptr, impl);
    if (!group) group = 1;
    unsigned w_dim = out_channels * inOutShapes.inDims[0][1] / group;
    for (auto k : kernel.dims)
        w_dim *= k;
    SizeVector weightsDim{w_dim};
    blobs["weights"] = make_shared_blob(Precision::fromType<size_t>(), weightsDim);
    ASSERT_NO_THROW(sts = impl->inferShapes(getBlobs(inOutShapes.inDims), getMapParams(), blobs, outShapes, &resp));
    ASSERT_EQ(int(OK), sts) << resp.msg;
    ASSERT_EQ(inOutShapes.outDims, outShapes);
}

TEST_P(BuiltInShapeInferConvImplTest, batch) {
    auto layerParams = getMapParams();
    auto cnnNetworkImplPtr = buildSingleLayerNetwork<4>(type, inOutShapes, &layerParams, dataName);
    auto reshaper = std::make_shared<Reshaper>(*cnnNetworkImplPtr);
    sts = cnnNetworkImplPtr->setBatchSizeReshape(BATCH, &resp);
    ASSERT_EQ((int) OK, sts) << resp.msg;
    inOutShapes.inDims[0][0] = inOutShapes.outDims[0][0] = BATCH;
    checkNetworkInOut(*cnnNetworkImplPtr, inOutShapes);
}

TEST_P(BuiltInShapeInferConvImplTest, reshaper) {
    auto layerParams = getMapParams();
    auto cnnNetworkImplPtr = buildSingleLayerNetwork<4>(type, inOutShapes, &layerParams, dataName);
    auto reshaper = std::make_shared<Reshaper>(*cnnNetworkImplPtr);
    auto inputShapes = setInputShapes(*cnnNetworkImplPtr, newInOutShapes.inDims);
    reshaper->run(inputShapes);
    checkNetworkInOut(*cnnNetworkImplPtr, newInOutShapes);
}

INSTANTIATE_TEST_CASE_P(
        BuiltInImplsConv, BuiltInShapeInferConvImplTest,
        ::testing::Values(
                // fixate pad
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 229, 115}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 1}), auto_pad(""), out_channels(64), group(1), dilation_factor({0, 0}),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 229, 115}}}),
                                      CanInfer(true), pad_end(), IsTransposed(false)),
                // fixate pad + dilation
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 225, 109}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 1}), auto_pad(""), out_channels(64), group(1), dilation_factor({5, 5}),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 225, 109}}}),
                                      CanInfer(true), pad_end(), IsTransposed(false)),
                // fixate pad + right/bottom
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 230, 115}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 1}), auto_pad(""), out_channels(64), group(1), dilation_factor({0, 0}),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 230, 115}}}),
                                      CanInfer(true), pad_end({3, 2}), IsTransposed(false)),
                // valid + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 227, 113}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("valid"), out_channels(64), group(1), dilation_factor({0, 0}),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 227, 113}}}),
                                      CanInfer(true), pad_end(), IsTransposed(false)),
                // valid + dilation
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 223, 107}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("valid"), out_channels(64), group(1), dilation_factor({5, 5}),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 223, 107}}}),
                                      CanInfer(true), pad_end({0, 0}), IsTransposed(false)),
                // valid + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 227, 113}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 4}), auto_pad("valid"), out_channels(64), group(1), dilation_factor({0, 0}),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 227, 113}}}),
                                      CanInfer(true), pad_end({3, 2}), IsTransposed(false)),
                // same_upper + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 114}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("same_upper"), out_channels(64), group(1),
                                      dilation_factor({0, 0}),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 114}}}),
                                      CanInfer(true), pad_end(), IsTransposed(false)),
                // same_upper + dilation paddings
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 114}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("same_upper"), out_channels(64), group(1),
                                      dilation_factor({5, 5}),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 114}}}),
                                      CanInfer(true), pad_end({0, 0}), IsTransposed(false)),
                // same_upper + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 114}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 4}), auto_pad("same_upper"), out_channels(64), group(1),
                                      dilation_factor({0, 0}),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 114}}}),
                                      CanInfer(true), pad_end({0, 0}), IsTransposed(false)),
                // same_lower + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 113}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("same_lower"), out_channels(64), group(1),
                                      dilation_factor({0, 0}),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 113}}}),
                                      CanInfer(true), pad_end(), IsTransposed(false)),
                // same_lower + dilation
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 113}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("same_lower"), out_channels(64), group(1),
                                      dilation_factor({0, 0}),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 113}}}),
                                      CanInfer(true), pad_end({0, 0}), IsTransposed(false)),
                // same_lower + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 113}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 4}), auto_pad("same_lower"), out_channels(64), group(1),
                                      dilation_factor({0, 0}),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 113}}}),
                                      CanInfer(true), pad_end({0, 0}), IsTransposed(false)),
                // 5D tensors
                // fixate pad
                ::testing::make_tuple(InOutShapes({{{4, 3, 64, 100, 120}},
                                                   {{4, 64, 66, 101, 61}}}), kernel({4, 2, 1}), stride({2, 1, 1}),
                                      pad({2, 1, 1}), auto_pad(""), out_channels(64), group(1), dilation_factor({0, 0, 0}),
                                      NewInOutShapes({{{1, 3, 64, 100, 120}},
                                                      {{1, 64, 66, 101, 61}}}),
                                      CanInfer(true), pad_end(), IsTransposed(false)),
                // fixate pad + right/bottom
                ::testing::make_tuple(InOutShapes({{{4, 3, 16, 128, 128}},
                                                   {{4, 64, 18, 130, 65}}}), kernel({4, 2, 2}), stride({2, 1, 1}),
                                      pad({2, 1, 1}), auto_pad(""), out_channels(64), group(1), dilation_factor({0, 0, 0}),
                                      NewInOutShapes({{{1, 3, 16, 128, 128}},
                                                      {{1, 64, 18, 130, 65}}}),
                                      CanInfer(true), pad_end({3, 2, 2}), IsTransposed(false)),
                // valid + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3, 16, 128, 130}},
                                                   {{4, 64, 15, 127, 64}}}), kernel({4, 2, 2}), stride({2, 1, 1}),
                                      pad({2, 4, 2}), auto_pad("valid"), out_channels(64), group(1), dilation_factor({0, 0, 0}),
                                      NewInOutShapes({{{1, 3, 16, 128, 130}},
                                                      {{1, 64, 15, 127, 64}}}),
                                      CanInfer(true), pad_end({3, 2, 2}), IsTransposed(false)),
                // same_lower + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3, 16, 128, 130}},
                                                   {{4, 64, 16, 128, 65}}}), kernel({4, 2, 1}), stride({2, 1, 1}),
                                      pad({0, 0, 0}), auto_pad("same_lower"), out_channels(64), group(1),
                                      dilation_factor({0, 0, 0}),
                                      NewInOutShapes({{{1, 3, 16, 128, 130}},
                                                      {{1, 64, 16, 128, 65}}}),
                                      CanInfer(true), pad_end(), IsTransposed(false))
        )
);

INSTANTIATE_TEST_CASE_P(
        BuiltInImplsDeConv, BuiltInShapeInferConvImplTest,
        ::testing::Values(
                // fixate pad
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 227, 454}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 1}), auto_pad(""), out_channels(64), group(1), dilation_factor({0, 0}),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 227, 454}}}),
                                      CanInfer(true), pad_end(), IsTransposed(true)),
                // fixate pad + dilation
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 231, 466}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 1}), auto_pad(""), out_channels(64), group(1), dilation_factor({5, 5}),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 231, 466}}}),
                                      CanInfer(true), pad_end(), IsTransposed(true)),
                // fixate pad + right/bottom
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 226, 453}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 1}), auto_pad(""), out_channels(64), group(1), dilation_factor({0, 0}),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 226, 453}}}),
                                      CanInfer(true), pad_end({3, 2}), IsTransposed(true)),
                // valid + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 229, 459}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("valid"), out_channels(64), group(1), dilation_factor({0, 0}),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 229, 459}}}),
                                      CanInfer(true), pad_end({0, 0}), IsTransposed(true)),
                // valid + dilation
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 233, 471}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("valid"), out_channels(64), group(1), dilation_factor({5, 5}),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 233, 471}}}),
                                      CanInfer(true), pad_end({0, 0}), IsTransposed(true)),
                // valid + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 233, 471}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 4}), auto_pad("valid"), out_channels(64), group(1), dilation_factor({5, 5}),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 233, 471}}}),
                                      CanInfer(true), pad_end({3, 2}), IsTransposed(true)),
                // same_upper + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 454}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("same_upper"), out_channels(64), group(1),
                                      dilation_factor({0, 0}),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 454}}}),
                                      CanInfer(true), pad_end({0, 0}), IsTransposed(true)),
                // same_upper + dilation paddings
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 454}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("same_upper"), out_channels(64), group(1),
                                      dilation_factor({5, 5}),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 454}}}),
                                      CanInfer(true), pad_end({0, 0}), IsTransposed(true)),
                // same_upper + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 454}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 4}), auto_pad("same_upper"), out_channels(64), group(1),
                                      dilation_factor({0, 0}),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 454}}}),
                                      CanInfer(true), pad_end({0, 0}), IsTransposed(true)),
                // same_lower + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 454}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("same_lower"), out_channels(64), group(1),
                                      dilation_factor({0, 0}),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 454}}}),
                                      CanInfer(true), pad_end({0, 0}), IsTransposed(true)),
                // same_lower + dilation
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 454}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("same_lower"), out_channels(64), group(1),
                                      dilation_factor({0, 0}),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 454}}}),
                                      CanInfer(true), pad_end({0, 0}), IsTransposed(true)),
                // same_lower + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 454}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 4}), auto_pad("same_lower"), out_channels(64), group(1),
                                      dilation_factor({0, 0}),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 454}}}),
                                      CanInfer(true), pad_end({0, 0}), IsTransposed(true)),
                // 5D tensors
                // fixate pad
                ::testing::make_tuple(InOutShapes({{{4, 3, 64, 100, 120}},
                                                   {{4, 64, 66, 101, 61}}}), kernel({4, 2, 1}), stride({2, 1, 1}),
                                      pad({2, 1, 1}), auto_pad(""), out_channels(64), group(1), dilation_factor({0, 0 ,0}),
                                      NewInOutShapes({{{1, 3,  64, 100, 120}},
                                                      {{1, 64, 66, 101, 61}}}),
                                      CanInfer(true), pad_end(), IsTransposed(false)),
                // fixate pad + right/bottom
                ::testing::make_tuple(InOutShapes({{{4, 3, 16, 128, 130}},
                                                   {{4, 64, 14, 126, 257}}}), kernel({4, 2, 2}), stride({2, 1, 1}),
                                      pad({2, 1, 1}), auto_pad(""), out_channels(64), group(1), dilation_factor({0, 0, 0}),
                                      NewInOutShapes({{{1, 3, 16, 128, 130}},
                                                      {{1, 64, 14, 126, 257 }}}),
                                      CanInfer(true), pad_end({3, 2, 2}), IsTransposed(true)),
                // valid + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3, 16, 128, 130}},
                                                   {{4, 64, 15, 127, 64}}}), kernel({4, 2, 2}), stride({2, 1, 1}),
                                      pad({2, 4, 2}), auto_pad("valid"), out_channels(64), group(1), dilation_factor({0, 0, 0}),
                                      NewInOutShapes({{{1, 3, 16, 128, 130}},
                                                      {{1, 64, 15, 127, 64}}}),
                                      CanInfer(true), pad_end({3, 2, 2}), IsTransposed(false)),
                // same_lower + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3, 16, 128, 130}},
                                                   {{4, 64, 16, 128, 65}}}), kernel({4, 2, 1}), stride({2, 1, 1}),
                                      pad({0, 0, 0}), auto_pad("same_lower"), out_channels(64), group(1),
                                      dilation_factor({0, 0, 0}),
                                      NewInOutShapes({{{1, 3, 16, 128, 130}},
                                                      {{1, 64, 16, 128, 65}}}),
                                      CanInfer(true), pad_end(), IsTransposed(false))
        )
);
