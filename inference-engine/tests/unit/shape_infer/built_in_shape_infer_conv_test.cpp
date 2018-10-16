// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <inference_engine/shape_infer/built-in/ie_built_in_holder.hpp>
#include <xml_net_builder.hpp>
#include <inference_engine/cnn_network_impl.hpp>
#include <inference_engine/v2_format_parser.h>
#include <xml_helper.hpp>
#include <inference_engine/shape_infer/ie_reshaper.hpp>
#include "built_in_shape_infer_general_test.hpp"

using namespace InferenceEngine;
using namespace ShapeInfer;

class BuiltInShapeInferConvImplTest
        : public BuiltInShapeInferTestWithParam<std::tuple<InOutShapes, kernel, stride, pad, auto_pad, out_channels, group, dilation_factor, NewInOutShapes, CanInfer, padrb, IsTransposed>> {
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
        padrb = std::get<10>(params);
        isTransposed = std::get<11>(params);
        if (isTransposed) {
            type = "Deconvolution";
            dataName = "deconvolution_data";
        }
    }

    std::map<std::string, std::string> getMapParams() {
        std::map<std::string, std::string> params = {
                {"kernel-x",   std::to_string(kernel.x)},
                {"kernel-y",   std::to_string(kernel.y)},
                {"stride-x",   std::to_string(stride.x)},
                {"stride-y",   std::to_string(stride.y)},
                {"pad-x",      std::to_string(pad.x)},
                {"pad-y",      std::to_string(pad.y)},
                {"output",     std::to_string(out_channels)},
                {"group",      std::to_string(group)},
                {"dilation-x", std::to_string(dilation_factor)},
                {"dilation-y", std::to_string(dilation_factor)}
        };
        if (!auto_pad.empty()) params["auto_pad"] = auto_pad;
        if (padrb.x) params["pad-r"] = std::to_string(padrb.x);
        if (padrb.y) params["pad-b"] = std::to_string(padrb.y);
        return params;
    }

protected:
    std::string type = "Convolution";
    std::string dataName = "convolution_data";
    testing::InOutData inOutShapes;
    testing::InOutData newInOutShapes;
    param_size kernel{};
    param_size stride{};
    param_size pad{};
    param_size padrb{};
    std::string auto_pad;
    unsigned out_channels{};
    unsigned group{};
    unsigned dilation_factor{};
    bool canInfer;
    bool isTransposed;
};


TEST_P(BuiltInShapeInferConvImplTest, impl) {
    auto impl = getShapeInferImpl(type);
    ASSERT_NE(nullptr, impl);
    if (!group) group = 1;
    SizeVector weightsDim{kernel.x * kernel.y * out_channels * inOutShapes.inDims[0][1] / group};
    blobs["weights"] = make_shared_blob(Precision::UNSPECIFIED, weightsDim);
    ASSERT_NO_THROW(sts = impl->inferShapes(inOutShapes.inDims, getMapParams(), blobs, outShapes, &resp));
    ASSERT_EQ(int(OK), sts) << resp.msg;
    ASSERT_EQ(inOutShapes.outDims, outShapes);
}

TEST_P(BuiltInShapeInferConvImplTest, batch) {
    auto layerParams = getMapParams();
    auto cnnNetworkImplPtr = buildSingleLayerNetwork(type, inOutShapes, &layerParams, dataName);
    auto reshaper = std::make_shared<Reshaper>(*cnnNetworkImplPtr);
    sts = cnnNetworkImplPtr->setBatchSizeReshape(BATCH, &resp);
    ASSERT_EQ((int) OK, sts) << resp.msg;
    inOutShapes.inDims[0][0] = inOutShapes.outDims[0][0] = BATCH;
    checkNetworkInOut(*cnnNetworkImplPtr, inOutShapes);
}

TEST_P(BuiltInShapeInferConvImplTest, reshaper) {
    auto layerParams = getMapParams();
    auto cnnNetworkImplPtr = buildSingleLayerNetwork(type, inOutShapes, &layerParams, dataName);
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
                                      pad({2, 1}), auto_pad(""), out_channels(64), group(1), dilation_factor(0),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 229, 115}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(false)),
                // fixate pad + dilation
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 225, 109}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 1}), auto_pad(""), out_channels(64), group(1), dilation_factor(5),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 225, 109}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(false)),
                // fixate pad + right/bottom
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 230, 115}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 1}), auto_pad(""), out_channels(64), group(1), dilation_factor(0),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 230, 115}}}),
                                      CanInfer(true), padrb({3, 2}), IsTransposed(false)),
                // valid + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 227, 113}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("valid"), out_channels(64), group(1), dilation_factor(0),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 227, 113}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(false)),
                // valid + dilation
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 223, 107}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("valid"), out_channels(64), group(1), dilation_factor(5),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 223, 107}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(false)),
                // valid + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 227, 113}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 4}), auto_pad("valid"), out_channels(64), group(1), dilation_factor(0),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 227, 113}}}),
                                      CanInfer(true), padrb({3, 2}), IsTransposed(false)),
                // same_upper + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 114}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("same_upper"), out_channels(64), group(1),
                                      dilation_factor(0),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 114}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(false)),
                // same_upper + dilation paddings
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 114}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("same_upper"), out_channels(64), group(1),
                                      dilation_factor(5),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 114}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(false)),
                // same_upper + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 114}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 4}), auto_pad("same_upper"), out_channels(64), group(1),
                                      dilation_factor(0),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 114}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(false)),
                // same_lower + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 113}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("same_lower"), out_channels(64), group(1),
                                      dilation_factor(0),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 113}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(false)),
                // same_lower + dilation
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 113}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("same_lower"), out_channels(64), group(1),
                                      dilation_factor(0),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 113}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(false)),
                // same_lower + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 113}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 4}), auto_pad("same_lower"), out_channels(64), group(1),
                                      dilation_factor(0),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 113}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(false))
        )
);

INSTANTIATE_TEST_CASE_P(
        BuiltInImplsDeConv, BuiltInShapeInferConvImplTest,
        ::testing::Values(
                // fixate pad
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 227, 454}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 1}), auto_pad(""), out_channels(64), group(1), dilation_factor(0),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 227, 454}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(true)),
                // fixate pad + dilation
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 231, 466}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 1}), auto_pad(""), out_channels(64), group(1), dilation_factor(5),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 231, 466}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(true)),
                // fixate pad + right/bottom
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 226, 453}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 1}), auto_pad(""), out_channels(64), group(1), dilation_factor(0),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 226, 453}}}),
                                      CanInfer(true), padrb({3, 2}), IsTransposed(true)),
                // valid + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 229, 459}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("valid"), out_channels(64), group(1), dilation_factor(0),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 229, 459}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(true)),
                // valid + dilation
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 233, 471}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("valid"), out_channels(64), group(1), dilation_factor(5),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 233, 471}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(true)),
                // valid + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3,  228, 228}},
                                                   {{4, 64, 233, 471}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 4}), auto_pad("valid"), out_channels(64), group(1), dilation_factor(5),
                                      NewInOutShapes({{{1, 3,  228, 228}},
                                                      {{1, 64, 233, 471}}}),
                                      CanInfer(true), padrb({3, 2}), IsTransposed(true)),
                // same_upper + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 454}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("same_upper"), out_channels(64), group(1),
                                      dilation_factor(0),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 454}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(true)),
                // same_upper + dilation paddings
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 454}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("same_upper"), out_channels(64), group(1),
                                      dilation_factor(5),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 454}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(true)),
                // same_upper + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 454}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 4}), auto_pad("same_upper"), out_channels(64), group(1),
                                      dilation_factor(0),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 454}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(true)),
                // same_lower + empty paddings
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 454}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("same_lower"), out_channels(64), group(1),
                                      dilation_factor(0),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 454}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(true)),
                // same_lower + dilation
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 454}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({0, 0}), auto_pad("same_lower"), out_channels(64), group(1),
                                      dilation_factor(0),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 454}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(true)),
                // same_lower + fixated paddings (shouldn't affect)
                ::testing::make_tuple(InOutShapes({{{4, 3,  227, 227}},
                                                   {{4, 64, 227, 454}}}), kernel({4, 2}), stride({2, 1}),
                                      pad({2, 4}), auto_pad("same_lower"), out_channels(64), group(1),
                                      dilation_factor(0),
                                      NewInOutShapes({{{1, 3,  227, 227}},
                                                      {{1, 64, 227, 454}}}),
                                      CanInfer(true), padrb({0, 0}), IsTransposed(true))
        )
);
