// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <gtest/gtest.h>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <legacy/ngraph_ops/fully_connected.hpp>

#include <legacy/details/ie_cnn_network_tools.h>
#include <legacy/convert_function_to_cnn_network.hpp>
#include <bf16transformer.h>

using ngraph::Shape;
using ngraph::element::Type;
using namespace ngraph::opset5;
using std::make_shared;
using InferenceEngine::Precision;

std::map<std::string, InferenceEngine::CNNLayerPtr> get_layer_collection(InferenceEngine::CNNNetwork net) {
    IE_SUPPRESS_DEPRECATED_START
    auto all_layers = InferenceEngine::details::CNNNetSortTopologically(net);

    std::map<std::string, InferenceEngine::CNNLayerPtr> res;
    for (auto &layer : all_layers) {
        res[layer->name] = layer;
    }
    IE_SUPPRESS_DEPRECATED_END
    return res;
}

enum TypeOfNet { NG, IE };
InferenceEngine::CNNNetwork create_net(std::shared_ptr<ngraph::Function> &func, TypeOfNet type) {
    InferenceEngine::CNNNetwork ng_net(func);
    if (type == NG)
        return ng_net;
    else
        return InferenceEngine::CNNNetwork {InferenceEngine::details::convertFunctionToICNNNetwork(func, ng_net)};
}


TEST(BF16TransformerTest, KeepMemoryPrecision) {
    /*
     *  Suggested pattern
     *     _______   _____
     *    [_mem_r_] [_inp_]
     *        _|______|_
     *       [___mul____]
     *          __|__
     *         [_sig_]
     *          __|__
     *         [_fc1_]
     *         ___|____
     *     ___|___   __|__
     *    [_mem_w_] [_fc2_]
     *               __|__
     *              [_out_]
     *
     *  If does'n care about memory precision the mem_w will have precicion of data
     *  between fc1 and fc2 operations. In case of enabled BF16 it should be BF16.
     *  However mem_r still keep original precision.
     */
    Shape shape = {3, 2};
    Type type = ngraph::element::f32;
    auto input = make_shared<Parameter>(type, shape);
    auto mem_i = make_shared<Constant>(type, shape, 0);
    auto mem_r = make_shared<ReadValue>(mem_i, "id");
    mem_r->set_friendly_name("mem_r");

    auto mul = make_shared<ngraph::op::v1::Multiply>(mem_r, input);
    auto sig = make_shared<Sigmoid>(mul);

    auto fc1_w = make_shared<Constant>(type, Shape{2, 2}, 1);
    auto fc1_b = make_shared<Constant>(type, Shape{2}, 1);
    auto fc1 = make_shared<ngraph::op::FullyConnected>(sig, fc1_w, fc1_b, shape);

    auto fc2_w = make_shared<Constant>(type, Shape{2, 2}, 1);
    auto fc2_b = make_shared<Constant>(type, Shape{2}, 1);
    auto fc2 = make_shared<ngraph::op::FullyConnected>(fc1, fc2_w, fc2_b, shape);

    auto mem_w = make_shared<Assign>(fc1, "id");
    mem_w->set_friendly_name("mem_w");

    // WA. Limitation of ngraph. control_dependency are required.
    mem_w->add_control_dependency(mem_r);
    fc2->add_control_dependency(mem_w);

    auto function = make_shared<ngraph::Function>(
            ngraph::NodeVector      {fc2},
            ngraph::ParameterVector {input});

    auto net = create_net(function, IE);

    // Apply tested BF16 transformation
    MKLDNNPlugin::BF16Transformer transformer;
    transformer.convertToBFloat16(net);

    // Check precision
    auto layers = get_layer_collection(net);
    IE_SUPPRESS_DEPRECATED_START
    Precision prc_mem_r = layers["mem_r"]->outData[0]->getPrecision();
    Precision prc_mem_w = layers["mem_w"]->insData[0].lock()->getPrecision();
    IE_SUPPRESS_DEPRECATED_END

    ASSERT_EQ(prc_mem_r, Precision::BF16);
    ASSERT_EQ(prc_mem_w, Precision::BF16);
}

TEST(BF16TransformerTest, DISABLED_KeepMemoryPrecisionWithGEMM) {
    /*     _______   _____
     *    [_mem_r_] [_inp_]
     *        _|______|_
     *       [___mul____]
     *          __|__
     *         [_sig_]
     *          __|____
     *         [_gemm1_]
     *         ___|____
     *     ___|___   __|____
     *    [_mem_w_] [_gemm2_]
     *               __|__
     *              [_out_]
     *
     *  Same as KeepMemoryPrecision test with replacing FC -> GEMM
     */
    Shape shape = {3, 2};
    Type type = ngraph::element::f32;
    auto input = make_shared<Parameter>(type, shape);
    auto mem_i = make_shared<Constant>(type, shape, 0);
    auto mem_r = make_shared<ReadValue>(mem_i, "id");
    mem_r->set_friendly_name("mem_r");

    auto mul = make_shared<ngraph::op::v1::Multiply>(mem_r, input);
    auto sig = make_shared<Sigmoid>(mul);

    auto fc1_w = make_shared<Constant>(type, Shape{2, 2}, 1);
    auto fc1 = make_shared<MatMul>(sig, fc1_w);

    auto fc2_w = make_shared<Constant>(type, Shape{2, 2}, 1);
    auto fc2 = make_shared<MatMul>(fc1, fc2_w);

    auto mem_w = make_shared<Assign>(fc1, "id");
    mem_w->set_friendly_name("mem_w");

    // WA. Limitation of ngraph. control_dependency are required.
    mem_w->add_control_dependency(mem_r);
    fc2->add_control_dependency(mem_w);

    auto function = make_shared<ngraph::Function>(
            ngraph::NodeVector      {fc2},
            ngraph::ParameterVector {input});

    auto net = create_net(function, IE);

    // Apply tested BF16 transformation
    MKLDNNPlugin::BF16Transformer transformer;
    transformer.convertToBFloat16(net);

    // Check precision
    auto layers = get_layer_collection(net);
    IE_SUPPRESS_DEPRECATED_START
    Precision prc_mem_r = layers["mem_r"]->outData[0]->getPrecision();
    Precision prc_mem_w = layers["mem_w"]->insData[0].lock()->getPrecision();
    IE_SUPPRESS_DEPRECATED_END

    ASSERT_EQ(prc_mem_r, Precision::BF16);
    ASSERT_EQ(prc_mem_w, Precision::BF16);
}
