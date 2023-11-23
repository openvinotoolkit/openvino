// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/recurrent_cell.hpp"

#include <openvino/opsets/opset1.hpp>
#include "ov_ops/type_relaxed.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "low_precision/rt_info/intervals_alignment_attribute.hpp"
#include "low_precision/rt_info/quantization_alignment_attribute.hpp"

#include "ov_models/builders.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ov::pass;

std::shared_ptr<ov::Model> RecurrentCellFunction::get(
    const ov::element::Type inputPrecision,
    const std::vector<ov::PartialShape>& inputActivationsShapes,
    const std::vector<ov::Shape>& inputWeightsShapes,
    const RNNType type,
    const std::vector<FakeQuantizeOnDataWithConstant>& fqOnDatas,
    const std::vector<DequantizationOperations::Convert>& converts,
    const std::vector<DequantizationOperations>& dequantizations) {
    auto X = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputActivationsShapes[0]);
    X->set_friendly_name("X");
    std::shared_ptr<Node> parent_X = makeQuantizationAndDequantization(X,
                                                                       inputPrecision,
                                                                       X->get_friendly_name(),
                                                                       fqOnDatas[0],
                                                                       converts[0],
                                                                       dequantizations[0]);
    auto H = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputActivationsShapes[1]);
    H->set_friendly_name("H");
    std::shared_ptr<Node> parent_H = makeQuantizationAndDequantization(H,
                                                                       inputPrecision,
                                                                       H->get_friendly_name(),
                                                                       fqOnDatas[1],
                                                                       converts[1],
                                                                       dequantizations[1]);
    auto C = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputActivationsShapes[2]);
    C->set_friendly_name("C");

    auto W = ov::opset1::Constant::create(fqOnDatas[2].empty() ? ov::element::i8 : inputPrecision,
                                              inputWeightsShapes[0],
                                              {1});
    std::shared_ptr<Node> parent_W = makeQuantizationAndDequantization(W,
                                                                       inputPrecision,
                                                                       W->get_friendly_name(),
                                                                       fqOnDatas[2],
                                                                       converts[2],
                                                                       dequantizations[2]);
    auto R = ov::opset1::Constant::create(fqOnDatas[2].empty() ? ov::element::i8 : inputPrecision,
                                              inputWeightsShapes[1],
                                              {1});
    std::shared_ptr<Node> parent_R = makeQuantizationAndDequantization(R,
                                                                       inputPrecision,
                                                                       R->get_friendly_name(),
                                                                       fqOnDatas[3],
                                                                       converts[3],
                                                                       dequantizations[3]);
    auto B = ov::opset1::Constant::create(inputPrecision, inputWeightsShapes[2], {1});
    auto max_seq_length = inputActivationsShapes[0][1].get_max_length();
    auto seq_lengths = ov::opset1::Constant::create(element::i32, Shape{1}, {max_seq_length});

    std::shared_ptr<ov::op::util::RNNCellBase> rnn_layer;
    switch (type) {
    case RNNType::LSTMSequence:
        rnn_layer = std::make_shared<ov::op::v0::LSTMSequence>(parent_X,
                                                               parent_H,
                                                               C,
                                                               seq_lengths,
                                                               parent_W,
                                                               parent_R,
                                                               B,
                                                               128,
                                                               op::RecurrentSequenceDirection::FORWARD);
        rnn_layer->set_friendly_name("lstm_sequense");
        break;
    case RNNType::GRUSequence:
        rnn_layer = std::make_shared<ov::op::v5::GRUSequence>(parent_X,
                                                              parent_H,
                                                              seq_lengths,
                                                              parent_W,
                                                              parent_R,
                                                              B,
                                                              3,
                                                              op::RecurrentSequenceDirection::FORWARD);
        rnn_layer->set_friendly_name("gru_sequence");
        break;
    default:
        break;
    }

    auto& rtInfo = rnn_layer->get_rt_info();
    bool is_lstm = type == RNNType::LSTMSequence;
    rtInfo["Variant::std::string"] = "rnn_layer";

    auto rnn_layer_res_1 = std::make_shared<ov::op::v0::Result>(rnn_layer->output(0));
    rnn_layer_res_1->set_friendly_name("output_1");
    std::shared_ptr<ov::op::v0::Result> rnn_layer_res_2 = {};
    if (is_lstm) {
        rnn_layer_res_2 = std::make_shared<ov::op::v0::Result>(rnn_layer->output(1));
        rnn_layer_res_2->set_friendly_name("output_2");
    }

    ov::ResultVector results{rnn_layer_res_2 ? rnn_layer_res_1, rnn_layer_res_2 : rnn_layer_res_1};
    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        is_lstm ? ov::ParameterVector{X, H, C} : ov::ParameterVector{X, H},
        "LSTMTransformation");

    return function;
}

std::shared_ptr<Node> makeQuantizationAndDequantization(const std::shared_ptr<Node> input,
                                                        const ov::element::Type inputPrecision,
                                                        const std::string friendly_name,
                                                        const FakeQuantizeOnDataWithConstant& fqOnData,
                                                        const DequantizationOperations::Convert& convert,
                                                        const DequantizationOperations& dequantization) {
    std::shared_ptr<Node> parent;
    if (fqOnData.empty()) {
        parent = input;
    } else {
        std::shared_ptr<Node> fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input, inputPrecision, fqOnData);
        fakeQuantize1->set_friendly_name("fakeQuantize_" + friendly_name);
        parent = fakeQuantize1;
    }
    if (!convert.empty()) {
        parent = std::make_shared<ov::opset1::Convert>(parent, convert.outPrecision);
    }
    if (!dequantization.empty()) {
        parent = makeDequantization(parent, dequantization);
    }
    return parent;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
