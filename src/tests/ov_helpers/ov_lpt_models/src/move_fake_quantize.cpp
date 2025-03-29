// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/move_fake_quantize.hpp"
#include <low_precision/relu.hpp>

#include "openvino/opsets/opset1.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "low_precision/network_helper.hpp"

#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/builders.hpp"

namespace ov {
namespace builder {
namespace subgraph {

using namespace ov::pass;

std::shared_ptr<ov::Model> MoveFakeQuantize::get(
    const ov::element::Type inputPrecision,
    const std::vector<ov::PartialShape>& inputShapes,
    const size_t concatInputsCount,
    const std::vector<FakeQuantizeOnDataWithConstant>& fqOnDataBefore,
    const DequantizationOperations::Convert& convertBefore,
    const DequantizationOperations& dequantizationBefore,
    const std::string& operation,
    const FakeQuantizeOnDataWithConstant& fqOnDataAfter,
    const DequantizationOperations::Convert& convertAfter,
    const DequantizationOperations& dequantizationAfter,
    const std::vector<ov::Any>& concatAttributes,
    const ov::element::Type precisionAfterOperation,
    const std::int64_t& axis,
    const bool oneInputWithSplit) {
    std::vector<std::shared_ptr<ov::opset1::Parameter>> inputs(oneInputWithSplit ? 1 : concatInputsCount);
    std::vector<ov::Output<ov::Node>> concatParents(concatInputsCount);
    if (oneInputWithSplit) {
        auto newInputShape = inputShapes[0];
        int channels = 0;
        bool channelsWasIdentified = false;
        for (const auto inputShape : inputShapes) {
            if (inputShape[axis].is_static()) {
                channels += inputShape[axis].get_length();
                channelsWasIdentified = true;
            }
        }

        if (channelsWasIdentified) {
            newInputShape[axis] = channels;
        }

        inputs[0] = std::make_shared<ov::opset1::Parameter>(inputPrecision, newInputShape);
        inputs[0]->set_friendly_name("input");

        const auto axis_constant =
            std::make_shared<ov::opset1::Constant>(ov::element::i32, Shape{}, std::vector<int64_t>({axis}));
        std::vector<int> split_lengths_values(inputShapes.size(), 1);
        split_lengths_values[split_lengths_values.size() - 1] = channels - (split_lengths_values.size() - 1);
        const auto split_lengths = std::make_shared<ov::opset1::Constant>(ov::element::i32,
                                                                          Shape{split_lengths_values.size()},
                                                                          split_lengths_values);
        const auto split = std::make_shared<ov::opset1::VariadicSplit>(inputs[0], axis_constant, split_lengths);
        for (size_t i = 0; i < concatInputsCount; i++) {
            // added unary op to avoid Split -> Concat pair elimination
            concatParents[i] = std::make_shared<ov::opset1::Sigmoid>(split->output(i));
        }
    } else {
        for (size_t i = 0; i < concatInputsCount; i++) {
            auto ind = 0;
            if (inputShapes.size() != 1) {
                ind = i;
            }
            inputs[i] = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShapes[ind]);
            inputs[i]->set_friendly_name(std::string("input") + "_" + std::to_string(i + 1));
            concatParents[i] = inputs[i];
        }
    }

    if (!fqOnDataBefore.empty()) {
        for (size_t i = 0; i < concatInputsCount; i++) {
            size_t ind = i;
            if (fqOnDataBefore.size() == 1) {
                ind = 0;
            }
            if (operation == "relu") {
                auto relu = std::make_shared<ov::opset1::Relu>(concatParents[i]);
                concatParents[i] = makeFakeQuantize(relu, inputPrecision, fqOnDataBefore[ind]);
            } else {
                concatParents[i] = makeFakeQuantize(concatParents[i], inputPrecision, fqOnDataBefore[ind]);
            }
            concatParents[i].get_node()->set_friendly_name(std::string("concat_fq") + "_" + std::to_string(i + 1));
            if (!convertBefore.empty()) {
                concatParents[i] = std::make_shared<ov::opset1::Convert>(concatParents[i], convertBefore.outPrecision);
            }
            if (!dequantizationBefore.empty()) {
                concatParents[i] = makeDequantization(concatParents[i], dequantizationBefore);
            }
        }
    }

    const auto concat = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector(concatParents.begin(), concatParents.end()),
        axis);
    concat->set_friendly_name("concat");
    std::shared_ptr<ov::Node> parent = concat;
    addAttributes({ parent }, concatAttributes);
    if (!fqOnDataAfter.empty()) {
        std::shared_ptr<ov::Node> fq;
        if (operation == "relu") {
            auto relu = std::make_shared<ov::opset1::Relu>(concat->output(0));
            fq = makeFakeQuantize(relu, inputPrecision, fqOnDataAfter);
        } else {
            fq = makeFakeQuantize(concat, inputPrecision, fqOnDataAfter);
        }
        fq->set_friendly_name("fakeQuantizeAfter");
        parent = fq;
        if (!convertAfter.empty()) {
            parent = std::make_shared<ov::opset1::Convert>(parent, convertAfter.outPrecision);
        }
        if (!dequantizationAfter.empty()) {
            parent = makeDequantization(parent, dequantizationAfter);
        }
    }
    parent->set_friendly_name("output");
    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(parent) };
    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector(inputs.begin(), inputs.end()),
        "MoveFakeQuantize");
    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
