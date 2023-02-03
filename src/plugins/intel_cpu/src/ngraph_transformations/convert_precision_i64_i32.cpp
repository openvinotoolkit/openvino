// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "convert_precision_i64_i32.hpp"
#include <openvino/opsets/opset10.hpp>
#include "transformations/utils/utils.hpp"
#include "cpu_types.h"

#include <unordered_set>

bool isNativelySupported(const ov::Node::type_info_t &type) {
    static const std::unordered_set<ov::Node::type_info_t> i64Ops = {
        ov::opset10::Add::get_type_info_static(),
        ov::op::v1::Broadcast::get_type_info_static(),
        ov::opset10::Broadcast::get_type_info_static(),
        ov::opset10::Constant::get_type_info_static(),
        ov::opset10::Convert::get_type_info_static(),
        ov::opset10::Divide::get_type_info_static(),
        ov::opset10::Equal::get_type_info_static(),
        ov::opset10::FloorMod::get_type_info_static(),
        ov::op::v7::Gather::get_type_info_static(),
        ov::opset10::Gather::get_type_info_static(),
        ov::opset10::Greater::get_type_info_static(),
        ov::opset10::Less::get_type_info_static(),
        ov::opset10::Multiply::get_type_info_static(),
        ov::opset10::NonMaxSuppression::get_type_info_static(),
        ov::opset10::OneHot::get_type_info_static(),
        ov::opset10::Parameter::get_type_info_static(),
        ov::opset10::ReduceL1::get_type_info_static(),
        ov::opset10::ReduceL2::get_type_info_static(),
        ov::opset10::ReduceLogicalAnd::get_type_info_static(),
        ov::opset10::ReduceMax::get_type_info_static(),
        ov::opset10::ReduceMean::get_type_info_static(),
        ov::opset10::ReduceMin::get_type_info_static(),
        ov::opset10::ReduceProd::get_type_info_static(),
        ov::opset10::ReduceSum::get_type_info_static(),
        ov::opset10::Reshape::get_type_info_static(),
        ov::opset10::Result::get_type_info_static(),
        ov::opset10::ScatterElementsUpdate::get_type_info_static(),
        ov::opset10::ScatterNDUpdate::get_type_info_static(),
        ov::opset10::ScatterUpdate::get_type_info_static(),
        ov::opset10::Select::get_type_info_static(),
        ov::opset10::ShapeOf::get_type_info_static(),
        ov::opset10::SquaredDifference::get_type_info_static(),
        ov::opset10::Squeeze::get_type_info_static(),
        ov::opset10::Subtract::get_type_info_static(),
        ov::opset10::Tile::get_type_info_static(),
        ov::opset10::Transpose::get_type_info_static(),
        ov::opset10::Unique::get_type_info_static(),
        ov::opset10::Unsqueeze::get_type_info_static()
    };

    return i64Ops.find(type) != i64Ops.end();
}

std::shared_ptr<ov::Node> changeConstantPrecision(std::shared_ptr<ov::opset10::Constant>& constant) {
    const auto* srcData = constant->get_data_ptr<int64_t>();
    const auto size = shape_size(constant->get_shape());

    auto newConstant = std::make_shared<ov::opset10::Constant>(ov::element::i32, constant->get_shape());
    newConstant->output(0).set_names(constant->output(0).get_names());
    auto* dstData = const_cast<int32_t*>(reinterpret_cast<const int32_t*>(newConstant->get_data_ptr()));
    if (dstData == nullptr) {
        throw ngraph::ngraph_error("Can't get destination data pointer");
    }

    for (size_t i = 0; i < size; ++i) {
        if (srcData[i] >= std::numeric_limits<int32_t>::max()) {
            dstData[i] = std::numeric_limits<int32_t>::max();
        } else if (srcData[i] <= std::numeric_limits<int32_t>::lowest()) {
            dstData[i] = std::numeric_limits<int32_t>::lowest();
        } else {
            dstData[i] = static_cast<int32_t>(srcData[i]);
        }
    }
    return newConstant;
}

bool ov::intel_cpu::ConvertPrecisionI64ToI32::run_on_model(const std::shared_ptr<ov::Model> &model) {
    const auto orderedOps = model->get_ordered_ops();
    for (const auto& op : orderedOps) {
        if (isNativelySupported(op->get_type_info()) || TypeFromName(op->get_type_name()) == Type::Unknown) {
            continue;
        }

        bool convertForOutputsRequired = false;
        for (const auto& input : op->inputs()) {
            if (input.get_element_type() == ov::element::i64) {
                auto parentOutput = input.get_source_output();
                auto parentNode = parentOutput.get_node_shared_ptr();
                if (ov::is_type<ov::opset10::Convert>(parentNode) &&
                        parentNode->get_rt_info().find("convert_i32_i64") != parentNode->get_rt_info().end()) {
                    input.replace_source_output(parentNode->input_value(0));
                } else if (auto constOp = ov::as_type_ptr<ov::opset10::Constant>(parentNode)) {
                    auto newConst = changeConstantPrecision(constOp);
                    input.replace_source_output(newConst);
                    newConst->set_friendly_name(constOp->get_friendly_name());
                } else {
                    auto convert = std::make_shared<ov::opset10::Convert>(input.get_source_output(), ov::element::i32);
                    convert->output(0).add_names(parentOutput.get_names());
                    input.replace_source_output(convert);
                }
                convertForOutputsRequired = true;
            }
        }

        if (convertForOutputsRequired) {
            // Propagate i32 precision into outputs.
            op->validate_and_infer_types();
            for (auto& output : op->outputs()) {
                if (output.get_element_type() == ov::element::i32) {
                    auto targetInputs = output.get_target_inputs();
                    auto convert = std::make_shared<ov::opset10::Convert>(output, ov::element::i64);

                    auto& rt_info = convert->get_rt_info();
                    rt_info["convert_i32_i64"] = "";
                    for (const auto& targetInput : targetInputs) {
                        targetInput.replace_source_output(convert);
                    }

                    auto& convertTensor = convert->output(0).get_tensor();
                    const std::string newName = ov::op::util::get_ie_output_name(output);
                    if (ov::descriptor::get_ov_tensor_legacy_name(convertTensor).empty()) {
                        ov::descriptor::set_ov_tensor_legacy_name(convertTensor, newName);
                    }
                    if (!output.get_names().empty()) {
                        convertTensor.set_names(output.get_names());
                    }
                }
            }
        }

        if (auto multisubgraph_op = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(op)) {
            for (size_t idx = 0; idx < multisubgraph_op->get_internal_subgraphs_size(); ++idx) {
                run_on_model(multisubgraph_op->get_function(static_cast<int>(idx)));
            }
        }
    }

    return true;
}
