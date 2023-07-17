// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "convert_precision_i64_i32.hpp"
#include <openvino/opsets/opset12.hpp>
#include "transformations/utils/utils.hpp"
#include "cpu_types.h"

#include <unordered_set>

namespace ov {
namespace intel_cpu {

// Returns list of operations that support i64.
bool ConvertPrecisionI64ToI32::isNativelySupported(const ov::Node::type_info_t& type) const {
    static const std::unordered_set<ov::Node::type_info_t> i64Ops = {
        opset12::Add::get_type_info_static(),
        opset12::Ceiling::get_type_info_static(),
        opset12::Constant::get_type_info_static(),
        opset12::Divide::get_type_info_static(),
        opset12::Equal::get_type_info_static(),
        opset12::Floor::get_type_info_static(),
        opset12::FloorMod::get_type_info_static(),
        opset12::Greater::get_type_info_static(),
        opset12::Less::get_type_info_static(),
        opset12::Maximum::get_type_info_static(),
        opset12::Minimum::get_type_info_static(),
        opset12::Multiply::get_type_info_static(),
        opset12::Parameter::get_type_info_static(),
        opset12::Result::get_type_info_static(),
        opset12::Sqrt::get_type_info_static(),
        opset12::SquaredDifference::get_type_info_static(),
        opset12::Subtract::get_type_info_static(),
        opset12::Transpose::get_type_info_static()
    };

    return i64Ops.find(type) != i64Ops.end();
}

std::shared_ptr<ov::Node> ConvertPrecisionI64ToI32::changeConstantPrecision(std::shared_ptr<op::v0::Constant>& constant) const {
    const auto* srcData = constant->get_data_ptr<int64_t>();
    const auto size = shape_size(constant->get_shape());

    auto newConstant = std::make_shared<op::v0::Constant>(element::i32, constant->get_shape());
    newConstant->output(0).set_names(constant->output(0).get_names());
    auto* dstData = const_cast<int32_t *>(reinterpret_cast<const int32_t *>(newConstant->get_data_ptr()));
    OPENVINO_ASSERT(dstData != nullptr, "Can't get destination data pointer");

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

bool ConvertPrecisionI64ToI32::run_on_model(const std::shared_ptr<ov::Model>& model) {
    const auto orderedOps = model->get_ordered_ops();
    for (const auto& op : orderedOps) {
        if (isNativelySupported(op->get_type_info()) || TypeFromName(op->get_type_name()) == Type::Unknown) {
            continue;
        }

        bool convertForOutputsRequired = false;
        for (const auto& input : op->inputs()) {
            if (input.get_element_type() == element::i64) {
                auto parentOutput = input.get_source_output();
                auto parentNode = parentOutput.get_node_shared_ptr();
                if (is_type<opset12::Convert>(parentNode) &&
                        parentNode->get_input_element_type(0) == element::i32 &&
                        parentNode->get_output_element_type(0) == element::i64) {
                    input.replace_source_output(parentNode->input_value(0));
                } else if (is_type<opset12::Convert>(op) &&
                        op->get_input_element_type(0) == element::i64 &&
                        op->get_output_element_type(0) == element::i32) {
                    continue;
                } else if (auto constOp = as_type_ptr<op::v0::Constant>(parentNode)) {
                    auto newConst = changeConstantPrecision(constOp);
                    input.replace_source_output(newConst);
                    newConst->set_friendly_name(constOp->get_friendly_name());
                } else {
                    auto convert = std::make_shared<opset12::Convert>(input.get_source_output(), element::i32);
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
                if (output.get_element_type() == element::i32) {
                    auto convert = std::make_shared<opset12::Convert>(output, element::i64);
                    replace_output_update_name(output, convert->output(0));
                }
            }
        }

        if (auto multisubgraph_op = as_type_ptr<op::util::MultiSubGraphOp>(op)) {
            for (size_t idx = 0; idx < multisubgraph_op->get_internal_subgraphs_size(); ++idx) {
                run_on_model(multisubgraph_op->get_function(static_cast<int>(idx)));
            }
        }
    }

    return true;
}

} // namespace intel_cpu
} // namespace ov
