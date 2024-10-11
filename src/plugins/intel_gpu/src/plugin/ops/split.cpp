// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"

#include "intel_gpu/primitives/crop.hpp"

namespace ov {
namespace intel_gpu {

static bool IsDynamic(const std::shared_ptr<ov::Node>& op) {
    if (op->is_dynamic()) {
        return true;
    }

    for (size_t i = 0; i < op->get_output_size(); i++) {
        const auto outPartialShape = op->get_output_partial_shape(i);
        if (outPartialShape.is_dynamic()) {
            return true;
        }
    }

    return false;
}

static void CreateCommonSplitOp(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op) {
    auto get_layer_name = [&](size_t idx)->std::string {
        return layer_type_name_ID(op) + ((op->get_output_size() == 1)? "" : ".out" + std::to_string(idx));
    };

    auto inputs = p.GetInputInfo(op);
    if (p.use_new_shape_infer() || op->is_dynamic()) {
        std::vector<cldnn::tensor> offsets;

        // op->is_dynamic() does not check if output shape is dynamic. it only check dynamism for input shapes
        // Even if op->is_dynamic() is false, output shape can be dynamic.
        // Thus, it is necessary to check if output shape is dynamic.
        if (!IsDynamic(op)) {
            auto input_pshape = op->get_input_partial_shape(0);
            ov::Shape start_offset(input_pshape.size());
            for (size_t i = 0; i < op->get_output_size(); i++) {
                const auto outPartialShape = op->get_output_partial_shape(i);

                auto offsetTensor = tensor_from_dims(start_offset, 0);
                offsets.push_back(offsetTensor);

                for (size_t idx = 0; idx < input_pshape.size(); idx++) {
                    if (outPartialShape[idx] != input_pshape[idx]) {
                        start_offset[idx] += outPartialShape.to_shape()[idx];
                    }
                }
            }
        }

        int64_t axis = -1;
        auto const_axis = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
        if (const_axis) {
            axis = ov::util::try_normalize_axis(const_axis->cast_vector<int64_t>()[0],
                                                op->get_input_partial_shape(0).rank(),
                                                *op);
        }
        cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
        auto num_splits = static_cast<size_t>(1);
        if (ov::is_type<ov::op::v1::Split>(op)) {
            num_splits = ov::as_type_ptr<ov::op::v1::Split>(op)->get_num_splits();
            op_mode = cldnn::crop_ngraph_op_mode::split;
        }

        for (size_t i = 0; i < op->get_output_size(); i++) {
            const auto& users = op->get_output_target_inputs(i);
            // don't add crop primitive if port is not used by anyone
            if (users.size() == 0)
                continue;
            auto cropPrim = cldnn::crop(get_layer_name(i),
                                        inputs,
                                        cldnn::tensor(1),
                                        (offsets.empty() ? cldnn::tensor(0) : offsets[i]),
                                        op_mode,
                                        static_cast<int>(i),
                                        axis,
                                        num_splits);
            p.add_primitive(*op, cropPrim);
        }
    } else {
        auto input_pshape = op->get_input_partial_shape(0);
        ov::Shape start_offset(input_pshape.size());
        for (size_t i = 0; i < op->get_output_size(); i++) {
            const auto outPartialShape = op->get_output_partial_shape(i);
            if (outPartialShape.size() != start_offset.size()) {
                OPENVINO_THROW("Invalid dimesions in split layer: ", op->get_friendly_name(),
                               " output: ", op->get_output_tensor(i).get_any_name());
            }
            for (size_t idx = 0; idx < input_pshape.size(); idx++) {
                if ((outPartialShape[idx].get_length() + static_cast<ov::Dimension::value_type>(start_offset[idx])) > input_pshape[idx].get_length()) {
                    OPENVINO_THROW("Invalid dimesions in split layer: ", op->get_friendly_name(),
                                   " output: ", op->get_output_tensor(idx).get_any_name());
                }
            }

            auto offsetTensor = tensor_from_dims(start_offset, 0);
            auto outTensor = tensor_from_dims(op->get_output_shape(i), 1);
            auto cropPrim = cldnn::crop(get_layer_name(i), inputs[0], outTensor, offsetTensor);
            p.add_primitive(*op, cropPrim);

            for (size_t idx = 0; idx < input_pshape.size(); idx++) {
                if (outPartialShape[idx] != input_pshape[idx]) {
                    start_offset[idx] += outPartialShape.to_shape()[idx];
                }
            }
        }
    }
}

static void CreateSplitOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Split>& op) {
    validate_inputs_count(op, {2});
    CreateCommonSplitOp(p, op);
}

static void CreateVariadicSplitOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::VariadicSplit>& op) {
    validate_inputs_count(op, {3});
    CreateCommonSplitOp(p, op);
}

REGISTER_FACTORY_IMPL(v1, Split);
REGISTER_FACTORY_IMPL(v1, VariadicSplit);

}  // namespace intel_gpu
}  // namespace ov
