// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/reduce.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"

namespace ov::intel_gpu {

static void CreateReduceOp(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op, cldnn::reduce_mode mode, bool keep_dims) {
    validate_inputs_count(op, {2});
    std::string layerName = layer_type_name_ID(op);
    auto input_pshape = op->get_input_partial_shape(0);
    int64_t rank = input_pshape.size();

    auto axes_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    OPENVINO_ASSERT(axes_constant != nullptr, "[GPU] Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");

    std::vector<int64_t> axes = axes_constant->cast_vector<int64_t>();
    for (size_t i = 0; i < axes.size(); i++) {
        if (axes[i] < 0) {
            axes[i] += rank;
        }

        if (axes[i] >= static_cast<int64_t>(rank) || axes[i] < 0) {
            OPENVINO_THROW("[GPU] Unsupported axis value in ", op->get_friendly_name(), " (", axes[i], ")");
        }
    }

    bool use_weighted_reduce = false;
    std::vector<cldnn::input_info> weighted_inputs;
    if (mode == cldnn::reduce_mode::sum && axes.size() == 1 && axes[0] == rank - 1) {
        auto src = op->get_input_node_shared_ptr(0);
        auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(src);
        if (multiply && multiply->output(0).get_target_inputs().size() == 1 && multiply->get_input_partial_shape(0).is_static() &&
            multiply->get_input_partial_shape(1).is_static()) {
            const auto input_type = multiply->get_input_element_type(0);
            const bool supported_type = input_type == multiply->get_input_element_type(1) && (input_type == ov::element::f16 || input_type == ov::element::f32);
            const auto shape0 = multiply->get_input_shape(0);
            const auto shape1 = multiply->get_input_shape(1);
            if (supported_type && shape0.size() == 4 && shape1.size() == 4 && shape0[0] == shape1[0] && shape0[2] == shape1[2] && shape0[2] > 1024 &&
                shape0[3] == shape1[3] && shape0[3] == 16) {
                size_t values_idx = 0;
                size_t weights_idx = 1;
                if (shape0[1] == 1 && shape1[1] > 1) {
                    values_idx = 1;
                    weights_idx = 0;
                }
                if ((values_idx == 0 ? shape1[1] : shape0[1]) == 1 && (values_idx == 0 ? shape0[1] : shape1[1]) > 1) {
                    auto multiply_inputs = p.GetInputInfo(multiply);
                    weighted_inputs = {multiply_inputs[values_idx], multiply_inputs[weights_idx]};
                    use_weighted_reduce = true;
                }
            }
        }
    }

    if (use_weighted_reduce) {
        p.add_primitive(*op, cldnn::reduce(layerName, weighted_inputs[0], weighted_inputs[1], mode, axes, keep_dims));
    } else {
        auto inputs = p.GetInputInfo(op);
        p.add_primitive(*op, cldnn::reduce(layerName, inputs[0], mode, axes, keep_dims));
    }

    if (input_pshape.is_dynamic() || p.use_new_shape_infer()) {
        return;
    }

    auto resultLayerName = layerName;
    auto out_dims = op->get_output_shape(0).size();
    if (out_dims == 3 && !keep_dims && rank >= 4) {
        resultLayerName = layerName + "_reshape";
        auto out_shape = op->get_output_shape(0);
        cldnn::tensor outTensor;
        switch (rank) {
            case 6:
                outTensor = cldnn::tensor(TensorValue(out_shape[0]), TensorValue(out_shape[1]),
                                          1, TensorValue(out_shape[2]), 1, 1);
            case 5:
                outTensor = cldnn::tensor(TensorValue(out_shape[0]), TensorValue(out_shape[1]),
                                          1, TensorValue(out_shape[2]), 1);
            case 4:
                outTensor = cldnn::tensor(TensorValue(out_shape[0]), TensorValue(out_shape[1]),
                                          1, TensorValue(out_shape[2]));
        }
        auto reshape_prim = cldnn::reshape(resultLayerName, cldnn::input_info(layerName), outTensor);
        p.add_primitive(*op, reshape_prim);
    }

    auto reorderLayerName = layerName + "_reorder";
    cldnn::format out_format = cldnn::format::any;
    auto out_dt = cldnn::element_type_to_data_type(op->get_output_element_type(0));
    if (!keep_dims && rank > 4) {
        if (rank - axes.size() == 6) {
            out_format = cldnn::format::bfwzyx;
        } else if (rank - axes.size() == 5) {
            out_format = cldnn::format::bfzyx;
        } else if (rank - axes.size() <= 4) {
            out_format = cldnn::format::bfyx;
        }

        auto reorder_prim = cldnn::reorder(reorderLayerName,
                                           cldnn::input_info(resultLayerName),
                                           out_format,
                                           out_dt);
        p.add_primitive(*op, reorder_prim);
    }
}

static void CreateReduceMaxOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::ReduceMax>& op) {
    CreateReduceOp(p, op, cldnn::reduce_mode::max, op->get_keep_dims());
}

static void CreateReduceLogicalAndOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::ReduceLogicalAnd>& op) {
    CreateReduceOp(p, op, cldnn::reduce_mode::logical_and, op->get_keep_dims());
}

static void CreateReduceLogicalOrOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::ReduceLogicalOr>& op) {
    CreateReduceOp(p, op, cldnn::reduce_mode::logical_or, op->get_keep_dims());
}

static void CreateReduceMeanOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::ReduceMean>& op) {
    CreateReduceOp(p, op, cldnn::reduce_mode::mean, op->get_keep_dims());
}

static void CreateReduceMinOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::ReduceMin>& op) {
    CreateReduceOp(p, op, cldnn::reduce_mode::min, op->get_keep_dims());
}

static void CreateReduceProdOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::ReduceProd>& op) {
    CreateReduceOp(p, op, cldnn::reduce_mode::prod, op->get_keep_dims());
}

static void CreateReduceSumOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::ReduceSum>& op) {
    CreateReduceOp(p, op, cldnn::reduce_mode::sum, op->get_keep_dims());
}

static void CreateReduceL1Op(ProgramBuilder& p, const std::shared_ptr<ov::op::v4::ReduceL1>& op) {
    CreateReduceOp(p, op, cldnn::reduce_mode::l1, op->get_keep_dims());
}

static void CreateReduceL2Op(ProgramBuilder& p, const std::shared_ptr<ov::op::v4::ReduceL2>& op) {
    CreateReduceOp(p, op, cldnn::reduce_mode::l2, op->get_keep_dims());
}

REGISTER_FACTORY_IMPL(v1, ReduceMax);
REGISTER_FACTORY_IMPL(v1, ReduceLogicalAnd);
REGISTER_FACTORY_IMPL(v1, ReduceLogicalOr);
REGISTER_FACTORY_IMPL(v1, ReduceMean);
REGISTER_FACTORY_IMPL(v1, ReduceMin);
REGISTER_FACTORY_IMPL(v1, ReduceProd);
REGISTER_FACTORY_IMPL(v1, ReduceSum);
REGISTER_FACTORY_IMPL(v4, ReduceL1);
REGISTER_FACTORY_IMPL(v4, ReduceL2);

}  // namespace ov::intel_gpu
