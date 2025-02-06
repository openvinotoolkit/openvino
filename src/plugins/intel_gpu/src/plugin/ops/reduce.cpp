// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/primitives/reduce.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"

namespace ov::intel_gpu {

static void CreateReduceOp(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op, cldnn::reduce_mode mode, bool keep_dims) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);
    auto input_pshape = op->get_input_partial_shape(0);
    int64_t rank = input_pshape.size();

    auto axes_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    OPENVINO_ASSERT(axes_constant != nullptr, "[GPU] Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");

    std::vector<int64_t> axes = axes_constant->cast_vector<int64_t>();
    for (size_t i = 0; i < axes.size(); i++) {
        if (axes[i] < 0)
            axes[i] += rank;

        if (axes[i] >= static_cast<int64_t>(rank) || axes[i] < 0)
            OPENVINO_THROW("[GPU] Unsupported axis value in ", op->get_friendly_name(), " (", axes[i], ")");
    }

    auto reducePrim = cldnn::reduce(layerName,
                                    inputs[0],
                                    mode,
                                    axes,
                                    keep_dims);

    p.add_primitive(*op, reducePrim);

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
        if (rank - axes.size() == 6)
            out_format = cldnn::format::bfwzyx;
        else if (rank - axes.size() == 5)
            out_format = cldnn::format::bfzyx;
        else if (rank - axes.size() <= 4)
            out_format = cldnn::format::bfyx;

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
