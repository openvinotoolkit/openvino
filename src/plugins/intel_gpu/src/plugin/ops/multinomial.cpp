// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/multinomial.hpp"

#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/broadcast.hpp"
#include "intel_gpu/primitives/cum_sum.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/multinomial.hpp"
#include "intel_gpu/primitives/random_uniform.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/shape_of.hpp"
#include "intel_gpu/primitives/slice.hpp"

#include <memory>
#include <cstring>

namespace ov::intel_gpu {
namespace {

template<typename T>
cldnn::data CreateScalarDataPrimitive(ProgramBuilder& p, const cldnn::primitive_id& name, T value) {
    auto mem = p.get_engine().allocate_memory(
        cldnn::layout{element::from<T>(), cldnn::format::bfyx, {1, 1, 1, 1}}, false);
    cldnn::mem_lock<int8_t> host_mem{mem, p.get_engine().get_service_stream()};
    std::memcpy(host_mem.data(), &value, sizeof value);
    return {name, mem};
}

cldnn::data CreateShapeDataPrimitive(ProgramBuilder& p, const cldnn::primitive_id& name, Shape& value) {
    auto mem = p.get_engine().allocate_memory(
        cldnn::layout{element::Type_t::i64, cldnn::format::bfyx, {1, 1, 1, static_cast<int>(value.size())}}, false);
    cldnn::mem_lock<int8_t> host_mem{mem, p.get_engine().get_service_stream()};
    std::vector<std::int64_t> shape {};
    std::copy(value.begin(), value.end(), std::back_inserter(shape));
    std::memcpy(host_mem.data(), shape.data(), sizeof(std::int64_t) * shape.size());
    return {name, mem};
}

static void CreateMultinomialOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v13::Multinomial>& op) {
    validate_inputs_count(op, { 2 });
    auto inputs = p.GetInputInfo(op);
    auto input_max_shape = op->get_input_partial_shape(0).get_max_shape();

    const std::int64_t last_axis = input_max_shape.size() - 1;
    cldnn::primitive_id cumsum {layer_type_name_ID(op) + "_cumsum"};
    if (op->get_log_probs()) {
        cldnn::primitive_id exp {layer_type_name_ID(op) + "_exp"};
        cldnn::activation exp_prim {exp, inputs[0], cldnn::activation_func::exp};
        p.add_primitive(*op, exp_prim);
        cldnn::cum_sum cumsum_primitive{cumsum, exp, last_axis};
        p.add_primitive(*op, cumsum_primitive);
    } else {
        cldnn::cum_sum cumsum_primitive{cumsum, inputs[0], last_axis};
        p.add_primitive(*op, cumsum_primitive);
    }

    const cldnn::primitive_id slice_start {layer_type_name_ID(op) + "_slice_start"};
    auto slice_start_prim = CreateScalarDataPrimitive(p, slice_start, static_cast<std::int64_t>(-1));
    p.add_primitive(*op, slice_start_prim);

    const cldnn::primitive_id slice_stop {layer_type_name_ID(op) + "_slice_stop"};
    auto slice_stop_prim = CreateScalarDataPrimitive(p, slice_stop, static_cast<std::int64_t>(-2));
    p.add_primitive(*op, slice_stop_prim);

    const cldnn::primitive_id slice_step {layer_type_name_ID(op) + "_slice_step"};
    auto slice_step_prim = CreateScalarDataPrimitive(p, slice_step, static_cast<std::int64_t>(-1));
    p.add_primitive(*op, slice_step_prim);

    const cldnn::primitive_id slice_axis {layer_type_name_ID(op) + "_slice_axis"};
    auto slice_axis_prim = CreateScalarDataPrimitive(p, slice_axis, last_axis);
    p.add_primitive(*op, slice_axis_prim);

    const cldnn::primitive_id slice {layer_type_name_ID(op) + "_slice"};
    cldnn::slice slice_prim {
        slice,
        { cumsum,
            slice_start,
            slice_stop,
            slice_step,
            slice_axis
        }
    };
    p.add_primitive(*op, slice_prim);

    const cldnn::primitive_id cdf {layer_type_name_ID(op) + "_cdf"};
    cldnn::eltwise divide_prim{cdf, cumsum, slice, cldnn::eltwise_mode::div};
    p.add_primitive(*op, divide_prim);

    const cldnn::primitive_id input0_shape_of {layer_type_name_ID(op) + "_input0_shape_of"};
    cldnn::shape_of input0_shape_of_prim{input0_shape_of, inputs[0], cldnn::data_types::i32};
    p.add_primitive(*op, input0_shape_of_prim);

    const cldnn::primitive_id cdf_reshaped {layer_type_name_ID(op) + "_cdf_reshaped"};
    cldnn::reshape cdf_reshaped_prim {cdf_reshaped, cdf, input0_shape_of, false, op->get_input_partial_shape(0)};
    p.add_primitive(*op, cdf_reshaped_prim);

    const cldnn::primitive_id random {layer_type_name_ID(op) + "_random"};
    const cldnn::primitive_id random_shape {layer_type_name_ID(op) + "_random_shape"};
    auto random_shape_prim = CreateShapeDataPrimitive(p, random_shape, input_max_shape);
    p.add_primitive(*op, random_shape_prim);
    const cldnn::primitive_id random_minval {layer_type_name_ID(op) + "_random_minval"};
    auto random_minval_prim = CreateScalarDataPrimitive(p, random_minval, 0.f);
    p.add_primitive(*op, random_minval_prim);
    const cldnn::primitive_id random_maxval {layer_type_name_ID(op) + "_random_maxval"};
    auto random_maxval_prim = CreateScalarDataPrimitive(p, random_maxval, 1.f);
    p.add_primitive(*op, random_maxval_prim);
    cldnn::random_uniform random_prim {
        random,
        {random_shape, random_minval, random_maxval},
        op->get_input_element_type(0),
        op->get_global_seed(),
        op->get_op_seed(),
        input_max_shape
    };
    p.add_primitive(*op, random_prim);

    auto const_num_samples = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    OPENVINO_ASSERT(const_num_samples != nullptr, "[GPU] Unsupported num_samples node type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");

    std::int64_t num_samples{};
    if (const_num_samples->get_output_element_type(0) == ov::element::Type_t::i32)
        num_samples = const_num_samples->cast_vector<std::int32_t>(1)[0];
    else
        num_samples = const_num_samples->cast_vector<std::int64_t>(1)[0];

    cldnn::multinomial multinomial_prim {
        layer_type_name_ID(op),
        cdf_reshaped,
        random,
        op->get_convert_type(),
        op->get_with_replacement(),
        op->get_log_probs(),
        op->get_global_seed(),
        op->get_op_seed(),
        num_samples
    };
    p.add_primitive(*op, multinomial_prim);
}
} // namespace

REGISTER_FACTORY_IMPL(v13, Multinomial);

}  // namespace ov::intel_gpu
