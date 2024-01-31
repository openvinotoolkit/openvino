// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/rope.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/rope.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/slice.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/strided_slice.hpp"

namespace ov {
namespace op {
namespace internal {
using RoPE = ov::intel_gpu::op::RoPE;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

namespace {

template<typename T>
cldnn::data CreateScalarDataPrimitive(ProgramBuilder& p, const cldnn::primitive_id& name, T value) {
    auto mem = p.get_engine().allocate_memory(
        cldnn::layout{element::from<T>(), cldnn::format::bfyx, {1, 1, 1, 1}}, false);
    cldnn::mem_lock<int8_t> host_mem{mem, p.get_engine().get_service_stream()};
    std::memcpy(host_mem.data(), &value, sizeof value);
    return {name, mem};
}

cldnn::data CreateDataPrimitive(ProgramBuilder& p, const cldnn::primitive_id& name, const std::vector<int64_t>& value) {
    auto mem = p.get_engine().allocate_memory({{ 3 }, cldnn::data_types::i64, cldnn::format::bfyx });
    cldnn::mem_lock<int64_t> host_mem(mem, p.get_engine().get_service_stream());

    auto it = host_mem.begin();
    for (auto x : value)
        *it++ = x;

    return {name, mem};
}

static void CreateRoPEOp(ProgramBuilder& p, const std::shared_ptr<op::RoPE>& op) {
    validate_inputs_count(op, {3, 4});
    auto inputs = p.GetInputInfo(op);
    const auto& config = op->get_config();

    // if (config.slice_stop - config.slice_start > 0) {
    //     const cldnn::primitive_id slice_start {layer_type_name_ID(op) + "_slice_start"};
    //     auto slice_start_prim = CreateScalarDataPrimitive(p, slice_start, static_cast<std::int64_t>(config.slice_start));
    //     p.add_primitive(*op, slice_start_prim);

    //     const cldnn::primitive_id slice_stop {layer_type_name_ID(op) + "_slice_stop"};
    //     auto slice_stop_prim = CreateScalarDataPrimitive(p, slice_stop, static_cast<std::int64_t>(config.slice_stop));
    //     p.add_primitive(*op, slice_stop_prim);

    //     const cldnn::primitive_id slice_step {layer_type_name_ID(op) + "_slice_step"};
    //     auto slice_step_prim = CreateScalarDataPrimitive(p, slice_step, static_cast<std::int64_t>(1));
    //     p.add_primitive(*op, slice_step_prim);

    //     const cldnn::primitive_id slice_axis {layer_type_name_ID(op) + "_slice_axis"};
    //     auto slice_axis_prim = CreateScalarDataPrimitive(p, slice_axis, static_cast<std::int64_t>(2));
    //     p.add_primitive(*op, slice_axis_prim);


    //     auto sliceName = op->get_friendly_name() + "_slice";
    //     auto slicePrim = cldnn::slice(sliceName,
    //                                   { cldnn::input_info(inputs[0].pid),
    //                                     slice_start,
    //                                     slice_stop,
    //                                     slice_step,
    //                                     slice_axis });
    //     p.add_primitive(*op, slicePrim);
    //     inputs[0] = cldnn::input_info(sliceName);
    // }

    if (config.slice_stop - config.slice_start > 0) {
        const cldnn::primitive_id begin {layer_type_name_ID(op) + "_slice_begin"};
        auto slice_start_prim = CreateDataPrimitive(p, begin, { 0, 0, static_cast<std::int64_t>(config.slice_start) });
        p.add_primitive(*op, slice_start_prim);

        const cldnn::primitive_id end {layer_type_name_ID(op) + "_slice_end"};
        auto slice_stop_prim = CreateDataPrimitive(p, end, { 1000000, 1000000, static_cast<std::int64_t>(config.slice_stop) });
        p.add_primitive(*op, slice_stop_prim);

        const cldnn::primitive_id strides {layer_type_name_ID(op) + "_slice_strides"};
        auto slice_step_prim = CreateDataPrimitive(p, strides, { 1, 1, 1 });
        p.add_primitive(*op, slice_step_prim);


        auto strided_slice_name = op->get_friendly_name() + "_strided_slice";
        auto strided_slice_prim = cldnn::strided_slice(strided_slice_name,
                                                       cldnn::input_info(inputs[0].pid),
                                                       begin,
                                                       end,
                                                       strides,
                                                       {}, {}, {}, {}, {}, {});
        p.add_primitive(*op, strided_slice_prim);
        inputs[0] = cldnn::input_info(strided_slice_name);
    }

    if (config.input_trans0213) {
        auto& input_pshape = op->get_input_partial_shape(0);
        std::vector<uint16_t> transposeOrder(input_pshape.size());
        std::iota(transposeOrder.begin(), transposeOrder.end(), 0);
        std::swap(*(transposeOrder.begin() + 1), *(transposeOrder.begin() + 2));

        auto permuteName = op->get_friendly_name() + "_trans0213";
        auto permutePrim = cldnn::permute(permuteName,
                                          cldnn::input_info(inputs[0].pid),
                                          transposeOrder);
        p.add_primitive(*op, permutePrim);
        inputs[0] = cldnn::input_info(permuteName);
    }

    // if (config.is_interleaved) {
        // add transpose afer RoPE
    // }

    auto rope = cldnn::rope(layer_type_name_ID(op),
                            inputs,
                            config);

    p.add_primitive(*op, rope);
}
} // namespace


REGISTER_FACTORY_IMPL(internal, RoPE);

}  // namespace intel_gpu
}  // namespace ov
