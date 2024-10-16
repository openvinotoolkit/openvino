// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/gemm.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "multi_stage_primitive.hpp"

#include "kv_cache_inst.h"
#include "gemm_inst.h"
#include "gemm/gemm_kernel_base.h"
#include "gemm/gemm_kernel_selector.h"

namespace cldnn {
namespace ocl {

// Gemm impl may create 2 versions of the kernel internally
// 1. default kernel
// 2. kernel with indirect access to one of the inputs
// This feature is used to avoid perf drop when we create single kernel which checks batch size in runtime
// Can be reverted once performance of the kernel is improved
struct gemm_impl : multi_stage_primitive<gemm> {
    using parent = multi_stage_primitive<gemm>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::gemm_kernel_selector;
    using kernel_params_t = kernel_selector::gemm_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::gemm_impl)

    const uint32_t default_gemm = 0;
    const uint32_t indirect_gemm = 1;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gemm_impl>(*this);
    }

    gemm_impl() = default;

    gemm_impl(const std::vector<kernel_selector::kernel_data>& kd) : parent(kd) {
        this->can_reuse_memory = true;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernels_data[default_gemm].kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[default_gemm]);
            if (_kernels_data.size() == 2) {
                auto bt_kernel_impl = kernel_selector.GetImplementation(_kernels_data[indirect_gemm].kernelName);
                bt_kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[indirect_gemm]);
            }
        }
    }

protected:
    static size_t get_beam_table_id(std::shared_ptr<const gemm> primitive) {
        return primitive->input_size() == 3 ? 3 : 2;
    }

    kernel_arguments_data get_arguments(const gemm_inst& instance, size_t stage) const override {
        kernel_arguments_data args;

        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }

        if (instance.has_fused_primitives()) {
            size_t count = instance.get_fused_mem_count();
            for (size_t i = 0; i < count; i++) {
                args.fused_op_inputs.push_back(instance.fused_memory(i));
            }
        }

        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            args.outputs.push_back(instance.output_memory_ptr(i));
        }

        args.shape_info = instance.shape_info_memory_ptr();

        const auto& desc = instance.get_typed_desc<gemm>();
        if (stage == indirect_gemm) {
            args.inputs.push_back(instance.dep_memory_ptr(get_beam_table_id(desc)));
        }

        return args;
    }

    void set_arguments_impl(gemm_inst& instance) override {}

    event::ptr execute_stage(const std::vector<event::ptr>& events, gemm_inst& instance, size_t stage) {
        stream& stream = instance.get_network().get_stream();
        std::vector<event::ptr> tmp_events(events);
        std::vector<event::ptr> all_events;
        size_t kernel_offset = 0;
        for (size_t s = 0; s < stage; s++) {
            kernel_offset += _kernels_data[s].kernels.size();
        }
        for (size_t kd_idx = 0; kd_idx < _kernels_data[stage].kernels.size(); ++kd_idx) {
            if (_kernels_data[stage].kernels[kd_idx].skip_execution)
                continue;

            size_t idx_final = kernel_offset + kd_idx;
            // If any user of the prim's users is CPU implementation or network's output, set prim as a output event (event won't be nullptr)
            bool needs_completion_event = instance.needs_completion_event();

            auto& params = _kernels_data[stage].kernels[kd_idx].params;
            auto args = get_arguments(instance, stage);
            args.scalars = &params.scalars;

            for (const auto& m : instance.get_intermediates_memories()) {
                args.intermediates.push_back(m);
            }

            stream.set_arguments(*_kernels[idx_final], _kernels_data[stage].kernels[kd_idx].params, args);

            const auto& gws = params.workGroups.global;
            const auto& lws = params.workGroups.local;

            GPU_DEBUG_TRACE_DETAIL << "Enqueue stage " << stage << " kernel " << idx_final << ": gws=[" << gws[0] << ", " << gws[1] << ", " << gws[2] << "] "
                                   << "lws=[" << lws[0] << ", " << lws[1] << ", " << lws[2] << "]"
                                   << (needs_completion_event ? " has_completion_event=true" : "") << std::endl;

            auto ev = stream.enqueue_kernel(*_kernels[idx_final], params, args, tmp_events, needs_completion_event);
            if (_kernels_data[stage].needs_sub_kernels_sync) {
                tmp_events = {ev};
            }
            all_events.push_back(ev);
        }

        return stream.aggregate_events(all_events, all_events.size() > 1);
    }

    bool need_indirect_load(const gemm_inst& inst) const {
        auto desc = inst.get_typed_desc<gemm>();
        if (!desc->indirect_a && !desc->indirect_b)
            return false;

        const auto& params = *inst.get_impl_params();
        const auto indirect_axis = desc->indirect_axis;
        if (params.input_layouts[get_beam_table_id(desc)].get_partial_shape()[indirect_axis].get_length() == 1)
            return false;

        const auto& deps = inst.dependencies();

        const auto& indirect_dep = deps[desc->indirect_a ? 0 : 1].first;
        if (dynamic_cast<const kv_cache_inst*>(indirect_dep) == nullptr)
            return true;

        auto state_layout = indirect_dep->get_impl_params()->get_input_layout(0);
        bool is_prefill = state_layout.count() == 0;
        return !is_prefill;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, gemm_inst& instance) override {
        if (instance.get_input_layout(0).count() == 0 ||
            instance.get_input_layout(1).count() == 0) {
            stream& stream = instance.get_network().get_stream();
            stream.enqueue_barrier();
            return instance.output_memory_ptr()->fill(stream, false);
        }

        if (need_indirect_load(instance))
            return execute_stage(events, instance, indirect_gemm);
        else
            return execute_stage(events, instance, default_gemm);
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false, bool indirect = false) {
        const auto& primitive = impl_param.typed_desc<gemm>();

        auto params = get_default_params<kernel_selector::gemm_params>(impl_param, is_shape_agnostic);

        for (size_t i = 1; i < primitive->input_size(); ++i) {
            params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[i]));
        }

        params.stage_id = indirect ? 1 : 0;

        params.alpha = primitive->alpha;
        params.beta = primitive->beta;
        params.transpose_input0 = primitive->transpose_input0;
        params.transpose_input1 = primitive->transpose_input1;
        params.input0_order = primitive->input0_transpose_order;
        params.input1_order = primitive->input1_transpose_order;
        params.output_order = primitive->output_transpose_order;

        auto input0_pshape = impl_param.input_layouts[0].get_partial_shape();
        auto input1_pshape = impl_param.input_layouts[1].get_partial_shape();
        const auto is_broadcastable = input0_pshape.rank().is_static() &&
                                      input1_pshape.rank().is_static() &&
                                      input0_pshape.size() > 1 &&
                                      input1_pshape.size() > 1 &&
                                      (primitive->input_rank == primitive->weight_rank);
        if (is_broadcastable) {
            auto transpose_pshape = [](const ov::PartialShape pshape, const std::vector<int64_t>& order) {
                if (order.size() < pshape.size()) {
                    auto transposed_pshape = pshape;
                    auto rank_diff = pshape.size() - order.size();
                    for (size_t i = 0; i < order.size(); i++) {
                        transposed_pshape[i + rank_diff] = pshape[rank_diff + order[i]];
                    }
                    return transposed_pshape;
                } else {
                    auto transposed_pshape = ov::PartialShape::dynamic(pshape.rank());
                    for (size_t i = 0; i < order.size(); i++) {
                        transposed_pshape[i] = pshape[order[i]];
                    }
                    return transposed_pshape;
                }
            };
            size_t max_rank = input0_pshape.size();
            auto default_order = ov::intel_gpu::op::Gemm::default_order(max_rank);
            auto input0_trans_pshape = (primitive->input0_transpose_order != default_order) ?
                                       transpose_pshape(input0_pshape, primitive->input0_transpose_order) :
                                       input0_pshape;
            auto input1_trans_pshape = (primitive->input1_transpose_order != default_order) ?
                                       transpose_pshape(input1_pshape, primitive->input1_transpose_order) :
                                       input1_pshape;
            for (size_t i = 0; i < max_rank - 2; ++i) {
                if (input0_trans_pshape[i].is_static() && input1_trans_pshape[i].is_static()) {
                    if (input1_trans_pshape[i].get_length() > input0_trans_pshape[i].get_length()) {
                        params.input0_reshape_axes = primitive->input0_transpose_order[i];
                        params.input0_broadcast_val = input1_trans_pshape[i].get_length() / input0_trans_pshape[i].get_length();
                    } else if (input0_trans_pshape[i].get_length() > input1_trans_pshape[i].get_length()) {
                        params.input1_reshape_axes = primitive->input1_transpose_order[i];
                        params.input1_broadcast_val = input0_trans_pshape[i].get_length() / input1_trans_pshape[i].get_length();
                    }
                }
            }
        }

        params.indirect_input0 = primitive->indirect_a && indirect;
        params.indirect_input1 = primitive->indirect_b && indirect;
        params.indirect_axis = primitive->indirect_axis;
        if (indirect && (primitive->indirect_a || primitive->indirect_b)) {
            OPENVINO_ASSERT(impl_param.input_layouts.size() >= 3, "[GPU] Actual inputs count: ", impl_param.input_layouts.size());
            params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[get_beam_table_id(primitive)]));
        }

        bool is_quantized = true;
        for (auto& input : impl_param.input_layouts)
            is_quantized &= data_type_traits::is_quantized(input.data_type);

        if (is_quantized) {
            params.quantization = kernel_selector::QuantizationType::SYMMETRIC;
        } else {
            params.quantization = kernel_selector::QuantizationType::NONE;
        }

        params.set_dynamic_shape_offsets();
        if ((primitive->indirect_a || primitive->indirect_b) && !indirect) {
            // Need to adjust regular gemm kernel offset to skip beam table input
            for (auto& fd : params.fused_ops) {
                if (!fd.has_outer_dep())
                    continue;
                auto& fused_op_inputs = fd.tensors;
                for (auto& fused_input : fused_op_inputs) {
                    if (fused_input.is_dynamic())
                        fused_input.SetDynamicShapeOffset(fused_input.get_dynamic_shape_offset() + kernel_selector::DataTensor::max_rank());
                }
            }
            for (auto& out : params.outputs) {
                if (out.is_dynamic()) {
                    out.SetDynamicShapeOffset(out.get_dynamic_shape_offset() + kernel_selector::DataTensor::max_rank());
                }
            }
        }
        return params;
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        const auto& primitive = impl_params.typed_desc<gemm>();
        auto updated_impl_params = canonicalize_fused_shapes(impl_params);

        updated_impl_params.input_layouts = gemm_inst::transform_input_layouts(primitive, impl_params.input_layouts);
        updated_impl_params.output_layouts[0] = gemm_inst::transform_output_layout(primitive, updated_impl_params.input_layouts, impl_params.output_layouts[0]);

        for (auto& input_layout : updated_impl_params.input_layouts) {
            input_layout.set_partial_shape(extend_shape_to_rank_from_begin(input_layout.get_partial_shape()));
        }

        auto& output_layout = updated_impl_params.output_layouts[0];
        output_layout.set_partial_shape(extend_shape_to_rank_from_begin(output_layout.get_partial_shape()));

        return updated_impl_params;
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const override {
        return static_canonicalize_shapes(impl_params);
    }

    static std::unique_ptr<primitive_impl> create(const typed_program_node<gemm>& arg, const kernel_impl_params& impl_param) {
        std::vector<kernel_selector::kernel_data> kernels_data;
        auto& kernel_selector = kernel_selector_t::Instance();
        auto params = static_canonicalize_shapes(impl_param);

        auto default_kernel_params = get_kernel_params(params, params.is_dynamic(), false);
        default_kernel_params.is_shape_agnostic = params.is_dynamic();
        kernels_data.push_back(kernel_selector.get_best_kernel(default_kernel_params));
        const auto desc = params.typed_desc<gemm>();
        if (desc->indirect_a || desc->indirect_b) {
            auto indirect_kernel_params = get_kernel_params(params, params.is_dynamic(), true);
            indirect_kernel_params.is_shape_agnostic = params.is_dynamic();
            kernels_data.push_back(kernel_selector.get_best_kernel(indirect_kernel_params));
        }
        return cldnn::make_unique<gemm_impl>(kernels_data);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernels_data[default_gemm].params == nullptr) {
            _kernels_data[default_gemm].params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true, false));
        }
        update_shapes(*_kernels_data[default_gemm].params, impl_param);
        (_kernels_data[default_gemm].update_dispatch_data_func)(*_kernels_data[default_gemm].params, _kernels_data[default_gemm]);

        if (_kernels_data.size() == 2) {
            if (_kernels_data[indirect_gemm].params == nullptr) {
                _kernels_data[indirect_gemm].params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true, true));
            }
            update_shapes(*_kernels_data[indirect_gemm].params, impl_param);
            (_kernels_data[indirect_gemm].update_dispatch_data_func)(*_kernels_data[indirect_gemm].params, _kernels_data[indirect_gemm]);
        }
    }
};

namespace detail {

attach_gemm_impl::attach_gemm_impl() {
    const std::vector<data_types> types{data_types::f16,
                                        data_types::f32,
                                        data_types::i8,
                                        data_types::u8,
                                        data_types::i32};

    const std::vector<format::type> formats {
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,

        format::bfzyx,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,
        format::bs_fs_zyx_bsv32_fsv16,

        format::bfwzyx,
    };

    implementation_map<gemm>::add(impl_types::ocl, shape_types::static_shape, gemm_impl::create, types, formats);

    const std::vector<format::type> dyn_formats {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    implementation_map<gemm>::add(impl_types::ocl,
                                  shape_types::dynamic_shape,
                                  gemm_impl::create, types, dyn_formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::gemm_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::gemm)
