// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cmath>
#include <algorithm>

#include "vl_sdpa_opt.hpp"

#include "common_utils/kernel_generator_base.hpp"
#include "primitive_cm_base.hpp"
#include "primitive_inst.h"
#include "registry/implementation_manager.hpp"
#include "utils/kernel_generator.hpp"
#include "intel_gpu/primitives/vl_sdpa.hpp"

namespace ov::intel_gpu::cm {
namespace {

constexpr auto get_vlsdpa_build_options() {
    return " -cmc -Qxcm_register_file_size=256 -mdump_asm -g2 ";
}

// Overload << operator for vectors
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}
class VLSDPAGenerator : public KernelGenerator {
public:
    VLSDPAGenerator() : KernelGenerator("vl_sdpa_opt") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_vlsdpa_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        // transpose shape into BHLS(4D), or HLS(3D)
        auto transpose_pshape = [](const ov::PartialShape& pshape, const std::vector<int64_t>& order) {
            if (order.empty())
                return pshape;

            auto transposed_pshape = ov::PartialShape::dynamic(pshape.rank());
            for (size_t i = 0; i < order.size(); i++) {
                transposed_pshape[i] = pshape[order[i]];
            }
            return transposed_pshape;
        };

        auto desc = params.typed_desc<vl_sdpa>();
        const auto query_shape = transpose_pshape(params.get_input_layout(0).get_partial_shape(), desc->input_q_transpose_order);
        const auto key_shape = transpose_pshape(params.get_input_layout(1).get_partial_shape(), desc->input_k_transpose_order);

        std::cout << "----------------- VLSDPA::get_jit_constants -----------------" << std::endl;
        std::cout << "----------------- input_q_transpose_order: " << desc->input_q_transpose_order <<
        "," << params.get_input_layout(0).get_partial_shape() << "->" << query_shape << std::endl;
        std::cout << "----------------- input_k_transpose_order: " << desc->input_k_transpose_order <<
        "," << params.get_input_layout(1).get_partial_shape() << "->" << key_shape<< std::endl;

        const size_t head_size = key_shape[query_shape.size()-1].get_length();
        const size_t num_q_heads = query_shape[query_shape.size()-3].get_length();
        const size_t num_kv_heads = key_shape[key_shape.size()-3].get_length();
        const float scale_factor = 1.0 / std::sqrt(static_cast<double>(head_size));

        std::cout << "========== KERNEL_NAME(" << get_entry_point(params) << ") head_size=" << head_size <<
         ", num_q_heads=" << num_q_heads << ", num_kv_heads=" << num_kv_heads <<
         ", scale_factor=" << scale_factor << std::endl;

        // TODO: jit for transpose
        jit.add({
            make_jit_constant("KERNEL_NAME", get_entry_point(params)),
            make_jit_constant("num_heads", num_q_heads),
            make_jit_constant("num_kv_heads", num_kv_heads),
            make_jit_constant("head_size", head_size),
            make_jit_constant("q_step", 16),
            make_jit_constant("kv_step", 16),
            make_jit_constant("scale_factor", scale_factor),
            make_jit_constant("args_verbose", -1),
        });

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        std::cout << "----------------- VLSDPA::get_arguments_desc -----------------" << std::endl;

        for (uint32_t i = 0; i < params.input_layouts.size() - 1; i++) { // skip attention_mask input
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }

        for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        }

        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 2});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 3});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 4});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 5});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            auto desc = params.typed_desc<vl_sdpa>();

            // transpose shape into BHLS(4D), or HLS(3D)
            auto transpose_pshape = [](const ov::Shape& pshape, const std::vector<int64_t>& order) {
                if (order.empty())
                    return pshape;

                auto transposed_pshape = ov::Shape(pshape.size());
                for (size_t i = 0; i < order.size(); i++) {
                    transposed_pshape[i] = pshape[order[i]];
                }
                return transposed_pshape;
            };
            const auto& out_shape = transpose_pshape(params.output_layouts[0].get_shape(), desc->output_transpose_order);

            std::cout << "----------------- VLSDPA::get_dispatch_data_func -----------------" << std::endl;
            std::cout << "----------------- output_transpose_order: " << desc->output_transpose_order <<
            "," << params.output_layouts[0].get_shape() << "->" << out_shape << std::endl;
            std::cout << "----------------- input_k_transpose_order: " << desc->input_k_transpose_order <<
            "," << params.input_layouts[1].get_shape() << "->" << transpose_pshape(params.input_layouts[1].get_shape(), desc->input_k_transpose_order) << std::endl;

            // output_transpose_order: [0, 1, 2],[2304,16,80]->[2304,16,80] FIXME BLHS/LHS???
            const size_t batch = out_shape.size() < 4 ? 1 : out_shape[0];
            const size_t q_len = out_shape[out_shape.size()-3];
            const size_t num_heads = out_shape[out_shape.size()-2];
            const size_t kv_len = q_len;
            constexpr size_t q_step = 16;

            const size_t q_steps = static_cast<size_t>(std::floor(q_len / q_step));
            size_t WG_SIZE = std::min(q_steps, static_cast<size_t>(8ul));

            std::cout << "========== batch=" << batch << ", q_len=" << q_len <<
             ", num_heads=" << num_heads << ", q_steps=" << q_steps << ", WG_SIZE=" << WG_SIZE << std::endl;

            wgs.global = {batch, num_heads, q_steps};
            wgs.local = {1, 1, WG_SIZE};

            std::vector<size_t> scalars {q_len, kv_len, q_len, kv_len, 0, 0};
            kd.params.scalars.clear();
            for (auto i : scalars) {
                scalar_desc desc;
                desc.t = scalar_desc::Types::INT32;
                desc.v.s32 = static_cast<int32_t>(i);
                kd.params.scalars.push_back(desc);
            }
        }};
    }
};

class VLSDPAOptImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::VLSDPAOptImpl)

    Stage::Ptr vl_sdpa = make_stage<VLSDPAGenerator>();

    VLSDPAOptImpl() : PrimitiveImplOCL(VLSDPAOptImplementationManager::get_type_info_static()) {}
    VLSDPAOptImpl(const program_node& node, const RuntimeParams& params) : VLSDPAOptImpl() {
        add_stage(vl_sdpa, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<VLSDPAOptImpl>(this);
    }

    // execute an inner piece of attention mask
    cldnn::event::ptr execute_stage(const std::vector<cldnn::event::ptr>& events,
                                    cldnn::primitive_inst& instance,
                                    Stage& stage,
                                    const std::pair<int32_t, int32_t>& indices,
                                    bool needs_completion_event = false) const {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("MoeExpertOptImpl::execute_stage"));
        cldnn::stream& stream = instance.get_network().get_stream();

        auto& kd = stage.kd;
        cldnn::kernel_arguments_desc& desc = kd.params;

        if (kd.need_dispatch_data_update) {
            kd.update_dispatch_data_func(*instance.get_impl_params(), kd, m_rt_params.get());
            kd.need_dispatch_data_update = false;
        }

        // update a proportion for each iteration however
        auto [start_idx, end_idx] = indices;
        auto valid_q_len = end_idx - start_idx;

        constexpr size_t q_step = 16;
        const size_t q_steps = static_cast<size_t>(std::floor(valid_q_len / q_step));
        size_t WG_SIZE = std::min(q_steps, static_cast<size_t>(8ul));

        desc.workGroups.global.back() = q_steps;
        desc.workGroups.local.back() = WG_SIZE;
        auto& scalars = desc.scalars;
        scalars[0].v.s32 = valid_q_len;
        scalars[1].v.s32 = valid_q_len;
        scalars[4].v.s32 = start_idx;
        scalars[5].v.s32 = end_idx;

        // always need_args_update
        {
            // get_arguments
            cldnn::kernel_arguments_data args;
            for (size_t i = 0; i < instance.inputs_memory_count() - 1; i++) { // skip attention_mask input
                args.inputs.push_back(instance.input_memory_ptr(i));
            }

            for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
                args.outputs.push_back(instance.output_memory_ptr(i));
            }

            args.scalars = &desc.scalars;
            stream.set_arguments(*stage.kernel, desc, args);
        }

        const auto& gws = desc.workGroups.global;
        const auto& lws = desc.workGroups.local;

        GPU_DEBUG_TRACE_DETAIL << "Enqueue stage " << stage.kernel->get_id() << " : gws=[" << gws[0] << ", " << gws[1] << ", " << gws[2] << "] "
                               << "lws=[" << lws[0] << ", " << lws[1] << ", " << lws[2] << "]" << (needs_completion_event ? " has_completion_event=true" : "")
                               << '\n';

        return stream.enqueue_kernel(*stage.kernel, desc, {}, events, needs_completion_event);
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& ins) override {
        auto& instance = reinterpret_cast<typed_primitive_inst<cldnn::vl_sdpa>&>(ins);
        update_rt_params(instance);

        std::vector<cldnn::event::ptr> tmp_events(events);
        const auto& cu_seqlens = instance.get_mask_seqlens_from_memory();
        for (size_t i = 1; i < cu_seqlens.size(); i++) {
            auto start_idx = cu_seqlens[i - 1];
            auto end_idx = cu_seqlens[i];

            tmp_events = {execute_stage(tmp_events,
                          instance, *_stages[_order[0]],
                          std::pair{start_idx, end_idx},
                          (i == cu_seqlens.size() - 1) ? instance.needs_completion_event() : false)};
        }

        return tmp_events[0];
    }
};

}  // namespace

std::unique_ptr<primitive_impl> VLSDPAOptImplementationManager::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<vl_sdpa>());
    return std::make_unique<VLSDPAOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::cm

// BIND_BINARY_BUFFER_WITH_TYPE(cldnn::vl_sdpa)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::VLSDPAOptImpl)
