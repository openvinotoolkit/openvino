// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "vl_sdpa_opt.hpp"

#include <algorithm>
#include <cmath>

#include "common_utils/kernel_generator_base.hpp"
#include "intel_gpu/primitives/vl_sdpa.hpp"
#include "primitive_cm_base.hpp"
#include "primitive_inst.h"
#include "registry/implementation_manager.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::cm {
namespace {

constexpr auto get_vlsdpa_build_options() {
    return " -cmc -Qxcm_register_file_size=256";
}

struct VLSDPARuntimeParams : public ImplRuntimeParams {
    std::vector<int32_t> cu_seqlens;
};

class VLSDPAGenerator : public KernelGenerator {
public:
    VLSDPAGenerator() : KernelGenerator("cm_sdpa_vlen") {}

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

        const size_t head_size = key_shape[query_shape.size() - 1].get_length();
        const size_t num_q_heads = query_shape[query_shape.size() - 3].get_length();
        const size_t num_kv_heads = key_shape[key_shape.size() - 3].get_length();
        const float scale_factor = 1.0 / std::sqrt(static_cast<double>(head_size));

        GPU_DEBUG_TRACE_DETAIL << "VLSDPA query_shape " << query_shape << ", q_transpose_order " << PartialShape(desc->input_q_transpose_order)
                               << ", key_shape " << key_shape << ", k_transpose_order " << PartialShape(desc->input_k_transpose_order)
                               << ", head_size=" << head_size << ", num_q_heads=" << num_q_heads << ", num_kv_heads=" << num_kv_heads << '\n';

        jit.add({
            make_jit_constant("KERNEL_NAME", get_entry_point(params)),
            make_jit_constant("CMFLA_NUM_HEADS", num_q_heads),
            make_jit_constant("CMFLA_NUM_KV_HEADS", num_kv_heads),
            make_jit_constant("CMFLA_HEAD_SIZE", head_size),
            make_jit_constant("CMFLA_SCALE_FACTOR", scale_factor),
        });

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        for (uint32_t i = 0; i < params.input_layouts.size() - 1; i++) {  // inputs: q, k, v
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }

        for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        }

        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(params.input_layouts.size() - 1)});  // input: cu_seq_lens

        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
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
            const auto query_shape = transpose_pshape(params.get_input_layout(0).get_shape(), desc->input_q_transpose_order);
            const size_t num_q_heads = query_shape[query_shape.size() - 3];  // TODO: make it to be configuration of primitive_inst

            const auto& vlsdpa_rt_params = static_cast<VLSDPARuntimeParams&>(*rt_params);
            const auto& cu_seqlens = vlsdpa_rt_params.cu_seqlens;

            size_t max_seq_len = 0;
            for (size_t i = 1; i < cu_seqlens.size(); i++) {
                auto start_idx = cu_seqlens[i - 1];
                auto end_idx = cu_seqlens[i];
                max_seq_len = std::max(max_seq_len, static_cast<size_t>(end_idx - start_idx));
            }

            const auto& info = params.get_device_info();
            const size_t CM_GRF_WIDTH = (info.arch <= gpu_arch::xe_hpc) ? 256 : 512;
            const size_t q_step = static_cast<size_t>(std::floor(CM_GRF_WIDTH / 32));  // or 8 on Xe1
            size_t wg_size = static_cast<size_t>(std::floor((max_seq_len + q_step - 1) / q_step));
            int32_t need_wg_mapping = 0;
            if (wg_size > 16) {
                // # seq_len is too big to fit into a single work-group
                // # will use fixed work-group size 16, process 16*16 (or 16*8 on xe1)
                // # part of sequence, in this case, kernel needs to figure-out which part
                // # it needs to handle
                need_wg_mapping = 1;
                wg_size = 16;
            }

            size_t wg_count;
            if (need_wg_mapping) {
                wg_count = 0;
                const auto wg_seq_len = wg_size * q_step;
                for (size_t i = 1; i < cu_seqlens.size(); i++) {
                    auto start_idx = cu_seqlens[i - 1];
                    auto end_idx = cu_seqlens[i];
                    wg_count += static_cast<size_t>(std::floor((end_idx - start_idx + wg_seq_len - 1) / wg_seq_len));
                }
            } else {
                wg_count = cu_seqlens.size() - 1;
            }

            auto& wgs = kd.params.workGroups;
            wgs.global = {num_q_heads, wg_count * wg_size, 1};
            wgs.local = {1, wg_size, 1};

            std::vector<int32_t> scalars{need_wg_mapping};
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

    VLSDPAOptImpl() : PrimitiveImplOCL(VLSDPAOptImplementationManager::get_type_info_static()) {
        m_rt_params = std::make_unique<VLSDPARuntimeParams>();
    }
    VLSDPAOptImpl(const program_node& node, const RuntimeParams& params) : VLSDPAOptImpl() {
        add_stage(vl_sdpa, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<VLSDPAOptImpl>(this);
    }

    void update_rt_params(const cldnn::primitive_inst& instance) override {
        update_stages_flags(instance);

        auto rt_params = static_cast<VLSDPARuntimeParams*>(m_rt_params.get());
        auto& vlsdpa_instance = dynamic_cast<const typed_primitive_inst<cldnn::vl_sdpa>&>(instance);
        vlsdpa_instance.get_mask_seqlens_from_memory(rt_params->cu_seqlens);
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance) override {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<VLSDPARuntimeParams>();
        }
        return PrimitiveImplCM::execute(events, instance);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> VLSDPAOptImplementationManager::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<vl_sdpa>());
    return std::make_unique<VLSDPAOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::vl_sdpa)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::VLSDPAOptImpl)
