// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cmath>
#include <algorithm>

#include "vl_sdpa_opt2.hpp"

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
    VLSDPAGenerator() : KernelGenerator("sdpaMha80Xe1") {}

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
        const auto output_shape = transpose_pshape(params.get_output_layout(0).get_partial_shape(), desc->output_transpose_order);

        std::cout << "----------------- VLSDPA::get_jit_constants -----------------" << std::endl;
        std::cout << "----------------- input_q_transpose_order: " << desc->input_q_transpose_order <<
        "," << params.get_input_layout(0).get_partial_shape() << "->" << query_shape << std::endl;
        std::cout << "----------------- input_k_transpose_order: " << desc->input_k_transpose_order <<
        "," << params.get_input_layout(1).get_partial_shape() << "->" << key_shape<< std::endl;
        std::cout << "----------------- output_transpose_order: " << desc->output_transpose_order <<
        "," << params.get_output_layout(0).get_partial_shape() << "->" << output_shape<< std::endl;

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
        });

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        std::cout << "----------------- VLSDPA::get_arguments_desc -----------------" << std::endl;

        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }

        for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        }

        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});

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
            const auto& out_shape = transpose_pshape(params.output_layouts[0].get_shape(), desc->output_transpose_order);
            const auto key_shape = transpose_pshape(params.get_input_layout(1).get_shape(), desc->input_k_transpose_order);

            std::cout << "----------------- VLSDPA::get_dispatch_data_func -----------------" << std::endl;
            std::cout << "----------------- output_transpose_order: " << desc->output_transpose_order <<
            "," << params.output_layouts[0].get_shape() << "->" << out_shape << std::endl;
            std::cout << "----------------- input_k_transpose_order: " << desc->input_k_transpose_order <<
            "," << params.input_layouts[1].get_shape() << "->" << key_shape << std::endl;

            const size_t headNumKv = key_shape[key_shape.size()-3];

            const size_t batchSize = static_cast<size_t>(std::floor(params.input_layouts[3].get_shape()[0] / 2));
            constexpr bool hasMask = false;

            auto& instance = reinterpret_cast<typed_primitive_inst<cldnn::vl_sdpa>&>(*rt_params->instance);
            const auto& cu_seqlens = vl_sdpa_inst::get_mask_seqlens_from_memory2(instance.cu_seqlens_memory_ptr(), params.get_stream());
            std::cout << "------------------------- " << desc->id << ", cu_seqlens=" << cu_seqlens << std::endl;
            size_t longestBatch = 0;
            for (size_t i = 1; i < cu_seqlens.size(); i++) {
                auto start_idx = cu_seqlens[i - 1];
                auto end_idx = cu_seqlens[i];
                longestBatch = std::max(longestBatch, static_cast<size_t>(end_idx - start_idx));
            }

            size_t groupH = (longestBatch + 255) / 256;
            size_t groupV = headNumKv * batchSize / 2;
            size_t localH = 64;
            size_t localV = 1;

            auto& wgs = kd.params.workGroups;
            wgs.global = {groupH * localH, groupV * localV};
            wgs.local = {localH, localV};

            std::cout << "========== WGS=" << wgs.global << ", LGS=" << wgs.local <<
             ", batchSize=" << batchSize << ", longestBatch=" << longestBatch << ", headNumKv=" << headNumKv << std::endl;

            std::vector<size_t> scalars {batchSize, hasMask};
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

class VLSDPAOptImpl2 : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::VLSDPAOptImpl2)

    Stage::Ptr vl_sdpa = make_stage<VLSDPAGenerator>();

    VLSDPAOptImpl2() : PrimitiveImplOCL(VLSDPAOpt2ImplementationManager::get_type_info_static()) {}
    VLSDPAOptImpl2(const program_node& node, const RuntimeParams& params) : VLSDPAOptImpl2() {
        add_stage(vl_sdpa, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<VLSDPAOptImpl2>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> VLSDPAOpt2ImplementationManager::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<vl_sdpa>());
    return std::make_unique<VLSDPAOptImpl2>(node, params);
}

}  // namespace ov::intel_gpu::cm

// BIND_BINARY_BUFFER_WITH_TYPE(cldnn::vl_sdpa)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::VLSDPAOptImpl2)
