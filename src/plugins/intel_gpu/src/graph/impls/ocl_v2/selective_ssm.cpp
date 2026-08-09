// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm.hpp"

#include <algorithm>

#include "intel_gpu/primitives/selective_ssm.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

size_t get_lws(const size_t state_size, const device_info& info) {
    const size_t limit = std::min<size_t>(32, info.max_work_group_size);
    const size_t target = std::min(std::max<size_t>(state_size, 1), limit);
    size_t lws = 1;
    while (lws * 2 <= target)
        lws *= 2;
    return lws;
}

class SelectiveSSMOptGenerator : public KernelGenerator {
public:
    SelectiveSSMOptGenerator() : KernelGenerator("selective_ssm_opt") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const bool is_bf16 = params.get_input_layout(0).data_type == ov::element::bf16;
        jit.make("SSM_TO_FLOAT(v)", is_bf16 ? "_convert_as_bfloat16_float(v)" : "convert_float(v)");
        jit.make("SSM_ROUND_STATE(v)",
                 is_bf16 ? "_convert_as_bfloat16_float(TO_INPUT5_TYPE(v))"
                         : "convert_float(TO_INPUT5_TYPE(v))");
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic())
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }
        for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        }
        args.push_back({ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            auto& wgs = kd.params.workGroups;
            if (params.is_dynamic()) {
                wgs.global = {1, 1, 1};
                wgs.local = {1, 1, 1};
                kd.params.local_memory_args = {2 * sizeof(float)};
                return;
            }

            const auto& x_shape = params.get_input_layout(3).get_partial_shape();
            const auto& B_shape = params.get_input_layout(2).get_partial_shape();
            const size_t batch = x_shape[0].get_length();
            const size_t num_heads = x_shape[2].get_length();
            const size_t head_dim = x_shape[3].get_length();
            const size_t state_size = B_shape[3].get_length();
            const size_t lws = get_lws(state_size, params.get_device_info());
            const size_t local_bytes = (state_size + lws) * sizeof(float);

            OPENVINO_ASSERT(local_bytes <= params.get_device_info().max_local_mem_size,
                            "SelectiveSSM requires ", local_bytes, " bytes of local memory, but the device exposes ",
                            params.get_device_info().max_local_mem_size, " bytes");

            wgs.global = {std::max<size_t>(head_dim, 1) * lws,
                          std::max<size_t>(num_heads, 1),
                          std::max<size_t>(batch, 1)};
            wgs.local = {lws, 1, 1};
            kd.params.local_memory_args = {local_bytes};
        }};
    }
};

class SelectiveSSMOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::SelectiveSSMOptImpl)

    Stage::Ptr selective_ssm = make_stage<SelectiveSSMOptGenerator>();

    SelectiveSSMOptImpl() : PrimitiveImplOCL(SelectiveSSMOpt::get_type_info_static()) {}
    SelectiveSSMOptImpl(const program_node&, const RuntimeParams& params) : SelectiveSSMOptImpl() {
        add_stage(selective_ssm, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<SelectiveSSMOptImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> SelectiveSSMOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<selective_ssm>());
    return std::make_unique<SelectiveSSMOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::selective_ssm)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::SelectiveSSMOptImpl)
