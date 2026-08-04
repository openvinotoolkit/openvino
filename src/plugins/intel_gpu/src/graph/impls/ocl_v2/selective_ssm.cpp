// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm.hpp"

#include "intel_gpu/primitives/selective_ssm.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

class SelectiveSSMRefGenerator : public KernelGenerator {
public:
    SelectiveSSMRefGenerator() : KernelGenerator("selective_ssm_ref") {}

protected:
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }
        for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        }
        for (size_t i = 0; i < 5; i++) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, static_cast<uint32_t>(i)});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            const auto& x_shape = params.get_input_layout(3).get_partial_shape();
            const auto& B_shape = params.get_input_layout(2).get_partial_shape();
            const size_t batch = x_shape[0].get_length();
            const size_t seq_len = x_shape[1].get_length();
            const size_t num_heads = x_shape[2].get_length();
            const size_t head_dim = x_shape[3].get_length();
            const size_t num_groups = B_shape[2].get_length();
            const size_t state_size = B_shape[3].get_length();

            wgs.global = {batch, num_heads, head_dim};
            wgs.local = {1, 1, 1};

            kd.params.scalars.clear();
            for (auto v : {static_cast<int32_t>(seq_len),
                           static_cast<int32_t>(num_heads),
                           static_cast<int32_t>(num_groups),
                           static_cast<int32_t>(head_dim),
                           static_cast<int32_t>(state_size)}) {
                scalar_desc desc;
                desc.t = scalar_desc::Types::INT32;
                desc.v.s32 = v;
                kd.params.scalars.push_back(desc);
            }
        }};
    }
};

class SelectiveSSMRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::SelectiveSSMRefImpl)

    Stage::Ptr selective_ssm = make_stage<SelectiveSSMRefGenerator>();

    SelectiveSSMRefImpl() : PrimitiveImplOCL(SelectiveSSMRef::get_type_info_static()) {}
    SelectiveSSMRefImpl(const program_node&, const RuntimeParams& params) : SelectiveSSMRefImpl() {
        add_stage(selective_ssm, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<SelectiveSSMRefImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> SelectiveSSMRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<selective_ssm>());
    return std::make_unique<SelectiveSSMRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::selective_ssm)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::SelectiveSSMRefImpl)
