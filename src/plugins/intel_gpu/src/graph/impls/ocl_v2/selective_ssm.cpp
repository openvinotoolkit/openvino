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

class SelectiveSSMRefGenerator : public KernelGenerator {
public:
    SelectiveSSMRefGenerator() : KernelGenerator("selective_ssm_ref") {}

protected:
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }
        for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            auto& wgs = kd.params.workGroups;
            const auto& x_shape = params.get_input_layout(3).get_partial_shape();
            const auto get_dispatch_dim = [](const ov::PartialShape& shape, size_t dim) {
                if (shape.rank().is_dynamic() || dim >= shape.rank().get_length())
                    return size_t{1};
                const auto& d = shape[dim];
                return std::max<size_t>(1, static_cast<size_t>(d.is_static() ? d.get_length() : d.get_max_length()));
            };
            const size_t batch = get_dispatch_dim(x_shape, 0);
            const size_t num_heads = get_dispatch_dim(x_shape, 2);
            const size_t head_dim = get_dispatch_dim(x_shape, 3);

            wgs.global = {batch, num_heads, head_dim};
            wgs.local = {1, 1, 1};
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
