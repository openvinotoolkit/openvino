// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_selective_ssm.hpp"

#include <algorithm>

#include "intel_gpu/primitives/paged_selective_ssm.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

class PagedSelectiveSSMRefGenerator : public KernelGenerator {
public:
    PagedSelectiveSSMRefGenerator() : KernelGenerator("paged_selective_ssm_ref") {}

protected:
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            auto& wgs = kd.params.workGroups;
            const auto& x_shape = params.get_input_layout(3).get_partial_shape();
            const auto& seq_shape = params.get_input_layout(6).get_partial_shape();
            const auto get_dispatch_dim = [](const ov::PartialShape& shape, size_t dim) {
                if (shape.rank().is_dynamic() || dim >= shape.rank().get_length())
                    return size_t{1};
                const auto& d = shape[dim];
                return std::max<size_t>(1, static_cast<size_t>(d.is_static() ? d.get_length() : d.get_max_length()));
            };
            const size_t sequence_bounds = get_dispatch_dim(seq_shape, 0);
            const size_t sequences = sequence_bounds > 0 ? sequence_bounds - 1 : 0;
            const size_t num_heads = get_dispatch_dim(x_shape, 1);
            const size_t head_dim = get_dispatch_dim(x_shape, 2);

            wgs.global = {std::max<size_t>(1, sequences), num_heads, head_dim};
            wgs.local = {1, 1, 1};
        }};
    }
};

class PagedSelectiveSSMRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::PagedSelectiveSSMRefImpl)

    Stage::Ptr paged_selective_ssm = make_stage<PagedSelectiveSSMRefGenerator>();

    PagedSelectiveSSMRefImpl() : PrimitiveImplOCL(PagedSelectiveSSMRef::get_type_info_static()) {}
    PagedSelectiveSSMRefImpl(const program_node&, const RuntimeParams& params) : PagedSelectiveSSMRefImpl() {
        add_stage(paged_selective_ssm, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<PagedSelectiveSSMRefImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> PagedSelectiveSSMRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<paged_selective_ssm>());
    return std::make_unique<PagedSelectiveSSMRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::paged_selective_ssm)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::PagedSelectiveSSMRefImpl)
