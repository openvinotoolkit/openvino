// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/mvn.hpp"
#include "primitive_inst.h"

#include <algorithm>

namespace ov::intel_gpu {

using namespace cldnn;

namespace {

struct MvnImplementationManager : public ImplementationManagerLegacy<mvn> {
    using parent = ImplementationManagerLegacy<mvn>;

    explicit MvnImplementationManager(const parent& impl) : parent(impl) {}

    bool validate_impl(const program_node& node) const override {
        if (!parent::validate_impl(node))
            return false;

        const auto& prim = node.as<mvn>().get_primitive();
        const auto& input_layout = node.get_input_layout(0);
        const auto& fmt = input_layout.format;

        // Planar (default) layouts are always handled by the bfyx opt / ref kernels.
        if (format::is_default_format(fmt))
            return true;

        // Across-channels normalization has no optimized blocked-layout kernel: the bfyx opt kernel is
        // planar-only and the fsv16/fsv32 kernels implement WITHIN_CHANNELS only. Reject non-planar
        // layouts - for static and dynamic shapes alike - so a reorder to planar bfyx is inserted and the
        // fast bfyx opt kernel runs instead of falling back to the slow reference (mvn_gpu_ref) kernel.
        if (prim->across_channels())
            return false;

        const auto& input_pshape = input_layout.get_partial_shape();
        if (input_pshape.is_dynamic())
            return true;

        if (!prim->requires_alignment(input_pshape))
            return true;

        // Aligned MVN flattens the normalized axes into the innermost dimension (see
        // mvn_impl::static_canonicalize_shapes), which is only valid for planar or single feature-blocked
        // layouts; reject anything else (e.g. byxf) so a reorder to planar is inserted instead.
        const auto& block_sizes = format::block_sizes(fmt);
        auto axes = prim->reduction_axes;
        const auto rank = static_cast<int64_t>(input_pshape.size());
        std::for_each(axes.begin(), axes.end(), [rank](int64_t& v) {
            v = (v < 0) ? v + rank : v;
        });
        return block_sizes.size() == 1 && block_sizes[0].first == 1 &&
               (input_pshape[block_sizes[0].first].get_length() % block_sizes[0].second == 0) &&
               (std::count(axes.begin(), axes.end(), static_cast<int64_t>(block_sizes[0].first)) == 0);
    }
};

std::shared_ptr<ImplementationManager> get_mvn_implementation(shape_types shape_type) {
    const auto impl = std::dynamic_pointer_cast<ImplementationManagerLegacy<mvn>>(
        implementation_map<mvn>::get(impl_types::ocl, shape_type));
    return std::make_shared<MvnImplementationManager>(*impl);
}

}  // namespace

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<mvn>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        get_mvn_implementation(shape_types::static_shape),
        get_mvn_implementation(shape_types::dynamic_shape),
    };

    return impls;
}

}  // namespace ov::intel_gpu
