// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/rms.hpp"
#include "primitive_inst.h"
#include "registry.hpp"

namespace ov::intel_gpu {

using namespace cldnn;

namespace {

struct RMSImplementationManager : public ImplementationManagerLegacy<rms> {
    using parent = ImplementationManagerLegacy<rms>;

    explicit RMSImplementationManager(const parent& impl) : parent(impl) {}

    bool validate_impl(const program_node& node) const override {
        if (!parent::validate_impl(node)) {
            return false;
        }

        auto is_blocked = [](format f) {
            return f == format::b_fs_yx_fsv16 || f == format::b_fs_zyx_fsv16;
        };
        const auto& in_layout = node.get_input_layout(0);
        const auto out_fmt = node.is_valid_output_layout(0) ? node.get_output_layout(0).format : format(format::any);
        if (!is_blocked(in_layout.format) && !is_blocked(out_fmt)) {
            return true;
        }

        // Blocked layouts are served by RMSKernelBfyxOpt, which can only address them on the feature axis of a
        // rank-4/5 tensor; RMSKernelRef takes the plain layouts only.
        if (node.is_dynamic() || in_layout.is_dynamic()) {
            return false;
        }
        const auto& prim = node.as<rms>().get_primitive();
        const auto rank = static_cast<int64_t>(in_layout.get_partial_shape().size());
        const auto axis = prim->axis < 0 ? prim->axis + rank : prim->axis;
        return axis == 1 && (rank == 4 || rank == 5);
    }
};

std::shared_ptr<ImplementationManager> get_rms_implementation(shape_types shape_type) {
    const auto impl = std::dynamic_pointer_cast<ImplementationManagerLegacy<rms>>(implementation_map<rms>::get(impl_types::ocl, shape_type));
    return std::make_shared<RMSImplementationManager>(*impl);
}

}  // namespace

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<rms>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        get_rms_implementation(shape_types::static_shape),
        get_rms_implementation(shape_types::dynamic_shape),
    };

    return impls;
}

}  // namespace ov::intel_gpu
