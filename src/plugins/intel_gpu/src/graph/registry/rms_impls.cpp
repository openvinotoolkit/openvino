// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/rms.hpp"
#include "openvino/core/validation_util.hpp"
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

        const auto& in_layout = node.get_input_layout(0);
        const bool out_blocked = node.is_valid_output_layout(0) && node.get_output_layout(0).format.is_blocked();
        if (!in_layout.format.is_blocked() && !out_blocked) {
            return true;
        }

        // Blocked layouts are served by RMSKernelFsv16 and RMSKernelBfyxOpt, which can only address them on the feature
        // axis of a rank-4/5 tensor; RMSKernelRef takes the plain layouts only.
        if (node.is_dynamic()) {
            return false;
        }
        const auto rank = static_cast<int64_t>(in_layout.get_partial_shape().size());
        return ov::util::normalize(node.as<rms>().get_primitive()->axis, rank) == 1 && (rank == 4 || rank == 5);
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
