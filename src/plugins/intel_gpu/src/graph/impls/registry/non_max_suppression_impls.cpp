// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_type.hpp"
#include "registry.hpp"
#include "intel_gpu/primitives/non_max_suppression.hpp"
#include "non_max_suppression_inst.h"

#if OV_GPU_WITH_OCL
    #include "impls/ocl/non_max_suppression.hpp"
#endif


namespace ov {
namespace intel_gpu {

using namespace cldnn;

static std::vector<format> supported_blocked_fmts = {
    format::b_fs_yx_fsv16,
    format::b_fs_yx_fsv32,
    format::bs_fs_yx_bsv16_fsv16,
    format::bs_fs_yx_bsv32_fsv16,
    format::bs_fs_yx_bsv32_fsv32,
};

static std::vector<ov::element::Type_t> supported_in_types = {
    ov::element::f32,
    ov::element::f16,
};

static std::vector<ov::element::Type_t> supported_out_types = {
    ov::element::f32,
    ov::element::f16,
    ov::element::i32,
};

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<non_max_suppression>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::NMSImplementationManager, shape_types::static_shape,
            [](const program_node& node) {
                const auto& boxes_layout = node.get_input_layout(0);
                const auto& scores_layout = node.get_input_layout(1);
                const auto& out_layout = node.get_output_layout(0);

                if (!one_of(boxes_layout.data_type, supported_in_types) || !one_of(out_layout.data_type, supported_out_types))
                    return false;

                if (one_of(boxes_layout.format, supported_blocked_fmts)) {
                    return true;
                } else {
                    const auto& nms_node = node.as<non_max_suppression>();
                    if (nms_node.get_primitive()->rotation != non_max_suppression::Rotation::NONE) {
                        return true;
                    } else {
                        if (scores_layout.is_dynamic()) {
                            return false;
                        } else {
                            const size_t kBatchNum = static_cast<size_t>(scores_layout.get_partial_shape()[0].get_length());
                            const size_t kClassNum = static_cast<size_t>(scores_layout.get_partial_shape()[1].get_length());
                            const size_t kNStreams =
                                    static_cast<size_t>(node.get_program().get_config().get_property(ov::streams::num));
                            const size_t kKeyValue = kBatchNum * std::min(kClassNum, static_cast<size_t>(8)) * kNStreams;
                            return kKeyValue > 64;
                        }
                    }
                }

                return true;
        })
        OV_GPU_GET_INSTANCE_CPU(non_max_suppression, shape_types::static_shape)
    };

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov
