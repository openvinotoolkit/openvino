// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_type.hpp"
#include "registry.hpp"
#include "intel_gpu/primitives/detection_output.hpp"
#include "detection_output_inst.h"

#if OV_GPU_WITH_OCL
    #include "impls/ocl/detection_output.hpp"
#endif


namespace ov::intel_gpu {

using namespace cldnn;

static std::vector<format> supported_fmts = {
    format::bfyx,
    format::bs_fs_yx_bsv16_fsv32,
    format::bs_fs_zyx_bsv16_fsv32,
};

static std::vector<ov::element::Type_t> supported_types = {
    ov::element::f32,
    ov::element::f16,
};

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<detection_output>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::DetectionOutputImplementationManager, shape_types::static_shape,
            [](const program_node& node) {
                const auto& scores_layout = node.get_input_layout(0);
                const auto& confidence_layout = node.get_input_layout(1);
                const auto& out_layout = node.get_output_layout(0);

                if (!one_of(scores_layout.data_type, supported_types) ||
                    !one_of(confidence_layout.data_type, supported_types) ||
                    !one_of(out_layout.data_type, supported_types))
                    return false;

                if (!one_of(scores_layout.format, supported_fmts))
                    return false;
                const auto& program = node.get_program();
                const auto& device_info = program.get_engine().get_device_info();
                const int64_t lws_max = device_info.max_work_group_size;
                auto& detection_output_node = node.as<detection_output>();
                auto prim = detection_output_node.get_primitive();
                if (confidence_layout.is_dynamic()) {
                    return false;
                } else {
                    auto batch_size_limitations = (device_info.supports_immad && device_info.execution_units_count >= 256) ?
                                                    true : confidence_layout.batch() >= 4;
                    auto can_use_ocl_impl = confidence_layout.batch() <= lws_max &&
                                            batch_size_limitations &&
                                            prim->confidence_threshold >= 0.1 &&
                                            prim->top_k <= 400 && prim->num_classes >= 16 &&
                                            confidence_layout.feature() > 10000;
                    return can_use_ocl_impl;
                }
        })
        OV_GPU_GET_INSTANCE_CPU(detection_output, shape_types::static_shape)
        OV_GPU_GET_INSTANCE_CPU(detection_output, shape_types::dynamic_shape)
    };

    return impls;
}

}  // namespace ov::intel_gpu
