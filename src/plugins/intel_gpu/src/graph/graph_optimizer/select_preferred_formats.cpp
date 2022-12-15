// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "data_inst.h"
#include "mutable_data_inst.h"
#include "program_node.h"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include <iostream>

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl.hpp>
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "impls/onednn/utils.hpp"
#include "impls/onednn/convolution_onednn.hpp"
#include "impls/onednn/deconvolution_onednn.hpp"
#endif

using namespace cldnn;

void select_preferred_formats::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::select_preferred_formats");

    auto& engine = p.get_engine();
    const auto& device_info = engine.get_device_info();

    if (!device_info.supports_immad)
        return;

#ifdef ENABLE_ONEDNN_FOR_GPU
    engine.create_onednn_engine(p.get_config());
    for (auto n : p.get_processing_order()) {
        // Onednn primitive descriptor creation may fail, for example, due to asymmetric weight.
        try {
            if (n->is_type<convolution>()) {
                auto prim_desc = onednn::get_convolution_primitive_descriptor(*n->get_kernel_impl_params(),
                                                                              dnnl::primitive_attr(),
                                                                              dnnl::memory::format_tag::any);
                _lo.select_preferred_formats_for_onednn(*n, *prim_desc);
            } else if (n->is_type<deconvolution>()) {
                auto prim_desc = onednn::get_deconvolution_primitive_descriptor(*n->get_kernel_impl_params(),
                                                                                dnnl::primitive_attr(),
                                                                                dnnl::memory::format_tag::any);
                _lo.select_preferred_formats_for_onednn(*n, *prim_desc);
            }
        } catch(std::exception &exception) {
            GPU_DEBUG_INFO << "WARNING(select_preferred_formats): " << exception.what() << std::endl;
        }
    }
#endif  // ENABLE_ONEDNN_FOR_GPU
}
