// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "data_inst.h"
#include "mutable_data_inst.h"
#include "program_node.h"
#include "intel_gpu/runtime/engine.hpp"
#include "runtime/cldnn_itt.hpp"
#include <iostream>
#include "to_string_utils.h"
#include "intel_gpu/runtime/debug_configuration.hpp"
#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl.hpp>
#include "impls/onednn/utils.hpp"
#include "impls/onednn/convolution_onednn.hpp"
#include "impls/onednn/deconvolution_onednn.hpp"
#endif

using namespace cldnn;

#ifdef ENABLE_ONEDNN_FOR_GPU
static dnnl::primitive_desc get_convolution_prim_desc(cldnn::engine& engine, program_node& n) {
    auto desc = onednn::get_convolution_descriptor(*n.get_kernel_impl_params(), dnnl::memory::format_tag::any);
    // Note: did not handle attribute properly. especially for zero-point
    dnnl::primitive_desc prim_desc{&desc->data, nullptr, engine.get_onednn_engine(), nullptr};
    return prim_desc;
}

static dnnl::primitive_desc get_deconvolution_prim_desc(cldnn::engine& engine, program_node& n) {
    auto desc = onednn::get_deconvolution_descriptor(*n.get_kernel_impl_params(), dnnl::memory::format_tag::any);
    dnnl::primitive_desc prim_desc{&desc->data, nullptr, engine.get_onednn_engine(), nullptr};
    return prim_desc;
}
#endif

void set_required_layouts::run(program& p) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "CLDNN::pass::SetRequiredLayouts");

    auto& engine = p.get_engine();
    const auto& device_info = engine.get_device_info();

    if (!device_info.supports_immad)
        return;

#ifdef ENABLE_ONEDNN_FOR_GPU
    GPU_DEBUG_GET_INSTANCE(debug_config);
    for (auto n : p.get_processing_order()) {
        if (!(n->is_type<convolution>() || n->is_type<deconvolution>())
            || !layout_optimizer::are_data_types_suitable_for_onednn(*n)) {
            // only care for onednn convolutions
            continue;
        }

        // Onednn primitive descriptor creation may fail, for example, due to asymmetric weight.
        try {
            dnnl::primitive_desc prim_desc;
            if (n->is_type<convolution>()) {
                prim_desc = get_convolution_prim_desc(engine, *n);
            } else if (n->is_type<deconvolution>()) {
                prim_desc = get_deconvolution_prim_desc(engine, *n);
            }

            auto src_fmt = onednn::find_data_format(prim_desc.src_desc());
            auto dst_fmt = onednn::find_data_format(prim_desc.dst_desc());
            GPU_DEBUG_GET_INSTANCE(debug_config);
            GPU_DEBUG_IF(debug_config->verbose >= 2) {
                std::cout << "set_required_layouts:" << n->id() << ": " << fmt_to_str(src_fmt) << " --> " << fmt_to_str(dst_fmt) << std::endl;
            }
            n->set_required_input0(src_fmt);
            n->set_required_output(dst_fmt);
        } catch(std::exception &exception) {
            GPU_DEBUG_IF(debug_config->verbose >= 1) {
                std::cout << "WARNING(set_required_layouts): " << exception.what() << std::endl;
            }
        }
    }
#endif
}
