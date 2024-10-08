// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_type.hpp"
#include "registry.hpp"
#include "intel_gpu/primitives/softmax.hpp"
#include "program_node.h"
#include "primitive_inst.h"

#if OV_GPU_WITH_OCL
    #include "impls/ocl/softmax.hpp"
#endif


namespace ov {
namespace intel_gpu {

using namespace cldnn;

static std::vector<format> supported_static_fmts = {
    format::bfyx,
    format::byxf,
    format::yxfb,
    format::bfzyx
};

static std::vector<format> supported_dynamic_fmts = {
    format::bfyx,
    format::bfzyx,
};

static std::vector<ov::element::Type_t> supported_in_types = {
    ov::element::f32,
    ov::element::f16,
};

static std::vector<ov::element::Type_t> supported_out_types = {
    ov::element::f32,
    ov::element::f16,
    ov::element::i8,
    ov::element::u8,
};

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<softmax>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::SoftmaxImplementationManager, shape_types::static_shape,
            [](const program_node& node) {
                const auto& in_layout = node.get_input_layout(0);
                const auto& out_layout = node.get_output_layout(0);
                if (!one_of(in_layout.format, supported_static_fmts) || !one_of(out_layout.format, supported_static_fmts))
                    return false;

                if (!one_of(in_layout.data_type, supported_in_types))
                    return false;

                if (!one_of(out_layout.data_type, supported_out_types))
                    return false;

                return true;
        })
        OV_GPU_CREATE_INSTANCE_OCL(ocl::SoftmaxImplementationManager, shape_types::dynamic_shape,
            [](const program_node& node) {
                const auto& in_layout = node.get_input_layout(0);
                const auto& out_layout = node.get_output_layout(0);
                if (!one_of(in_layout.format, supported_dynamic_fmts) || !one_of(out_layout.format, supported_dynamic_fmts))
                    return false;

                if (!one_of(in_layout.data_type, supported_in_types))
                    return false;

                if (!one_of(out_layout.data_type, supported_out_types))
                    return false;

                return true;
        })
    };

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov
