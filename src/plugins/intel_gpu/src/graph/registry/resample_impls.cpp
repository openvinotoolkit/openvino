// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/resample.hpp"
#include "registry.hpp"
#include "resample_inst.h"

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<resample>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_GET_INSTANCE_OCL(resample, shape_types::static_shape, [](const program_node& node) {
            // byxf is offered only where resample_opt runs; otherwise resample_ref would read features at stride F.
            const auto in_fmt = node.get_input_layout(0).format;
            const auto out_fmt = (node.is_valid_output_layout(0) ? node.get_output_layout(0) : node.calc_output_layouts()[0]).format;
            if (in_fmt != format::byxf && out_fmt != format::byxf) {
                return true;
            }
            return in_fmt == out_fmt && node.as<resample>().is_byxf_opt_supported();
        })};

    return impls;
}

}  // namespace ov::intel_gpu
