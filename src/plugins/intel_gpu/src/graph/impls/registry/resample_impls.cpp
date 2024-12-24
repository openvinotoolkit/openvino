// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "predicates.hpp"
#include "registry.hpp"
#include "intel_gpu/primitives/resample.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_OCL
    #include "impls/ocl/resample.hpp"
#endif


namespace ov {
namespace intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<resample>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::ResampleImplementationManager, shape_types::static_shape,
            [](const cldnn::program_node& node){
                auto prim = node.as<resample>().get_primitive();
                const auto& in0_layout = node.get_input_layout(0);

                if (in0_layout.data_type == ov::element::f32 &&
                    prim->operation_type == ov::op::util::InterpolateBase::InterpolateMode::LINEAR_ONNX &&
                    prim->coord_trans_mode == ov::op::util::InterpolateBase::CoordinateTransformMode::ALIGN_CORNERS &&
                    prim->shape_calc_mode == ov::op::util::InterpolateBase::ShapeCalcMode::SCALES) {
                    return false;
                }

                return true;
        })
        OV_GPU_GET_INSTANCE_CPU(resample, shape_types::static_shape,
            [](const cldnn::program_node& node){
                return true;
        })
    };

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov
