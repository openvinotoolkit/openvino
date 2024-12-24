// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/registry/implementation_manager.hpp"
#include "program_node.h"
//#include "intel_gpu/primitives/resample.hpp"

#include <memory>
namespace cldnn {
namespace ocl {

struct ResampleImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::resample")
    ResampleImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, vf) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
    bool validate_impl(const program_node& node) const override {
        // auto prim = node.as<resample>().get_primitive();
        // const auto& in0_layout = node.get_input_layout(0);

        // if (in0_layout.data_type == ov::element::f32 &&
        //     prim->operation_type == ov::op::util::InterpolateBase::InterpolateMode::LINEAR_ONNX &&
        //     prim->coord_trans_mode == ov::op::util::InterpolateBase::CoordinateTransformMode::ALIGN_CORNERS &&
        //     prim->shape_calc_mode == ov::op::util::InterpolateBase::ShapeCalcMode::SCALES) {
        //     return false;
        // }

        return true;
    }
};

}  // namespace ocl
}  // namespace cldnn
