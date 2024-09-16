// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/registry/implementation_manager.hpp"
#include "program_node.h"

#include <memory>
namespace cldnn {
namespace ocl {

struct CTCLoss : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::ctc_loss")
    CTCLoss(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, vf) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
    bool validate_impl(const program_node& node) const override {
    //         auto types = {data_types::f16, data_types::f32};

    // auto formats = {format::bfyx,
    //                 format::b_fs_yx_fsv16,
    //                 format::b_fs_yx_fsv32,
    //                 format::bs_fs_yx_bsv16_fsv16,
    //                 format::bs_fs_yx_bsv32_fsv32,
    //                 format::bs_fs_yx_bsv32_fsv16};
        return true;
    }
};

}  // namespace ocl
}  // namespace cldnn
