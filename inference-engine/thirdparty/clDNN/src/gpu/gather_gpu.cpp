// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "gather/gather_kernel_selector.h"
#include "gather/gather_kernel_ref.h"
#include "error_handler.h"

using namespace cldnn;

namespace cldnn {
namespace gpu {
kernel_selector::gather_axis convert_axis(gather::gather_axis axis) {
    switch (axis) {
        case gather::along_x:
            return kernel_selector::gather_axis::X;
        case gather::along_y:
            return kernel_selector::gather_axis::Y;
        case gather::along_z:
            return kernel_selector::gather_axis::Z;
        case gather::along_w:
            return kernel_selector::gather_axis::W;
        case gather::along_f:
            return kernel_selector::gather_axis::FEATURE;
        case gather::along_b:
            return kernel_selector::gather_axis::BATCH;
        default:
            return kernel_selector::gather_axis::X;
    }
}

struct gather_gpu : typed_primitive_gpu_impl<gather> {
    using parent = typed_primitive_gpu_impl<gather>;
    using parent::parent;

public:
    static primitive_impl* create(const gather_node& arg) {
        auto gather_params = get_default_params<kernel_selector::gather_params>(arg);
        auto gather_optional_params =
            get_default_optional_params<kernel_selector::gather_optional_params>(arg.get_program());

        gather_params.axis = convert_axis(arg.get_primitive()->axis);
        gather_params.batch_dim = size_t(arg.get_primitive()->batch_dim);

        gather_params.inputs.push_back(convert_data_tensor(arg.input(1).get_output_layout()));

        auto& kernel_selector = kernel_selector::gather_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(gather_params, gather_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto gather = new gather_gpu(arg, best_kernels[0]);

        return gather;
    }
};

namespace detail {

attach_gather_gpu::attach_gather_gpu() {
    auto val_fw = gather_gpu::create;
    implementation_map<gather>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<gather>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<gather>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), val_fw);

    implementation_map<gather>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);
    implementation_map<gather>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
    implementation_map<gather>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfzyx), val_fw);

    implementation_map<gather>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfwzyx), val_fw);
    implementation_map<gather>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfwzyx), val_fw);
    implementation_map<gather>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfwzyx), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
