// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_to_space_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "batch_to_space/batch_to_space_kernel_selector.h"
#include "batch_to_space/batch_to_space_kernel_ref.h"
#include "error_handler.h"
#include "data_inst.h"
#include <vector>

using namespace cldnn;

namespace cldnn {
namespace gpu {
struct batch_to_space_gpu : typed_primitive_gpu_impl<batch_to_space> {
    using parent = typed_primitive_gpu_impl<batch_to_space>;
    using parent::parent;

public:
    static primitive_impl* create(const batch_to_space_node& arg) {
        auto batch_to_space_params = get_default_params<kernel_selector::batch_to_space_params>(arg);
        auto batch_to_space_optional_params =
            get_default_optional_params<kernel_selector::batch_to_space_optional_params>(arg.get_program());

        auto primitive = arg.get_primitive();

        batch_to_space_params.block_shape = convert_dim_vector(primitive->block_shape);
        batch_to_space_params.crops_begin = convert_dim_vector(primitive->crops_begin);
        batch_to_space_params.crops_end = convert_dim_vector(primitive->crops_end);

        auto& kernel_selector = kernel_selector::batch_to_space_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(batch_to_space_params, batch_to_space_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto batch_to_space = new batch_to_space_gpu(arg, best_kernels[0]);

        return batch_to_space;
    }
};

namespace detail {

attach_batch_to_space_gpu::attach_batch_to_space_gpu() {
    auto val_fw = batch_to_space_gpu::create;
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfzyx), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfzyx), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfwzyx), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfwzyx), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfwzyx), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfwzyx), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_yx_fsv16), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_yx_fsv16), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv16), val_fw);
    implementation_map<batch_to_space>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv16), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
