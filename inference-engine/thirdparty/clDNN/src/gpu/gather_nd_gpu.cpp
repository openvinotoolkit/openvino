// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_nd_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "gather/gather_nd_kernel_selector.h"
#include "gather/gather_nd_kernel_ref.h"

using namespace cldnn;

namespace cldnn {
namespace gpu {

struct gather_nd_gpu : typed_primitive_gpu_impl<gather_nd> {
    using parent = typed_primitive_gpu_impl<gather_nd>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_nd_gpu>(*this);
    }

    static primitive_impl* create(const gather_nd_node& arg) {
        auto gather_nd_params = get_default_params<kernel_selector::gather_nd_params>(arg);
        auto gather_nd_optional_params =
            get_default_optional_params<kernel_selector::gather_nd_optional_params>(arg.get_program());

        gather_nd_params.indices_rank = arg.get_primitive()->indices_rank;
        gather_nd_params.batch_dims = arg.get_primitive()->batch_dims;

        gather_nd_params.inputs.push_back(convert_data_tensor(arg.input(1).get_output_layout()));

        auto& kernel_selector = kernel_selector::gather_nd_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(gather_nd_params, gather_nd_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto gather_nd = new gather_nd_gpu(arg, best_kernels[0]);

        return gather_nd;
    }
};

namespace detail {

attach_gather_nd_gpu::attach_gather_nd_gpu() {
    auto val_fw = gather_nd_gpu::create;
    implementation_map<gather_nd>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<gather_nd>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<gather_nd>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), val_fw);

    implementation_map<gather_nd>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);
    implementation_map<gather_nd>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
    implementation_map<gather_nd>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfzyx), val_fw);

    implementation_map<gather_nd>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfwzyx), val_fw);
    implementation_map<gather_nd>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfwzyx), val_fw);
    implementation_map<gather_nd>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfwzyx), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
