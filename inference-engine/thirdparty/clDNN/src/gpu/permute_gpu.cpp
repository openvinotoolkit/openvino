// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "permute/permute_kernel_selector.h"
#include "permute/permute_kernel_ref.h"

using namespace cldnn;

namespace cldnn {
namespace gpu {

struct permute_gpu : typed_primitive_gpu_impl<permute> {
    using parent = typed_primitive_gpu_impl<permute>;
    using parent::parent;

    static primitive_impl* create(const permute_node& arg) {
        auto permute_params = get_default_params<kernel_selector::permute_params>(arg);
        auto permute_optional_params =
            get_default_optional_params<kernel_selector::permute_optional_params>(arg.get_program());

        const auto& permute_order = arg.get_primitive()->permute_order;
        permute_params.order = permute_order;
        auto& kernel_selector = kernel_selector::permute_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(permute_params, permute_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto permute = new permute_gpu(arg, best_kernels[0]);

        return permute;
    }
};

namespace detail {

attach_permute_gpu::attach_permute_gpu() {
    implementation_map<permute>::add({
        {engine_types::ocl, permute_gpu::create},
    });
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
