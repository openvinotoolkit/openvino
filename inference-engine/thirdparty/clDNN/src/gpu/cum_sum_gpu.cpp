// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cum_sum_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "cum_sum/cum_sum_kernel_selector.h"
#include "cum_sum/cum_sum_kernel_ref.h"
#include "error_handler.h"

using namespace cldnn;

namespace cldnn {
namespace gpu {

namespace {
kernel_selector::cum_sum_axis convert_axis(cum_sum::cum_sum_axis axis) {
    switch (axis) {
        case cum_sum::along_x:
            return kernel_selector::cum_sum_axis::X;
        case cum_sum::along_y:
            return kernel_selector::cum_sum_axis::Y;
        case cum_sum::along_z:
            return kernel_selector::cum_sum_axis::Z;
        case cum_sum::along_w:
            return kernel_selector::cum_sum_axis::W;
        case cum_sum::along_f:
            return kernel_selector::cum_sum_axis::FEATURE;
        case cum_sum::along_b:
            return kernel_selector::cum_sum_axis::BATCH;
        default:
            return kernel_selector::cum_sum_axis::BATCH;
    }
}
}  // namespace

struct cum_sum_gpu : typed_primitive_gpu_impl<cum_sum> {
    using parent = typed_primitive_gpu_impl<cum_sum>;
    using parent::parent;

public:
    static primitive_impl* create(const cum_sum_node& arg) {
        auto cum_sum_params = get_default_params<kernel_selector::cum_sum_params>(arg);
        auto cum_sum_optional_params =
            get_default_optional_params<kernel_selector::cum_sum_optional_params>(arg.get_program());

        cum_sum_params.axis = convert_axis(arg.get_primitive()->axis);
        cum_sum_params.exclusive = arg.get_primitive()->exclusive;
        cum_sum_params.reverse = arg.get_primitive()->reverse;

        auto& kernel_selector = kernel_selector::cum_sum_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(cum_sum_params, cum_sum_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto cum_sum = new cum_sum_gpu(arg, best_kernels[0]);

        return cum_sum;
    }
};

namespace detail {

attach_cum_sum_gpu::attach_cum_sum_gpu() {
    auto val_fw = cum_sum_gpu::create;
    implementation_map<cum_sum>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<cum_sum>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
    implementation_map<cum_sum>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfwzyx), val_fw);
    implementation_map<cum_sum>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<cum_sum>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);
    implementation_map<cum_sum>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfwzyx), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
