// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "gather/gather_kernel_selector.h"
#include "gather/gather_kernel_ref.h"
#include "cldnn/runtime/error_handler.hpp"

using namespace cldnn;

namespace cldnn {
namespace ocl {
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

struct gather_impl : typed_primitive_impl_ocl<gather> {
    using parent = typed_primitive_impl_ocl<gather>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_impl>(*this);
    }

public:
    static primitive_impl* create(const gather_node& arg) {
        auto gather_params = get_default_params<kernel_selector::gather_params>(arg);
        auto gather_optional_params =
            get_default_optional_params<kernel_selector::gather_optional_params>(arg.get_program());

        gather_params.axis = convert_axis(arg.get_primitive()->axis);
        gather_params.batch_dim = size_t(arg.get_primitive()->batch_dim);
        gather_params.support_neg_ind = arg.get_primitive()->support_neg_ind;

        gather_params.inputs.push_back(convert_data_tensor(arg.input(1).get_output_layout()));

        auto& kernel_selector = kernel_selector::gather_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(gather_params, gather_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto gather = new gather_impl(arg, best_kernels[0]);

        return gather;
    }
};

namespace detail {

attach_gather_impl::attach_gather_impl() {
    implementation_map<gather>::add(impl_types::ocl, gather_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
