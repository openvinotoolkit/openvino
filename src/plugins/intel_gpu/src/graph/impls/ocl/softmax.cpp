// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "softmax/softmax_kernel_selector.h"
#include "softmax/softmax_kernel_base.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

static inline kernel_selector::softmax_dim get_softmax_dim(int64_t axis, size_t rank) {
    if (axis < 0) {
        axis += rank;
    }
    switch (axis) {
        case 0: return kernel_selector::softmax_dim::BATCH;
        case 1: return kernel_selector::softmax_dim::FEATURE;
        case 2:
            if (rank > 4)
                return kernel_selector::softmax_dim::Z;
            else
                return kernel_selector::softmax_dim::Y;
        case 3:
            if (rank > 4)
                return kernel_selector::softmax_dim::Y;
            else
                return kernel_selector::softmax_dim::X;
        case 4: return kernel_selector::softmax_dim::X;
        default: IE_THROW() << "Invalid softmax axis " << axis;
    }
}

struct softmax_impl : typed_primitive_impl_ocl<softmax> {
    using parent = typed_primitive_impl_ocl<softmax>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<softmax_impl>(*this);
    }

    static primitive_impl* create(const softmax_node& arg, const kernel_impl_params& impl_param) {
        const auto primitive = arg.get_primitive();
        auto sm_params = get_default_params<kernel_selector::softmax_params>(impl_param);
        auto sm_optional_params =
            get_default_optional_params<kernel_selector::softmax_optional_params>(arg.get_program());

        size_t rank = arg.get_output_layout().get_rank();
        sm_params.dim = get_softmax_dim(primitive->dimension, rank);

        auto& kernel_selector = kernel_selector::softmax_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(sm_params, sm_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto softmax_node = new softmax_impl(arg, best_kernels[0]);

        return softmax_node;
    }
};

namespace detail {

attach_softmax_impl::attach_softmax_impl() {
    implementation_map<softmax>::add(impl_types::ocl, softmax_impl::create, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
