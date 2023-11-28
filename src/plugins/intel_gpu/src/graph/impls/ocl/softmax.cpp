// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "softmax_inst.h"
#include "softmax/softmax_kernel_selector.h"
#include "softmax/softmax_kernel_base.h"

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
        default: OPENVINO_THROW("Invalid softmax axis ", axis);
    }
}

struct softmax_impl : typed_primitive_impl_ocl<softmax> {
    using parent = typed_primitive_impl_ocl<softmax>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::softmax_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::softmax_params, kernel_selector::softmax_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::softmax_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<softmax_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->SetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<softmax>();
        auto params = get_default_params<kernel_selector::softmax_params>(impl_param, is_shape_agnostic);
        auto optional_params = get_default_optional_params<kernel_selector::softmax_optional_params>(impl_param.get_program());

        size_t rank = impl_param.get_output_layout().get_rank();
        params.dim = get_softmax_dim(primitive->dimension, rank);

        return {params, optional_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
    }
};

namespace detail {

attach_softmax_impl::attach_softmax_impl() {
    auto types = {data_types::f16, data_types::f32};
    auto formats = {
            format::bfyx,
            format::byxf,
            format::yxfb,
            format::bfzyx
    };

    implementation_map<softmax>::add(impl_types::ocl, shape_types::static_shape, typed_primitive_impl_ocl<softmax>::create<softmax_impl>, types, formats);

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx,
    };

    implementation_map<softmax>::add(impl_types::ocl, shape_types::dynamic_shape, typed_primitive_impl_ocl<softmax>::create<softmax_impl>, types, dyn_formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::softmax_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::softmax)
