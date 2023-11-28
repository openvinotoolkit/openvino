// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "shape_of_inst.h"
#include "shape_of/shape_of_kernel_selector.h"
#include "shape_of/shape_of_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct shape_of_impl : typed_primitive_impl_ocl<shape_of> {
    using parent = typed_primitive_impl_ocl<shape_of>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::shape_of_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::shape_of_params, kernel_selector::shape_of_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::shape_of_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<shape_of_impl>(*this);
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
        auto params = get_default_params<kernel_selector::shape_of_params>(impl_param, is_shape_agnostic);
        auto optional_params = get_default_optional_params<kernel_selector::shape_of_optional_params>(impl_param.get_program());

        auto input_layout = impl_param.get_input_layout(0);
        params.input_rank = input_layout.is_dynamic() ? input_layout.get_partial_shape().size() : input_layout.get_rank();
        params.input_dims = input_layout.is_dynamic() ? std::vector<cldnn::tensor::value_type>{} : input_layout.get_dims();

        return {params, optional_params};
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        auto updated_impl_params = canonicalize_fused_shapes(impl_params);

        auto& input_layout = updated_impl_params.input_layouts[0];
        input_layout.set_partial_shape(extend_shape_to_rank_from_end(input_layout.get_partial_shape(), layout::max_rank()));

        auto& output_layout = updated_impl_params.output_layouts[0];
        output_layout.set_partial_shape(extend_shape_to_rank_from_end(output_layout.get_partial_shape(), layout::max_rank()));

        return updated_impl_params;
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const override {
        return static_canonicalize_shapes(impl_params);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
    }
};

namespace detail {

attach_shape_of_impl::attach_shape_of_impl() {
    implementation_map<shape_of>::add(impl_types::ocl, shape_types::static_shape, typed_primitive_impl_ocl<shape_of>::create<shape_of_impl>, {});

    auto dyn_types = {
        data_types::f32,
        data_types::f16,
        data_types::i8,
        data_types::u8,
        data_types::i32
    };

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx
    };

    implementation_map<shape_of>::add(impl_types::ocl,
                                      shape_types::dynamic_shape,
                                      typed_primitive_impl_ocl<shape_of>::create<shape_of_impl>,
                                      dyn_types,
                                      dyn_formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::shape_of_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::shape_of)
