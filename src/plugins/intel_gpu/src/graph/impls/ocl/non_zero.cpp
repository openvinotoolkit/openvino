// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "non_zero_inst.h"
#include "non_zero/count_nonzero_kernel_ref.h"
#include "non_zero/count_nonzero_kernel_selector.h"
#include "non_zero/gather_nonzero_kernel_ref.h"
#include "non_zero/gather_nonzero_kernel_selector.h"

namespace cldnn {
namespace ocl {

struct count_nonzero_impl : typed_primitive_impl_ocl<count_nonzero> {
    using parent = typed_primitive_impl_ocl<count_nonzero>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::count_nonzero_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::count_nonzero_params, kernel_selector::count_nonzero_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::count_nonzero_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<count_nonzero_impl>(*this);
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
        auto params = get_default_params<kernel_selector::count_nonzero_params>(impl_param, is_shape_agnostic);
        auto optional_params = get_default_optional_params<kernel_selector::count_nonzero_optional_params>(impl_param.get_program());
        return {params, optional_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
    }
};

struct gather_nonzero_impl : typed_primitive_impl_ocl<gather_nonzero> {
    using parent = typed_primitive_impl_ocl<gather_nonzero>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::gather_nonzero_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::gather_nonzero_params, kernel_selector::gather_nonzero_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::gather_nonzero_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_nonzero_impl>(*this);
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
        auto params = get_default_params<kernel_selector::gather_nonzero_params>(impl_param, is_shape_agnostic);
        auto optional_params = get_default_optional_params<kernel_selector::gather_nonzero_optional_params>(impl_param.get_program());

        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        params.ov_input_rank = static_cast<uint32_t>(impl_param.get_input_layout().get_partial_shape().size());
        return {params, optional_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
    }
};

namespace detail {

attach_count_nonzero_impl::attach_count_nonzero_impl() {
    implementation_map<count_nonzero>::add(impl_types::ocl, shape_types::any, typed_primitive_impl_ocl<count_nonzero>::create<count_nonzero_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),

        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),

        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
        std::make_tuple(data_types::i8, format::bfwzyx),
        std::make_tuple(data_types::u8, format::bfwzyx),
    });
}

attach_gather_nonzero_impl::attach_gather_nonzero_impl() {
    implementation_map<gather_nonzero>::add(impl_types::ocl, shape_types::any, typed_primitive_impl_ocl<gather_nonzero>::create<gather_nonzero_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),

        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),

        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
        std::make_tuple(data_types::i8, format::bfwzyx),
        std::make_tuple(data_types::u8, format::bfwzyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::count_nonzero_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::gather_nonzero_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::count_nonzero)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::gather_nonzero)
