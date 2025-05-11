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
    using kernel_params_t = kernel_selector::count_nonzero_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::count_nonzero_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<count_nonzero_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic() && _kernel_data.kernelName.length() != 0) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, count_nonzero_inst& instance) override {
        if (instance.get_impl_params()->input_layouts[0].count() == 0) {
            // set count of non-zero elements to 0 in case if input tensor is empty to have correct memory alloc for gather_nonzero
            return instance.output_memory(0).fill(instance.get_network().get_stream());
        } else {
            return parent::execute_impl(events, instance);
        }
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        return get_default_params<kernel_selector::count_nonzero_params>(impl_param, is_shape_agnostic);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }

        update_shapes(*_kernel_data.params, impl_param);
        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }
};

struct gather_nonzero_impl : typed_primitive_impl_ocl<gather_nonzero> {
    using parent = typed_primitive_impl_ocl<gather_nonzero>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::gather_nonzero_kernel_selector;
    using kernel_params_t = kernel_selector::gather_nonzero_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::gather_nonzero_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<gather_nonzero_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic() && _kernel_data.kernelName.length() != 0) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        auto params = get_default_params<kernel_selector::gather_nonzero_params>(impl_param, is_shape_agnostic);
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        params.ov_input_rank = static_cast<uint32_t>(impl_param.get_input_layout().get_partial_shape().size());
        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }

        update_shapes(*_kernel_data.params, impl_param);
        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        auto updated_impl_params = canonicalize_fused_shapes(impl_params);
        return updated_impl_params;
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
