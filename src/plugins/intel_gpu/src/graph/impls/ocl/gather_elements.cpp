// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "gather_elements_inst.h"
#include "gather/gather_elements_kernel_selector.h"
#include "gather/gather_elements_kernel_ref.h"

namespace cldnn {
namespace ocl {

static inline kernel_selector::gather_elements_axis convert_axis(int64_t axis, size_t rank) {
    if (axis < 0) {
        axis += rank;
    }
    switch (axis) {
        case 0: return kernel_selector::gather_elements_axis::BATCH;
        case 1: return kernel_selector::gather_elements_axis::FEATURE;
        case 2:
            if (rank == 6)
                return kernel_selector::gather_elements_axis::W;
            else if (rank == 5)
                return kernel_selector::gather_elements_axis::Z;
            else
                return kernel_selector::gather_elements_axis::Y;
        case 3:
            if (rank == 6)
                return kernel_selector::gather_elements_axis::Z;
            else if (rank == 5)
                return kernel_selector::gather_elements_axis::Y;
            else
                return kernel_selector::gather_elements_axis::X;
        case 4:
            if (rank == 6)
                return kernel_selector::gather_elements_axis::Y;
            else
                return kernel_selector::gather_elements_axis::X;
        case 5: return kernel_selector::gather_elements_axis::X;
        default: OPENVINO_THROW("Incorrect gather_elements axis.");
    }
}

struct gather_elements_impl : typed_primitive_impl_ocl<gather_elements> {
    using parent = typed_primitive_impl_ocl<gather_elements>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::gather_elements_kernel_selector;
    using kernel_params_t = kernel_selector::gather_elements_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::gather_elements_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_elements_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<gather_elements>();
        auto params = get_default_params<kernel_selector::gather_elements_params>(impl_param, is_shape_agnostic);

        size_t rank = impl_param.get_output_layout().get_rank();
        params.axis = convert_axis(primitive->axis, rank);

        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
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
};

namespace detail {

attach_gather_elements_impl::attach_gather_elements_impl() {
    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i32,
        data_types::i8,
        data_types::u8
    };

    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx
    };

    implementation_map<gather_elements>::add(impl_types::ocl,
                                             shape_types::any,
                                             typed_primitive_impl_ocl<gather_elements>::create<gather_elements_impl>,
                                             types,
                                             formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::gather_elements_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::gather_elements)
