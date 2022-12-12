// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_elements_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "gather/gather_elements_kernel_selector.h"
#include "gather/gather_elements_kernel_ref.h"
#include "intel_gpu/runtime/error_handler.hpp"

using namespace cldnn;

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
        default: IE_THROW() << "Incorrect gather_elements axis.";
    }
}

struct gather_elements_impl : typed_primitive_impl_ocl<gather_elements> {
    using parent = typed_primitive_impl_ocl<gather_elements>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::gather_elements_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::gather_elements_params, kernel_selector::gather_elements_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_elements_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<gather_elements>();
        auto params = get_default_params<kernel_selector::gather_elements_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::gather_elements_optional_params>(impl_param.get_program());

        size_t rank = impl_param.get_output_layout().get_rank();
        params.axis = convert_axis(primitive->axis, rank);

        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        return {params, optional_params};
    }
};

namespace detail {

attach_gather_elements_impl::attach_gather_elements_impl() {
    implementation_map<gather_elements>::add(impl_types::ocl, typed_primitive_impl_ocl<gather_elements>::create<gather_elements_impl>, {
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
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

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::gather_elements_impl)
