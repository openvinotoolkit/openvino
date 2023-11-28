// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "cum_sum_inst.h"
#include "cum_sum/cum_sum_kernel_selector.h"
#include "cum_sum/cum_sum_kernel_ref.h"

namespace cldnn {
namespace ocl {

namespace {
kernel_selector::cum_sum_axis convert_axis(int64_t axis, size_t rank) {
    if (axis < 0) {
        axis += rank;
    }
    switch (axis) {
        case 0: return kernel_selector::cum_sum_axis::BATCH;
        case 1: return kernel_selector::cum_sum_axis::FEATURE;
        case 2:
            if (rank == 6)
                return kernel_selector::cum_sum_axis::W;
            else if (rank == 5)
                return kernel_selector::cum_sum_axis::Z;
            else
                return kernel_selector::cum_sum_axis::Y;
        case 3:
            if (rank == 6)
                return kernel_selector::cum_sum_axis::Z;
            else if (rank == 5)
                return kernel_selector::cum_sum_axis::Y;
            else
                return kernel_selector::cum_sum_axis::X;
        case 4:
            if (rank == 6)
                return kernel_selector::cum_sum_axis::Y;
            else
                return kernel_selector::cum_sum_axis::X;
        case 5: return kernel_selector::cum_sum_axis::X;
        default: return kernel_selector::cum_sum_axis::BATCH;
    }
}
}  // namespace

struct cum_sum_impl : typed_primitive_impl_ocl<cum_sum> {
    using parent = typed_primitive_impl_ocl<cum_sum>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::cum_sum_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::cum_sum_params, kernel_selector::cum_sum_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::cum_sum_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<cum_sum_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->SetUpdateDispatchDataFunc(_kernel_data);
        }
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<cum_sum>();
        auto params = get_default_params<kernel_selector::cum_sum_params>(impl_param, is_shape_agnostic);
        auto optional_params = get_default_optional_params<kernel_selector::cum_sum_optional_params>(impl_param.get_program());

        size_t rank = impl_param.get_output_layout().get_rank();
        params.axis = convert_axis(primitive->axis, rank);
        params.exclusive = primitive->exclusive;
        params.reverse = primitive->reverse;
        return {params, optional_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
    }
};

namespace detail {

attach_cum_sum_impl::attach_cum_sum_impl() {
    implementation_map<cum_sum>::add(impl_types::ocl, shape_types::any, typed_primitive_impl_ocl<cum_sum>::create<cum_sum_impl>, {
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
        std::make_tuple(data_types::i64, format::bfyx),
        std::make_tuple(data_types::i64, format::bfzyx),
        std::make_tuple(data_types::i64, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfwzyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::cum_sum_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cum_sum)
