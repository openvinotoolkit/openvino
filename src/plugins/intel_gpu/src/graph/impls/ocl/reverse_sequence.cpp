// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "reverse_sequence_inst.h"
#include "reverse_sequence/reverse_sequence_kernel_selector.h"
#include "reverse_sequence/reverse_sequence_kernel_ref.h"

namespace cldnn {
namespace ocl {
struct reverse_sequence_impl : typed_primitive_impl_ocl<reverse_sequence> {
    using parent = typed_primitive_impl_ocl<reverse_sequence>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::reverse_sequence_kernel_selector;
    using kernel_params_t = kernel_selector::reverse_sequence_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::reverse_sequence_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<reverse_sequence_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<reverse_sequence>();
        auto params = get_default_params<kernel_selector::reverse_sequence_params>(impl_param);

        params.seq_axis = primitive->seq_axis;
        params.batch_axis = primitive->batch_axis;

        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));

        return params;
    }
};

namespace detail {

attach_reverse_sequence_impl::attach_reverse_sequence_impl() {
    implementation_map<reverse_sequence>::add(impl_types::ocl, typed_primitive_impl_ocl<reverse_sequence>::create<reverse_sequence_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),

        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::reverse_sequence_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::reverse_sequence)
