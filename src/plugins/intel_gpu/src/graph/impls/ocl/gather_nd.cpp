// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_nd_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "gather/gather_nd_kernel_selector.h"
#include "gather/gather_nd_kernel_ref.h"

using namespace cldnn;

namespace cldnn {
namespace ocl {

struct gather_nd_impl : typed_primitive_impl_ocl<gather_nd> {
    using parent = typed_primitive_impl_ocl<gather_nd>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_nd_impl>(*this);
    }

    static std::unique_ptr<primitive_impl> create(const gather_nd_node& arg, const kernel_impl_params& impl_param) {
        const auto& prim = arg.get_primitive();
        auto gather_nd_params = get_default_params<kernel_selector::gather_nd_params>(impl_param);
        auto gather_nd_optional_params =
            get_default_optional_params<kernel_selector::gather_nd_optional_params>(arg.get_program());

        gather_nd_params.indices_rank = prim->indices_rank;
        gather_nd_params.batch_dims = prim->batch_dims;
        gather_nd_params.batch_merged_output = prim->batch_merged_output;

        gather_nd_params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[1]));

        auto& kernel_selector = kernel_selector::gather_nd_kernel_selector::Instance();
        auto best_kernel = kernel_selector.get_best_kernel(gather_nd_params, gather_nd_optional_params);

        return make_unique<gather_nd_impl>(arg, best_kernel);
    }
};

namespace detail {

attach_gather_nd_impl::attach_gather_nd_impl() {
    implementation_map<gather_nd>::add(impl_types::ocl, gather_nd_impl::create, {
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

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::gather_nd_impl, cldnn::object_type::GATHER_ND_IMPL)