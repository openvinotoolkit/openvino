// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_types.h"
#include "average_unpooling_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "average_unpooling/average_unpooling_kernel_selector.h"
#include "average_unpooling/average_unpooling_kernel_base.h"

namespace cldnn {
namespace ocl {

struct average_unpooling_impl : typed_primitive_impl_ocl<average_unpooling> {
    using parent = typed_primitive_impl_ocl<average_unpooling>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::average_unpooling_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::average_unpooling_params, kernel_selector::average_unpooling_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<average_unpooling_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<average_unpooling>& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);
        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<average_unpooling>();
        auto params = get_default_params<kernel_selector::average_unpooling_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::average_unpooling_optional_params>(impl_param.get_program());
        auto stride = primitive->stride;

        params.unpoolSize = {
            (uint32_t)primitive->size.spatial[0],
            (uint32_t)primitive->size.spatial[1],
        };

        params.unpoolStride = {(uint32_t)stride.spatial[0], (uint32_t)stride.spatial[1]};

        return {params, optional_params};
    }
};

namespace detail {

attach_average_unpooling_impl::attach_average_unpooling_impl() {
    implementation_map<average_unpooling>::add(impl_types::ocl, typed_primitive_impl_ocl<average_unpooling>::create<average_unpooling_impl>, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::i8, format::yxfb),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::i8, format::byxf),
    });

    impl_hash_key<average_unpooling>::add(typed_primitive_impl_ocl<average_unpooling>::get_impl_key<average_unpooling_impl>);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::average_unpooling_impl)
