// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "average_unpooling_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "cldnn/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "average_unpooling/average_unpooling_kernel_selector.h"
#include "average_unpooling/average_unpooling_kernel_base.h"

namespace cldnn {
namespace ocl {

struct average_unpooling_impl : typed_primitive_impl_ocl<average_unpooling> {
    using parent = typed_primitive_impl_ocl<average_unpooling>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<average_unpooling_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(typed_primitive_inst<average_unpooling>& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);
        return args;
    }

public:
    static primitive_impl* create(const average_unpooling_node& arg) {
        auto average_unpooling_params = get_default_params<kernel_selector::average_unpooling_params>(arg);
        auto average_unpooling_optional_params =
            get_default_optional_params<kernel_selector::average_unpooling_optional_params>(arg.get_program());
        auto& params = average_unpooling_params;

        auto primitive = arg.get_primitive();
        auto stride = primitive->stride;

        params.unpoolSize = {
            (uint32_t)primitive->size.spatial[0],
            (uint32_t)primitive->size.spatial[1],
        };

        params.unpoolStride = {(uint32_t)stride.spatial[0], (uint32_t)stride.spatial[1]};

        auto& kernel_selector = kernel_selector::average_unpooling_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(average_unpooling_params, average_unpooling_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto average_unpool = new average_unpooling_impl(arg, best_kernels[0]);

        return average_unpool;
    }
};

namespace detail {

attach_average_unpooling_impl::attach_average_unpooling_impl() {
    implementation_map<average_unpooling>::add(impl_types::ocl, average_unpooling_impl::create, {
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
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
