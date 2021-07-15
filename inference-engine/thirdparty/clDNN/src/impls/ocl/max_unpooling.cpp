// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "max_unpooling_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "cldnn/runtime/error_handler.hpp"
#include "network_impl.h"
#include "kernel_selector_helper.h"
#include "max_unpooling/max_unpooling_kernel_selector.h"
#include "max_unpooling/max_unpooling_kernel_base.h"
#include <vector>

namespace cldnn {
namespace ocl {

struct max_unpooling_impl : typed_primitive_impl_ocl<max_unpooling> {
    using parent = typed_primitive_impl_ocl<max_unpooling>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<max_unpooling_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(typed_primitive_inst<max_unpooling>& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);
        args.inputs.push_back(instance.dep_memory_ptr(1));
        return args;
    }

public:
    event::ptr execute_impl(const std::vector<event::ptr>& events, max_unpooling_inst& instance) override {
        // clear output buffer
        std::vector<event::ptr> tmp_events(events);
        auto& stream = instance.get_network().get_stream();
        auto ev = instance.output_memory().fill(stream);
        tmp_events.push_back(ev);
        return parent::execute_impl(tmp_events, instance);
    }

    static primitive_impl* create(const max_unpooling_node& arg) {
        auto max_unpooling_params = get_default_params<kernel_selector::max_unpooling_params>(arg);
        auto max_unpooling_optional_params =
            get_default_optional_params<kernel_selector::max_unpooling_optional_params>(arg.get_program());

        max_unpooling_params.inputs.push_back(convert_data_tensor(arg.argmax().get_output_layout()));

        auto& kernel_selector = kernel_selector::max_unpooling_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(max_unpooling_params, max_unpooling_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto max_unpool = new max_unpooling_impl(arg, best_kernels[0]);

        return max_unpool;
    }
};

namespace detail {

attach_max_unpooling_impl::attach_max_unpooling_impl() {
    implementation_map<max_unpooling>::add(impl_types::ocl, max_unpooling_impl::create, {
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
