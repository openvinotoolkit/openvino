/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "max_unpooling_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "network_impl.h"
#include "kernel_selector_helper.h"
#include "max_unpooling/max_unpooling_kernel_selector.h"
#include "max_unpooling/max_unpooling_kernel_base.h"
#include <vector>

namespace cldnn {
namespace gpu {

struct max_unpooling_gpu : typed_primitive_gpu_impl<max_unpooling> {
    using parent = typed_primitive_gpu_impl<max_unpooling>;
    using parent::parent;

protected:
    kernel::kernel_arguments_data get_arguments(typed_primitive_inst<max_unpooling>& instance,
                                                        int32_t split) const override {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, split);
        args.inputs.push_back((memory_impl::cptr) &instance.dep_memory(1));
        return args;
    }

public:
    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, max_unpooling_inst& instance) override {
        // clear output buffer
        std::vector<event_impl::ptr> tmp_events(events);
        auto ev = instance.get_network().get_engine().create_user_event(instance.get_network().get_stream_id(), false);
        instance.output_memory().fill(0, ev);
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

        auto max_unpool = new max_unpooling_gpu(arg, best_kernels[0]);

        return max_unpool;
    }
};

namespace detail {

attach_max_unpooling_gpu::attach_max_unpooling_gpu() {
    implementation_map<max_unpooling>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb),
                                           max_unpooling_gpu::create);
    implementation_map<max_unpooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb),
                                           max_unpooling_gpu::create);
    implementation_map<max_unpooling>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                           max_unpooling_gpu::create);
    implementation_map<max_unpooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx),
                                           max_unpooling_gpu::create);
    implementation_map<max_unpooling>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx),
                                           max_unpooling_gpu::create);
    implementation_map<max_unpooling>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::yxfb),
                                           max_unpooling_gpu::create);
    implementation_map<max_unpooling>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf),
                                           max_unpooling_gpu::create);
    implementation_map<max_unpooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf),
                                           max_unpooling_gpu::create);
    implementation_map<max_unpooling>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::byxf),
                                           max_unpooling_gpu::create);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
