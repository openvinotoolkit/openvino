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

#include "embed_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "kernel_runner.h"

#include "embed/embed_kernel_selector.h"
#include "embed/embed_params.h"

#include "api/CPP/input_layout.hpp"
#include <vector>

namespace cldnn {
namespace gpu {

struct embed_gpu : typed_primitive_gpu_impl<embed> {
    using parent = typed_primitive_gpu_impl<embed>;
    memory_impl::cptr new_input_mem;

    embed_gpu(const embed_node& arg, const kernel_selector::kernel_data& kd) : parent(arg, kd) {}

    kernel::kernel_arguments_data get_arguments(typed_primitive_inst<embed>& instance, int32_t) const override {
        kernel::kernel_arguments_data args;
        args.inputs = {new_input_mem};
        args.output = (memory_impl::cptr) &instance.output_memory();
        args.weights = (memory_impl::cptr) &instance.weights_memory();
        args.bias = (memory_impl::cptr) (instance.bias_term() ? &instance.bias_memory() : nullptr);

        return args;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, embed_inst& instance) override {
        std::vector<event_impl::ptr> tmp_events(events);
        new_input_mem = (memory_impl::cptr) &instance.input_memory();

        return parent::execute_impl(tmp_events, instance);
    }

    static primitive_impl* create(const embed_node& arg) {
        auto embed_params = get_weights_bias_default_params<kernel_selector::embed_params>(arg);
        auto embed_optional_params =
            get_default_weights_bias_optional_params<kernel_selector::embed_optional_params>(arg.get_program());
        embed_params.output = embed_params.output.FlattenFeatureAndSpatials();

        auto& kernel_selector = kernel_selector::embed_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(embed_params, embed_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto embed_node = new embed_gpu(arg, best_kernels[0]);

        return embed_node;
    }
};

namespace {
struct attach {
    attach() {
        implementation_map<embed>::add(
            {{std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), embed_gpu::create},
             {std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), embed_gpu::create}});
    }
    ~attach() {}
};

attach attach_impl;
}  // namespace

}  // namespace gpu
}  // namespace cldnn
