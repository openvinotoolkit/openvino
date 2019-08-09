/*
// Copyright (c) 2019 Intel Corporation
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

#include "quantize_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "quantize/quantize_kernel_selector.h"
#include "quantize/quantize_kernel_ref.h"
#include "error_handler.h"

using namespace cldnn;

namespace cldnn {
namespace gpu {

struct quantize_gpu : typed_primitive_gpu_impl<quantize> {
    using parent = typed_primitive_gpu_impl<quantize>;
    using parent::parent;

public:
    static primitive_impl* create(const quantize_node& arg) {
        auto quantize_params = get_default_params<kernel_selector::quantize_params>(arg);
        auto quantize_optional_params =
            get_default_optional_params<kernel_selector::quantize_optional_params>(arg.get_program());

        quantize_params.levels = arg.get_primitive()->levels;
        quantize_params.packed_binary_output = arg.get_packed_binary_output();

        for (size_t i = 1; i < arg.inputs_count(); i++) {
            quantize_params.inputs.push_back(convert_data_tensor(arg.input(i).get_output_layout()));
        }
        const auto& output_layout = arg.get_output_layout();
        quantize_params.output = convert_data_tensor(output_layout);

        auto& kernel_selector = kernel_selector::quantize_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(quantize_params, quantize_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto quantize = new quantize_gpu(arg, best_kernels[0]);

        return quantize;
    }
};

namespace {
struct attach {
    attach() {
        auto val_fw = quantize_gpu::create;

        implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), val_fw);
        implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
        implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    }
    ~attach() {}
};
attach attach_impl;
}  // namespace
}  // namespace gpu
}  // namespace cldnn
