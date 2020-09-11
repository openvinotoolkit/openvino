/*
// Copyright (c) 2020 Intel Corporation
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

#include "ctc_greedy_decoder_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "ctc_greedy_decoder/ctc_greedy_decoder_kernel_selector.h"
#include "ctc_greedy_decoder/ctc_greedy_decoder_kernel_base.h"

#include <algorithm>

using namespace cldnn;

namespace cldnn {
namespace gpu {

struct ctc_greedy_decoder_gpu : typed_primitive_gpu_impl<ctc_greedy_decoder> {
    using parent = typed_primitive_gpu_impl<ctc_greedy_decoder>;
    using parent::parent;

public:
    static primitive_impl* create(const ctc_greedy_decoder_node& arg) {
        auto ctc_gd_params = get_default_params<kernel_selector::ctc_greedy_decoder_params>(arg);
        auto ctc_gd_optional_params = get_default_optional_params<kernel_selector::ctc_greedy_decoder_optional_params>(arg.get_program());

        ctc_gd_params.inputs.push_back(
            convert_data_tensor(arg.seq_indicators().get_output_layout()));
        ctc_gd_params.merge_repeated = arg.get_primitive()->ctc_merge_repeated;

        auto& kernel_selector = kernel_selector::ctc_greedy_decoder_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(
            ctc_gd_params, ctc_gd_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto grn = new ctc_greedy_decoder_gpu(arg, best_kernels[0]);

        return grn;
    }
};

namespace detail {

attach_ctc_greedy_decoder_gpu::attach_ctc_greedy_decoder_gpu() {
    implementation_map<ctc_greedy_decoder>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), ctc_greedy_decoder_gpu::create);
    implementation_map<ctc_greedy_decoder>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), ctc_greedy_decoder_gpu::create);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
