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

#include "pyramid_roi_align_inst.h"

#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "pyramid_roi_align/pyramid_roi_align_kernel_selector.h"
#include "pyramid_roi_align/pyramid_roi_align_kernel_base.h"
#include "error_handler.h"
#include "pyramid_roi_align_inst.h"
#include "network_impl.h"


#define DEPTH_OF_FEATURE_MAP            4
#define NUM_COORDINATES                 4
#define META_OFFSET_X                   4
#define META_OFFSET_Y                   5

namespace cldnn {  namespace gpu {

struct pyramid_roi_align_gpu : typed_primitive_gpu_impl<pyramid_roi_align>
{
    using parent = typed_primitive_gpu_impl<pyramid_roi_align>;
    using parent::parent;

    static primitive_impl* create(const pyramidROIAlign_node& arg)
    {
        auto pyramidROIAlign_params = get_default_params<kernel_selector::PyramidROIAlign_params>(arg, 1);
        auto pyramidROIAlign_optional_params = get_default_optional_params<kernel_selector::PyramidROIAlign_optional_params>(arg.get_program());

        pyramidROIAlign_params.inputs.push_back(convert_data_tensor(arg.image_meta().get_output_layout()));
        pyramidROIAlign_params.inputs.push_back(convert_data_tensor(arg.P2().get_output_layout()));
        pyramidROIAlign_params.inputs.push_back(convert_data_tensor(arg.P3().get_output_layout()));
        pyramidROIAlign_params.inputs.push_back(convert_data_tensor(arg.P4().get_output_layout()));
        pyramidROIAlign_params.inputs.push_back(convert_data_tensor(arg.P5().get_output_layout()));
        pyramidROIAlign_params.inputs.push_back(convert_data_tensor(arg.pool_size().get_output_layout()));


        auto& kernel_selector = kernel_selector::PyramidROIAlign_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(pyramidROIAlign_params, pyramidROIAlign_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        return new pyramid_roi_align_gpu(arg, best_kernels[0]);
    }

};

namespace {
    struct attach {
        attach() {
            auto val_fw = pyramid_roi_align_gpu::create;
            implementation_map<pyramid_roi_align>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
            implementation_map<pyramid_roi_align>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
        }
        ~attach() = default;

    };

    attach attach_impl;

}
}}