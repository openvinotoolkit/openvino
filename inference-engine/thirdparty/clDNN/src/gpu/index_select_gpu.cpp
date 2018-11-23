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


#include "index_select_inst.h"

#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "index_select/index_select_kernel_selector.h"
#include "index_select/index_select_kernel_base.h"
#include "error_handler.h"

namespace cldnn { namespace gpu {

namespace
{
    inline kernel_selector::IndexSelectAxis convect_to_index_select_axis(index_select_axis_name axis)
    {
        switch (axis)
        {
        case index_select_axis_name::along_b:  return kernel_selector::IndexSelectAxis::BATCH;
        case index_select_axis_name::along_f:  return kernel_selector::IndexSelectAxis::FEATURE;
        case index_select_axis_name::along_x:  return kernel_selector::IndexSelectAxis::X;
        case index_select_axis_name::along_y: return kernel_selector::IndexSelectAxis::Y;
        default:
            return kernel_selector::IndexSelectAxis::BATCH;
        }
    }
}

struct index_select_gpu : typed_primitive_gpu_impl<index_select>
{
    using parent = typed_primitive_gpu_impl<index_select>;
    using parent::parent;

    static primitive_impl* create(const index_select_node& arg)
    { 
        auto index_select_params          = get_default_params<kernel_selector::index_select_params>(arg, 1);
        auto index_select_optional_params = get_default_optional_params<kernel_selector::index_select_optional_params>(arg.get_program());

        index_select_params.inputs.push_back(convert_data_tensor(arg.indices().get_output_layout()));
        index_select_params.axis = convect_to_index_select_axis(arg.get_axis());
        
        auto& kernel_selector = kernel_selector::index_select_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(index_select_params, index_select_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        return new index_select_gpu(arg, best_kernels[0]);
    }
};

namespace {
    struct attach {
        attach() {
            auto val_fw = index_select_gpu::create;
            implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
            implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
            implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), val_fw);
            implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), val_fw);
            implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), val_fw);

            implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), val_fw);
            implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), val_fw);
            implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::yxfb), val_fw);
            implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::yxfb), val_fw);
            implementation_map<index_select>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::yxfb), val_fw);
        }
        ~attach() = default;
    };

    attach attach_impl;

}
} }
