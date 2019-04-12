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

#include "strided_slice_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "strided_slice/strided_slice_kernel_ref.h"
#include "strided_slice/strided_slice_kernel_selector.h"
#include "error_handler.h"
#include "data_inst.h"

using namespace cldnn;

namespace cldnn
{
namespace gpu
{

struct strided_slice_gpu : typed_primitive_gpu_impl<strided_slice>
{
    using parent = typed_primitive_gpu_impl<strided_slice>;
    using parent::parent;
public:
    static primitive_impl* create(const strided_slice_node& arg)
    {
        auto strided_slice_params = get_default_params<kernel_selector::strided_slice_params>(arg);
        auto strided_slice_optional_params = get_default_optional_params<kernel_selector::strided_slice_optional_params>(arg.get_program());
        const int32_t numberOfDims = 4;

        auto complete_strided_slice_params = [&](std::vector<int32_t>& param) {
            for (size_t i = param.size(); i < numberOfDims; ++i)
                param.push_back(1);
        };

        auto completeStridedSliceMasks = [&](std::vector<uint8_t>& mask) {
            for (size_t i = mask.size(); i < numberOfDims; ++i)
                mask.push_back(0);
        };

        // Getting data from constant inputs. There are 3 args: Begin, End, Stride
        for (size_t i = 1; i < arg.get_dependencies().size(); ++i) {
            auto& input = arg.get_dependency(i).as<data>();
            auto& mem = input.get_attached_memory();
            int32_t* data = static_cast<int32_t*>(mem.lock());
            std::vector<int32_t> vData = std::vector<int32_t>(data, data + input.get_output_layout().count());
            complete_strided_slice_params(vData);
            strided_slice_params.striding_params.push_back(vData);
            mem.unlock();
        }

        strided_slice_params.end_mask = arg.get_primitive()->end_mask;
        completeStridedSliceMasks(strided_slice_params.end_mask);
        strided_slice_params.begin_mask = arg.get_primitive()->begin_mask;
        completeStridedSliceMasks(strided_slice_params.begin_mask);
        strided_slice_params.new_axis_mask = arg.get_primitive()->new_axis_mask;
        strided_slice_params.shrink_axis_mask = arg.get_primitive()->shrink_axis_mask;
        completeStridedSliceMasks(strided_slice_params.shrink_axis_mask);

        auto& kernel_selector = kernel_selector::strided_slice_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(strided_slice_params, strided_slice_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto strided_slice = new strided_slice_gpu(arg, best_kernels[0]);

        return strided_slice;
    }
};

namespace
{
    struct attach
    {
        attach()
        {
            auto val_fw = strided_slice_gpu::create;
            implementation_map<strided_slice>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
            implementation_map<strided_slice>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
        }
        ~attach() = default;
    };
    attach attach_impl;
}
} //namespace gpu
} //namespace cldnn
