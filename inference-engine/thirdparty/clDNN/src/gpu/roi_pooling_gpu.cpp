/*
// Copyright (c) 2017 Intel Corporation
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

#include "roi_pooling_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "roi_pooling/roi_pooling_kernel_selector.h"
#include "roi_pooling/roi_pooling_kernel_ref.h"

namespace cldnn { namespace gpu {

namespace {
    static inline bool hasSingleBatchOutput(const program_node & node)
    {
        const auto & batch = node.get_output_layout().size.batch;

        return batch.empty() || (batch.size() == 1 && batch[0] == 1);
    }
}


struct roi_pooling_gpu : typed_primitive_gpu_impl<roi_pooling>
{
    using parent = typed_primitive_gpu_impl<roi_pooling>;
    using parent::parent;

protected:

    virtual kernel::kernel_arguments_data get_arguments(typed_primitive_inst<roi_pooling>& instance, int32_t) const override
    {
        kernel::kernel_arguments_data args;

        args.inputs = { &instance.input_memory(), &instance.rois_memory() };
        args.output = &instance.output_memory();

        return args;
    }

public:

    static primitive_impl* create(const roi_pooling_node& arg)
    {
        const auto& input_layout    = arg.input().get_output_layout();
        const auto& output_layout   = arg.get_output_layout();
        const auto& rois_layout     = arg.rois().get_output_layout();
        const auto& primitive       = arg.get_primitive();

        const auto padding_filling_value = output_layout.data_padding.filling_value();

        CLDNN_ERROR_NOT_EQUAL(arg.id(), "roi_pooling padding filling value", padding_filling_value, "padding mode", 0.0f, "Unknown padding mode in roi_pooling.");
        CLDNN_ERROR_NOT_PROPER_FORMAT(arg.id(), "Input_layout.format", input_layout.format.value, "output_layout.format", output_layout.format);

        auto group_sz = primitive->group_sz;
        auto in_feat = input_layout.get_buffer_size().feature[0];
        auto out_feat = output_layout.get_buffer_size().feature[0];

        CLDNN_ERROR_LESS_THAN(arg.id(), "Group size", group_sz, "value", 0, "");
        if (group_sz) {
            CLDNN_ERROR_NOT_EQUAL(arg.id(), "input feture map", in_feat, "group_sz * group_sz * out_feat", group_sz * group_sz * out_feat, "");
        }
        CLDNN_ERROR_BOOL(arg.id(), "Batching", !hasSingleBatchOutput(arg.input()), "PS/ RoI Pooling doesn't support batching.");

        auto roi_params = get_default_params<kernel_selector::roi_pooling_params>(arg);
        auto roi_optional_params = get_default_optional_params<kernel_selector::roi_pooling_optional_params>(arg.get_program());

        const auto& out = roi_params.output;
        
        const auto roi_bfyx = convert_data_tensor(rois_layout);
        const auto roi_bf = roi_bfyx.FlattenFeatureAndSpatials();
        roi_params.inputs.push_back(roi_bf);
        roi_params.output = { out.GetDims(), out.GetDType(), kernel_selector::data_layout::brfyx, out.GetViewOffset(), out.PhysicalSize(), out.GetPaddedVal() }; // TOOD: it's an hack - cldnn doesn't support roi pooling with batching
        roi_params.mode         = primitive->mode == pooling_mode::max ? kernel_selector::pool_type::MAX : kernel_selector::pool_type::AVG;
        roi_params.pooledWidth  = primitive->pooled_width;
        roi_params.pooledHeight = primitive->pooled_height;
        roi_params.spatialScale = primitive->spatial_scale;
        roi_params.groupSize    = group_sz;

        auto& kernel_selector = kernel_selector::roi_pooling_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(roi_params, roi_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto roi_pool = new roi_pooling_gpu(arg, best_kernels[0]);

        return roi_pool;
    }
};

namespace {
    struct attach
    {
        attach()
        {
            implementation_map<roi_pooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), roi_pooling_gpu::create);
            implementation_map<roi_pooling>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), roi_pooling_gpu::create);
        }

        ~attach() {}
    };

    attach attach_impl;
}
} }