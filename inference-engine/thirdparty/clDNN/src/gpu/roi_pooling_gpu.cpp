/*
// Copyright (c) 2017-2018 Intel Corporation
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

namespace cldnn {
namespace gpu {

namespace {
kernel_selector::pool_type cldnn_2_pool_type(pooling_mode mode) {
    switch (mode) {
        case pooling_mode::max:
            return kernel_selector::pool_type::MAX;
        case pooling_mode::average:
            return kernel_selector::pool_type::AVG;
        case pooling_mode::average_no_padding:
            return kernel_selector::pool_type::AVG;
        case pooling_mode::bilinear:
            return kernel_selector::pool_type::BILINEAR;
        case pooling_mode::deformable_bilinear:
            return kernel_selector::pool_type::DEFORMABLE_BILINEAR;
        default:
            assert(0);
            return kernel_selector::pool_type::MAX;
    }
}
}  // namespace

struct roi_pooling_gpu : typed_primitive_gpu_impl<roi_pooling> {
    using parent = typed_primitive_gpu_impl<roi_pooling>;
    using parent::parent;

protected:
    kernel::kernel_arguments_data get_arguments(typed_primitive_inst<roi_pooling>& instance,
                                                        int32_t) const override {
        kernel::kernel_arguments_data args;

        if (instance.argument.mode == pooling_mode::deformable_bilinear && !instance.argument.no_trans)
            args.inputs = {
                (memory_impl::cptr) &instance.input_memory(),
                (memory_impl::cptr) &instance.rois_memory(),
                (memory_impl::cptr) &instance.trans_memory()};
        else
            args.inputs = {(memory_impl::cptr) &instance.input_memory(), (memory_impl::cptr) &instance.rois_memory()};

        args.output = (memory_impl::cptr) &instance.output_memory();

        return args;
    }

public:
    static primitive_impl* create(const roi_pooling_node& arg) {
        const auto& input_layout = arg.input().get_output_layout();
        const auto& output_layout = arg.get_output_layout();
        const auto& rois_layout = arg.rois().get_output_layout();
        const auto& primitive = arg.get_primitive();

        const auto padding_filling_value = output_layout.data_padding.filling_value();

        CLDNN_ERROR_NOT_EQUAL(arg.id(),
                              "roi_pooling padding filling value",
                              padding_filling_value,
                              "padding mode",
                              0.0f,
                              "Unknown padding mode in roi_pooling.");
        CLDNN_ERROR_NOT_PROPER_FORMAT(arg.id(),
                                      "Input_layout.format",
                                      input_layout.format.value,
                                      "output_layout.format",
                                      output_layout.format);

        auto roi_params = get_default_params<kernel_selector::roi_pooling_params>(arg);
        auto roi_optional_params =
            get_default_optional_params<kernel_selector::roi_pooling_optional_params>(arg.get_program());

        const auto roi_bfyx = convert_data_tensor(rois_layout);
        const auto roi_bf = roi_bfyx.FlattenFeatureAndSpatials();
        roi_params.inputs.push_back(roi_bf);
        if (primitive->mode == pooling_mode::deformable_bilinear && !primitive->no_trans)
            roi_params.inputs.push_back(convert_data_tensor(arg.trans().get_output_layout()));
        roi_params.mode = cldnn_2_pool_type(primitive->mode);
        roi_params.position_sensitive = primitive->position_sensitive;
        roi_params.pooled_width = primitive->pooled_width;
        roi_params.pooled_height = primitive->pooled_height;
        roi_params.spatial_scale = primitive->spatial_scale;
        roi_params.spatial_bins_x = primitive->spatial_bins_x;
        roi_params.spatial_bins_y = primitive->spatial_bins_y;
        roi_params.trans_std = primitive->trans_std;
        roi_params.no_trans = primitive->no_trans;
        roi_params.part_size = primitive->part_size;
        roi_params.group_size = primitive->group_size;

        auto& kernel_selector = kernel_selector::roi_pooling_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(roi_params, roi_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto roi_pool = new roi_pooling_gpu(arg, best_kernels[0]);

        return roi_pool;
    }
};

namespace detail {

attach_roi_pooling_gpu::attach_roi_pooling_gpu() {
    implementation_map<roi_pooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx),
                                         roi_pooling_gpu::create);
    implementation_map<roi_pooling>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                         roi_pooling_gpu::create);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
