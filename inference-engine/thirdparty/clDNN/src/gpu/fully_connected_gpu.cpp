/*
// Copyright (c) 2019-2020 Intel Corporation
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

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "fully_connected_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "fully_connected/fully_connected_kernel_selector.h"
#include "fully_connected/fully_connected_params.h"

#include "network_impl.h"
#include "error_handler.h"
#include "kernel_runner.h"

#include "api/reorder.hpp"
#include "api/input_layout.hpp"
#include <memory>

namespace cldnn {
namespace gpu {

struct fully_connected_gpu : typed_primitive_gpu_impl<fully_connected> {
    using parent = typed_primitive_gpu_impl<fully_connected>;
    using parent::parent;

protected:
    kernel::kernel_arguments_data get_arguments(typed_primitive_inst<fully_connected>& instance,
                                                        int32_t split) const override {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, split);

        args.weights = (memory_impl::cptr) &instance.weights_memory();
        args.bias = (memory_impl::cptr) (instance.bias_term() ? &instance.bias_memory() : nullptr);

        return args;
    }

public:
    static primitive_impl* create(const fully_connected_node& arg) {
        auto fc_params = get_weights_bias_default_params<kernel_selector::fully_connected_params>(arg);
        auto fc_optional_params =
            get_default_weights_bias_optional_params<kernel_selector::fully_connected_optional_params>(
                arg.get_program());
        fc_optional_params.allowInputReordering = true;

        fc_params.output = fc_params.output.FlattenFeatureAndSpatials();

        const auto primitive = arg.get_primitive();

        if (arg.get_output_layout().data_type == data_types::i8 ||
            arg.get_output_layout().data_type == data_types::u8) {
            fc_params.quantization = kernel_selector::QuantizationType::SYMMETRIC;
        } else {
            fc_params.quantization = kernel_selector::QuantizationType::NONE;
        }

        fc_optional_params.tuningParams.runner =
            std::make_shared<gpu::kernel_runner>(arg.get_program().get_engine(), arg.get_program().get_id(), true);

        auto& kernel_selector = kernel_selector::fully_connected_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(fc_params, fc_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto fc = new fully_connected_gpu(arg, best_kernels[0]);

        return fc;
    }
};

namespace detail {

attach_fully_connected_gpu::attach_fully_connected_gpu() {
    auto val_fw = fully_connected_gpu::create;

    implementation_map<fully_connected>::add({
        {std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), val_fw},
        {std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), val_fw},
        {std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw},
        {std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw},
        {std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), val_fw},
        {std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), val_fw},
        {std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), val_fw},
        {std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), val_fw},
        // MMAD
        {std::make_tuple(engine_types::ocl, data_types::i8, format::byxf_af32), val_fw},
        {std::make_tuple(engine_types::ocl, data_types::i8, format::fs_bs_yx_bsv4_fsv32), val_fw},
        {std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv32), val_fw},
        {std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv32), val_fw},
        // IMAD
        {std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv4), val_fw},
        {std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv4), val_fw},
        {std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_yx_fsv4), val_fw},
        {std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv16), val_fw},
        {std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv16), val_fw},
        {std::make_tuple(engine_types::ocl, data_types::f32, format::bs_fs_yx_bsv16_fsv16), val_fw},
        {std::make_tuple(engine_types::ocl, data_types::i8, format::bs_fs_yx_bsv16_fsv16), val_fw},
        {std::make_tuple(engine_types::ocl, data_types::u8, format::bs_fs_yx_bsv16_fsv16), val_fw},
        // fs_b_yx_fsv32
        {std::make_tuple(engine_types::ocl, data_types::f16, format::fs_b_yx_fsv32), val_fw},
    });
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
