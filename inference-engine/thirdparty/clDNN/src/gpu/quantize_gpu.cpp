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

protected:
    kernel::kernel_arguments_data get_arguments(typed_primitive_inst<quantize>& instance,
                                                int32_t) const override {
        kernel::kernel_arguments_data args;

        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back((memory_impl::cptr) &instance.input_memory(i));
        }
        if (instance.node.get_scale_shift_opt()) {
            if (instance.node.get_dependencies().size() == 9) {
                args.inputs.push_back((memory_impl::cptr) &instance.dep_memory(5));
                args.inputs.push_back((memory_impl::cptr) &instance.dep_memory(6));
                args.inputs.push_back((memory_impl::cptr) &instance.dep_memory(7));
                args.inputs.push_back((memory_impl::cptr) &instance.dep_memory(8));
            }
        }
        args.output = (memory_impl::cptr) &instance.output_memory();
        return args;
    }

public:
    static primitive_impl* create(const quantize_node& arg) {
        auto quantize_params = get_default_params<kernel_selector::quantize_params>(arg);
        auto quantize_optional_params =
            get_default_optional_params<kernel_selector::quantize_optional_params>(arg.get_program());

        quantize_params.levels = arg.get_levels();
        quantize_params.packed_binary_output = arg.get_packed_binary_output();
        quantize_params.scale_shift_opt = arg.get_scale_shift_opt();
        quantize_params.has_post_scale = arg.get_need_post_scale();
        quantize_params.has_post_shift = arg.get_need_post_shift();
        quantize_params.has_pre_shift = arg.get_need_pre_shift();
        quantize_params.has_clamp = arg.get_need_clamp();

        quantize_params.per_tensor_input_range = arg.get_per_tensor_input_range();
        quantize_params.per_tensor_input_scale = arg.get_per_tensor_input_scale();
        quantize_params.per_tensor_input_shift = arg.get_per_tensor_input_shift();
        quantize_params.per_tensor_output_scale = arg.get_per_tensor_output_scale();
        quantize_params.per_tensor_output_shift = arg.get_per_tensor_output_shift();

        quantize_params.in_lo = arg.get_input_lo_val();
        quantize_params.in_hi = arg.get_input_hi_val();
        quantize_params.in_scale = arg.get_input_scale_val();
        quantize_params.in_shift = arg.get_input_shift_val();
        quantize_params.out_scale = arg.get_output_scale_val();
        quantize_params.out_shift = arg.get_output_shift_val();

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

namespace detail {

attach_quantize_gpu::attach_quantize_gpu() {
    auto val_fw = quantize_gpu::create;

    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::fs_b_yx_fsv32), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::fs_b_yx_fsv32), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::fs_b_yx_fsv32), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::fs_b_yx_fsv32), val_fw);

    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_yx_fsv16), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_yx_fsv16), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv16), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv16), val_fw);

    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::byxf), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::byxf), val_fw);

    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_yx_fsv4), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_yx_fsv4), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv4), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv4), val_fw);

    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_yx_fsv32), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_yx_fsv32), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv32), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv32), val_fw);

    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_zyx_fsv32), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_zyx_fsv32), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_zyx_fsv32), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_zyx_fsv32), val_fw);

    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bs_fs_yx_bsv16_fsv16), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bs_fs_yx_bsv16_fsv16), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bs_fs_yx_bsv16_fsv16), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bs_fs_yx_bsv16_fsv16), val_fw);

    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bs_fs_zyx_bsv16_fsv16), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bs_fs_zyx_bsv16_fsv16), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bs_fs_zyx_bsv16_fsv16), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bs_fs_zyx_bsv16_fsv16), val_fw);

    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), val_fw);

    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::byxf), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::byxf), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::byxf), val_fw);

    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), val_fw);

    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfzyx), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfzyx), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfzyx), val_fw);

    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_zyx_fsv16), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_zyx_fsv16), val_fw);

    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bs_fs_yx_bsv16_fsv16), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bs_fs_yx_bsv16_fsv16), val_fw);
    implementation_map<quantize>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bs_fs_yx_bsv16_fsv16), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
