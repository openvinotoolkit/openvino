/*
// Copyright (c) 2016-2019 Intel Corporation
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

#include "concatenation_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "events_waiter.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "concatenation/concatenation_kernel_selector.h"
#include "concatenation/concatenation_kernel_base.h"

#include <initializer_list>

namespace cldnn {
namespace gpu {

namespace {
kernel_selector::concat_axis convert_axis(concatenation::concatenation_axis axis) {
    switch (axis) {
        case concatenation::along_x:
            return kernel_selector::concat_axis::X;
        case concatenation::along_y:
            return kernel_selector::concat_axis::Y;
        case concatenation::along_z:
            return kernel_selector::concat_axis::Z;
        case concatenation::along_w:
            return kernel_selector::concat_axis::W;
        case concatenation::along_f:
            return kernel_selector::concat_axis::FEATURE;
        case concatenation::along_b:
            return kernel_selector::concat_axis::BATCH;
        default:
            return kernel_selector::concat_axis::X;
    }
}
}  // namespace

struct concatenation_gpu : typed_primitive_gpu_impl<concatenation> {
    using parent = typed_primitive_gpu_impl<concatenation>;

    concatenation_gpu(const concatenation_node& arg, const kernel_selector::kernel_data& kd) : parent(arg, kd) {
        if (!_outer.can_be_optimized()) {
            CLDNN_ERROR_NOT_EQUAL(_outer.id(),
                                  "Input count",
                                  _outer.inputs_count(),
                                  "kds size",
                                  kd.kernels.size(),
                                  "Error - not enough kernels for concatenation");
        }
    }

protected:
    bool optimized_out(concatenation_inst& instance) const override {
        return parent::optimized_out(instance) || _outer.can_be_optimized();
    }

public:
    static primitive_impl* create(const concatenation_node& arg) {
        if (arg.can_be_optimized()) {
            return new concatenation_gpu(arg, {});
        }

        auto concat_params = get_default_params<kernel_selector::concatenation_params>(arg);
        auto concat_optional_params =
            get_default_optional_params<kernel_selector::concatenation_optional_params>(arg.get_program());
        auto axis = arg.get_primitive()->axis;

        concat_params.inputs.resize(arg.inputs_count());
        for (size_t i = 0; i < arg.inputs_count(); ++i) {
            const layout& input_layout = arg.input(i).get_output_layout();
            concat_params.inputs[i] = convert_data_tensor(input_layout);
        }

        concat_params.axis = convert_axis(axis);
        concat_optional_params.kernelPerInput = true;

        auto& kernel_selector = kernel_selector::concatenation_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(concat_params, concat_optional_params);
        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        concatenation_gpu* concat = new concatenation_gpu(arg, best_kernels[0]);

        return concat;
    }
};

namespace {
struct attach {
    attach() {
        implementation_map<concatenation>::add({
            {std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::i8, format::yxfb), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::u8, format::yxfb), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::i32, format::yxfb), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::i64, format::yxfb), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::i64, format::bfyx), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::i8, format::byxf), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::u8, format::byxf), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::i32, format::byxf), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::i64, format::byxf), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::f32, format::fyxb), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::f16, format::fyxb), concatenation_gpu::create},
            // 5D
            {std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::i8, format::bfzyx), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::i32, format::bfzyx), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::i64, format::bfzyx), concatenation_gpu::create},
            // block f16 format
            {std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx_f16), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx_f16), concatenation_gpu::create},
            // MMAD
            {std::make_tuple(engine_types::ocl, data_types::i8, format::byxf_af32), concatenation_gpu::create},
            // 6D
            {std::make_tuple(engine_types::ocl, data_types::f32, format::bfwzyx), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::f16, format::bfwzyx), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::u8, format::bfwzyx), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::i8, format::bfwzyx), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::i32, format::bfwzyx), concatenation_gpu::create},
            {std::make_tuple(engine_types::ocl, data_types::i64, format::bfwzyx), concatenation_gpu::create},
        });
    }
    ~attach() {}
};
}  // namespace

attach attach_impl;

}  // namespace gpu
}  // namespace cldnn
