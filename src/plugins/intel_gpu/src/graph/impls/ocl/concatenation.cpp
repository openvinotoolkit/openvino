// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concatenation_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "concatenation/concatenation_kernel_selector.h"
#include "concatenation/concatenation_kernel_base.h"

#include <initializer_list>

namespace cldnn {
namespace ocl {

namespace {
kernel_selector::concat_axis convert_axis(int64_t axis, size_t rank) {
    unsigned cldnn_axis = axis >= 0 ? axis : axis + static_cast<int64_t>(rank);
    if (cldnn_axis >= rank)
        IE_THROW() << "Concatenation axis exceeds number of dimensions";

    // Difference in dimension ordering between IE and GPU plugin,
    // reverse spatial dimensions after batch and feature.
    if (cldnn_axis >= 2) {
        auto spatial_axis = cldnn_axis - 2;
        // Default and minimum number of dimensions is 4
        auto spatial_size = std::max<size_t>(rank, 4) - 2;
        cldnn_axis = spatial_size - spatial_axis - 1 + 2;
    }

    switch (cldnn_axis) {
        case 0: return kernel_selector::concat_axis::BATCH;
        case 1: return kernel_selector::concat_axis::FEATURE;
        case 2: return kernel_selector::concat_axis::X;
        case 3: return kernel_selector::concat_axis::Y;
        case 4: return kernel_selector::concat_axis::Z;
        case 5: return kernel_selector::concat_axis::W;
        default: IE_THROW() << "Unsupported concatenation axis: " << axis;
    }

    return kernel_selector::concat_axis::FEATURE;  // shouldn't get here
}

}  // namespace

struct concatenation_impl : typed_primitive_impl_ocl<concatenation> {
    using parent = typed_primitive_impl_ocl<concatenation>;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<concatenation_impl>(*this);
    }

    concatenation_impl(const concatenation_node& arg, const kernel_selector::kernel_data& kd) : parent(arg, kd) {
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
            return new concatenation_impl(arg, {});
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

        concat_params.axis = convert_axis(axis, arg.get_output_layout().get_rank());
        concat_optional_params.kernelPerInput = true;

        auto& kernel_selector = kernel_selector::concatenation_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(concat_params, concat_optional_params);
        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        concatenation_impl* concat = new concatenation_impl(arg, best_kernels[0]);

        return concat;
    }
};

namespace detail {

attach_concatenation_impl::attach_concatenation_impl() {
    implementation_map<concatenation>::add(impl_types::ocl, concatenation_impl::create, {
        MAKE_TUPLE6(bfwzyx,                 f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE6(bfyx,                   f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE6(bfzyx,                  f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE6(byxf,                   f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE2(fyxb,                   f32, f16),
        MAKE_TUPLE6(yxfb,                   f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE2(b_fs_yx_fsv4,                     u8, i8),
        MAKE_TUPLE4(b_fs_yx_fsv16,          f32, f16, u8, i8),
        MAKE_TUPLE2(b_fs_yx_fsv32,                    u8, i8),
        MAKE_TUPLE6(b_fs_zyx_fsv16,         f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE1(fs_b_yx_fsv32,               f16),
        MAKE_TUPLE4(bs_fs_yx_bsv16_fsv16,   f32, f16, u8, i8),
        MAKE_TUPLE6(bs_fs_yx_bsv32_fsv16,   f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE6(bs_fs_yx_bsv32_fsv32,   f32, f16, u8, i8, i32, i64),
        MAKE_TUPLE6(bs_fs_zyx_bsv16_fsv16,  f32, f16, u8, i8, i32, i64),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
