// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "cldnn/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "pooling/pooling_kernel_selector.h"
#include "pooling/pooling_kernel_base.h"
#include <algorithm>

namespace cldnn {
namespace ocl {

namespace {
void validate_args(const pooling_node& arg) {
    auto const& input_buffer_size = arg.input().get_output_layout().get_buffer_size();
    auto const& input_dimensions =
        input_buffer_size.batch.size() + input_buffer_size.feature.size() + input_buffer_size.spatial.size();
    auto const& output_buffer_size = arg.get_output_layout().get_buffer_size();
    auto const& output_dimensions =
        output_buffer_size.batch.size() + output_buffer_size.feature.size() + output_buffer_size.spatial.size();
    auto& stride = arg.get_primitive()->stride;
    auto const& stride_dimensions = stride.batch.size() + stride.feature.size() + stride.spatial.size();
    auto& window = arg.get_primitive()->size;
    auto const& window_dimensions = window.batch.size() + window.feature.size() + window.spatial.size();

    CLDNN_ERROR_NOT_EQUAL(arg.id(), "input dimensions", input_dimensions, "output dimensions", output_dimensions, "");
    CLDNN_ERROR_NOT_EQUAL(arg.id(), "stride dimensions", stride_dimensions, "output dimensions", output_dimensions, "");
    CLDNN_ERROR_NOT_EQUAL(arg.id(), "window dimensions", window_dimensions, "output dimensions", output_dimensions, "");
}

kernel_selector::pool_type cldnn_2_pool_type(pooling_mode mode) {
    switch (mode) {
        case pooling_mode::max:
            return kernel_selector::pool_type::MAX;
        case pooling_mode::average:
            return kernel_selector::pool_type::AVG;
        case pooling_mode::average_no_padding:
            return kernel_selector::pool_type::AVG;
        case pooling_mode::max_with_argmax:
            return kernel_selector::pool_type::MAX_WITH_ARGMAX;
        default:
            assert(0);
            return kernel_selector::pool_type::MAX;
    }
}

kernel_selector::kernel_divider_mode cldnn_2_kernel_divider_mode(pooling_mode mode) {
    switch (mode) {
        case pooling_mode::max:
        case pooling_mode::max_with_argmax:
            return kernel_selector::kernel_divider_mode::DONT_CARE;
        case pooling_mode::average:
            return kernel_selector::kernel_divider_mode::FIXED;
        case pooling_mode::average_no_padding:
            return kernel_selector::kernel_divider_mode::DYNAMIC;
        default:
            assert(0);
            return kernel_selector::kernel_divider_mode::DONT_CARE;
    }
}
}  // namespace

struct pooling_impl : typed_primitive_impl_ocl<pooling> {
    using parent = typed_primitive_impl_ocl<pooling>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<pooling_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(typed_primitive_inst<pooling>& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);
        if (!instance.argument.argmax.empty())
            args.inputs.push_back(instance.dep_memory_ptr(1));
        return args;
    }

public:
    static primitive_impl* create(const pooling_node& arg) {
        validate_args(arg);

        auto pool_params = get_default_params<kernel_selector::pooling_params>(arg);
        auto pool_optional_params =
            get_default_optional_params<kernel_selector::pooling_optional_params>(arg.get_program());

        const auto primitive = arg.get_primitive();
        const auto& stride = primitive->stride;
        const auto& input_offset = primitive->input_offset;
        const auto& input_sizes = arg.input().get_output_layout().size;
        const auto& output_sizes = arg.get_output_layout().size;

        auto& pp = pool_params;

        pp.poolType = cldnn_2_pool_type(primitive->mode);
        pp.remainderAction = kernel_selector::pool_remainder::CEIL;

        if (primitive->global_pooling) {
            primitive->size.spatial[0] = input_sizes.spatial[0];
            primitive->size.spatial[1] = input_sizes.spatial[1];
            primitive->size.spatial[2] = input_sizes.spatial[2];
        }

        // check if last pooling window goes outside of input size + padding. If so the avg pooling size will be
        // adjusted to that, to work properly this calculation must take pad_end into account.
        auto dynamic_mode = (((output_sizes.spatial[0] - 1) * stride.spatial[0]) + primitive->size.spatial[0]) >
                                 (-input_offset.spatial[0] - primitive->pad_end.spatial[0]) + input_sizes.spatial[0] ||
                            (((output_sizes.spatial[1] - 1) * stride.spatial[1]) + primitive->size.spatial[1]) >
                                 (-input_offset.spatial[1] - primitive->pad_end.spatial[1]) + input_sizes.spatial[1] ||
                            (((output_sizes.spatial[2] - 1) * stride.spatial[2]) + primitive->size.spatial[2]) >
                                 (-input_offset.spatial[2] - primitive->pad_end.spatial[2]) + input_sizes.spatial[2];

        if (primitive->mode == pooling_mode::average && dynamic_mode)
            pp.divMode = kernel_selector::kernel_divider_mode::DYNAMIC_WITH_PADDING;
        else
            pp.divMode = cldnn_2_kernel_divider_mode(primitive->mode);

        const auto additional_offset = tensor::max(input_offset, (tensor) 0);
        if (additional_offset != (tensor) 0) {
            const auto& input_layout = arg.input().get_output_layout();
            pool_params.inputs[0] = convert_data_tensor(input_layout, 1, additional_offset);
        }

        if (primitive->mode == pooling_mode::max_with_argmax)
            pool_params.inputs.push_back(convert_data_tensor(arg.argmax().get_output_layout()));

        pp.poolSize = {
            (uint32_t)primitive->size.spatial[0],
            (uint32_t)primitive->size.spatial[1],
            (uint32_t)primitive->size.spatial[2],
        };

        pp.poolPad = {(uint32_t)std::max(-input_offset.spatial[0], 0),
                      (uint32_t)std::max(-input_offset.spatial[1], 0),
                      (uint32_t)std::max(-input_offset.spatial[2], 0)};

        pp.poolStride = {(uint32_t)stride.spatial[0], (uint32_t)stride.spatial[1], (uint32_t)stride.spatial[2]};

        auto& kernel_selector = kernel_selector::pooling_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(pool_params, pool_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto pool = new pooling_impl(arg, best_kernels[0]);

        return pool;
    }
};

namespace detail {

attach_pooling_impl::attach_pooling_impl() {
    implementation_map<pooling>::add(impl_types::ocl, pooling_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),

        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::i8, format::yxfb),
        std::make_tuple(data_types::u8, format::yxfb),

        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::i8, format::byxf),
        std::make_tuple(data_types::u8, format::byxf),

        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv16_fsv16),

        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv16_fsv16),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv4),

        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),

        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv32),

        std::make_tuple(data_types::f16, format::fs_b_yx_fsv32),
        std::make_tuple(data_types::f32, format::fs_b_yx_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
