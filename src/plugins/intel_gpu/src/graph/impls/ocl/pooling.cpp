// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "pooling/pooling_kernel_selector.h"
#include "pooling/pooling_kernel_base.h"
#include <algorithm>

namespace cldnn {
namespace ocl {

namespace {
void validate_args(const pooling_node& arg) {
    auto input_rank = arg.input().get_output_layout().get_spatial_rank();
    auto output_rank = arg.get_output_layout().get_spatial_rank();
    auto stride_rank = arg.get_primitive()->stride.size();
    auto window_rank = arg.get_primitive()->size.size();

    if (!arg.get_primitive()->global_pooling) {
        CLDNN_ERROR_NOT_EQUAL(arg.id(), "input dimensions", input_rank, "output dimensions", output_rank, "");
        CLDNN_ERROR_NOT_EQUAL(arg.id(), "stride dimensions", stride_rank, "output dimensions", output_rank, "");
        CLDNN_ERROR_NOT_EQUAL(arg.id(), "window dimensions", window_rank, "output dimensions", output_rank, "");
    }
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
        const auto primitive = arg.get_primitive();
        const auto& param_info = kernel_impl_params(arg.get_program(), primitive, arg.get_unique_id(),
                                                    arg.get_input_layouts(), arg.get_output_layout(),
                                                    arg.get_fused_primitives(),
                                                    arg.get_fused_activations_funcs(), arg.get_fused_activations_params());
        auto pool_params = get_default_params<kernel_selector::pooling_params>(param_info);
        auto pool_optional_params =
            get_default_optional_params<kernel_selector::pooling_optional_params>(arg.get_program());

        pool_params.maxPoolOpset8Features = primitive->maxPoolOpset8Features;
        if (pool_params.maxPoolOpset8Features) {
            switch (primitive->index_element_type) {
                case cldnn::data_types::i32: {
                    pool_params.poolIndexElementType = kernel_selector::Datatype::INT32;
                    break;
                }
                case cldnn::data_types::i64: {
                    pool_params.poolIndexElementType = kernel_selector::Datatype::INT64;
                    break;
                }
                default:
                    throw std::runtime_error{"Not supported index element type"};
            }
            pool_params.poolAxis = primitive->axis;
        }

        const auto& stride = primitive->stride;
        const auto& pad = primitive->pad;
        const auto& dilation = primitive->dilation;
        auto kernel = primitive->size;
        const auto& input_layout = arg.input().get_output_layout();
        const auto& output_layout = arg.get_output_layout();
        auto spatial_rank = output_layout.get_spatial_rank();

        auto& pp = pool_params;

        pp.poolType = cldnn_2_pool_type(primitive->mode);
        pp.remainderAction = kernel_selector::pool_remainder::CEIL;

        if (primitive->global_pooling) {
            kernel = ov::Shape(spatial_rank, 1);
            for (size_t i = 0; i < spatial_rank; i++) {
                kernel[i] = input_layout.spatial(spatial_rank - i - 1);
            }
        }

        // check if last pooling window goes outside of input size + padding. If so the avg pooling size will be
        // adjusted to that, to work properly this calculation must take pad_end into account.
        auto dynamic_mode = false;
        for (size_t i = 0; i < spatial_rank; i++) {
            dynamic_mode |= (((output_layout.spatial(i) - 1) * stride[spatial_rank - i - 1]) + primitive->size[spatial_rank - i - 1]) >
                                 (primitive->pad_end[spatial_rank - i - 1] + pad[spatial_rank - i - 1]) + input_layout.spatial(i);
        }

        if (primitive->mode == pooling_mode::average && dynamic_mode)
            pp.divMode = kernel_selector::kernel_divider_mode::DYNAMIC_WITH_PADDING;
        else
            pp.divMode = cldnn_2_kernel_divider_mode(primitive->mode);

        if (primitive->mode == pooling_mode::max_with_argmax)
            pool_params.inputs.push_back(convert_data_tensor(arg.argmax().get_output_layout()));

        uint32_t kernel_z = kernel.size() >= 3 ? kernel[kernel.size() - 3] : 1;
        uint32_t kernel_y = kernel.size() >= 2 ? kernel[kernel.size() - 2] : 1;
        uint32_t kernel_x = kernel.size() >= 1 ? kernel[kernel.size() - 1] : 1;
        pp.poolSize = {kernel_x, kernel_y, kernel_z};

        uint32_t pad_z = std::max<std::ptrdiff_t>(pad.size() >= 3 ? pad[pad.size() - 3] : 0, 0);
        uint32_t pad_y = std::max<std::ptrdiff_t>(pad.size() >= 2 ? pad[pad.size() - 2] : 0, 0);
        uint32_t pad_x = std::max<std::ptrdiff_t>(pad.size() >= 1 ? pad[pad.size() - 1] : 0, 0);
        pp.poolPad  = {pad_x, pad_y, pad_z};

        uint32_t stride_z = stride.size() >= 3 ? stride[stride.size() - 3] : 1;
        uint32_t stride_y = stride.size() >= 2 ? stride[stride.size() - 2] : 1;
        uint32_t stride_x = stride.size() >= 1 ? stride[stride.size() - 1] : 1;
        pp.poolStride = {stride_x, stride_y, stride_z};

        uint32_t dilation_z = dilation.size() >= 3 ? dilation[dilation.size() - 3] : 1;
        uint32_t dilation_y = dilation.size() >= 2 ? dilation[dilation.size() - 2] : 1;
        uint32_t dilation_x = dilation.size() >= 1 ? dilation[dilation.size() - 1] : 1;
        pp.poolDilation = {dilation_x, dilation_y, dilation_z};

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
