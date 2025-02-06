// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "crop_inst.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "eltwise/eltwise_kernel_base.h"

#include "openvino/core/validation_util.hpp"

namespace cldnn {
namespace ocl {

struct crop_impl : typed_primitive_impl_ocl<crop> {
    using parent = typed_primitive_impl_ocl<crop>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::eltwise_kernel_selector;
    using kernel_params_t = kernel_selector::eltwise_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::crop_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<crop_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic() && _kernel_data.kernelName.length() != 0) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        auto params = get_default_params<kernel_selector::eltwise_params>(impl_param, is_shape_agnostic);

        params.operations.push_back({{kernel_selector::eltwise_params::InputType::Buffer(0)}, kernel_selector::eltwise_mode::ASSIGN});
        if (impl_param.is_dynamic() || is_shape_agnostic) {
            // WA to always match compiled dynamic kernel with dispatch data
            // W/O enforcing this option we may generate kernel for "broadcast" scneario due to umatched tensor dimensions
            // but in runtime dispatch data will be generated for non-broadcast case as shapes are actually same.
            params.broadcast = true;
        } else {
            params.inputs[0] = convert_data_tensor(impl_param.get_input_layout(), impl_param.input_offsets[0]);
        }
        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }

        update_shapes(*_kernel_data.params, impl_param);

        // The padding sizes are reset to 0 up to crop_axis, as kernel reads data using
        // "input[GET_INDEX(INPUT, order) + runtime_offset]", where GET_INDEX handles all paddings before
        // specified axis. However, for proper runtime offset calculation, we have to consider paddings
        // after the crop_axis, which requires subtracting input_offset from the runtime buffer, since
        // padding for the first element is already included in the GET_INDEX() call.
        // For example, for input shape like: [1, 32, 128 (pad_before=512, pad_after=0), 8 (pad_before=4, pad_after=4)]
        // with crop_axis=2 and split_lengths = {64, 64},
        // runtime_offset should be set in terms of [1, 32, 128 (pad_before=0, pad_after=0), 8 (pad_before=4, pad_after=4)] shape.
        // So crop.out0's runtime_offset=0 and crop.out1's runtime_offset=1024.

        auto input_layout = impl_param.get_input_layout();
        auto crop_axis = ov::util::normalize(impl_param.typed_desc<crop>()->axis, static_cast<int64_t>(input_layout.get_partial_shape().size()));

        input_layout.data_padding._dynamic_dims_mask = padding::EMPTY_MASK;
        for (size_t i = 0; i <= static_cast<size_t>(crop_axis); i++) {
            input_layout.data_padding._lower_size[i] = 0;
            input_layout.data_padding._upper_size[i] = 0;
        }

        auto input_offset = convert_data_tensor(input_layout).GetFirstElementOffset();
        auto runtime_offset = convert_data_tensor(input_layout, impl_param.input_offsets[0]).GetFirstElementOffset() - input_offset;
        kernel_selector::ScalarDescriptor s;
        s.t = kernel_selector::ScalarDescriptor::Types::UINT32;
        s.v.u32 = static_cast<uint32_t>(runtime_offset);
        OPENVINO_ASSERT(_kernel_data.kernels[0].params.scalars.size() == 1,
                "[GPU] Scalar field for runtime offset is not added for crop shape agnostic impl");
        _kernel_data.kernels[0].params.scalars[0] = s;
        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }
};

namespace detail {

attach_crop_impl::attach_crop_impl() {
    auto dyn_types = {
        data_types::f32,
        data_types::f16,
        data_types::i8,
        data_types::u8,
        data_types::i32,
        data_types::i64
    };

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx
    };

    implementation_map<crop>::add(impl_types::ocl,
                                     shape_types::dynamic_shape,
                                     typed_primitive_impl_ocl<crop>::create<crop_impl>,
                                     dyn_types,
                                     dyn_formats);

    implementation_map<crop>::add(impl_types::ocl, typed_primitive_impl_ocl<crop>::create<crop_impl>, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::i64, format::yxfb),
        std::make_tuple(data_types::i32, format::yxfb),
        std::make_tuple(data_types::i8, format::yxfb),
        std::make_tuple(data_types::u8, format::yxfb),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i64, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::i64, format::byxf),
        std::make_tuple(data_types::i32, format::byxf),
        std::make_tuple(data_types::i8, format::byxf),
        std::make_tuple(data_types::u8, format::byxf),
        std::make_tuple(data_types::f32, format::fyxb),
        std::make_tuple(data_types::f16, format::fyxb),
        std::make_tuple(data_types::i64, format::fyxb),
        std::make_tuple(data_types::i32, format::fyxb),
        std::make_tuple(data_types::i8, format::fyxb),
        std::make_tuple(data_types::u8, format::fyxb),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i64, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::i64, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
        std::make_tuple(data_types::i8, format::bfwzyx),
        std::make_tuple(data_types::u8, format::bfwzyx),

        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),

        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i64, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i64, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv16_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv32_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv32_fsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::crop_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::crop)
