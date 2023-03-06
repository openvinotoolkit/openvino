// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "border_inst.h"

#include "border/border_kernel_selector.h"
#include "border/border_kernel_base.h"

namespace cldnn {
namespace ocl {

struct border_impl : typed_primitive_impl_ocl<border> {
    using parent = typed_primitive_impl_ocl<border>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::border_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::border_params, kernel_selector::border_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<border_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<border>();
        auto params = get_default_params<kernel_selector::border_params>(impl_param, is_shape_agnostic);
        auto optional_params = get_default_optional_params<kernel_selector::border_optional_params>(impl_param.get_program());

        size_t rank = impl_param.get_input_layout(0).get_rank();
        format pads_format = format::adjust_to_rank(format::bfyx, rank);

        std::vector<int32_t> begin(primitive->pads_begin.begin(), primitive->pads_begin.end());
        std::vector<int32_t> end(primitive->pads_end.begin(), primitive->pads_end.end());

        size_t input_offset = 1;
        if (!(primitive->non_constant_input_mask & border::PAD_NON_CONST_INPUT::BEGIN)) {
            params.begin_type = kernel_selector::base_params::ArgType::Constant;

            std::vector<int64_t> begin_vec;
            begin_vec.assign(primitive->pads_begin.begin(), primitive->pads_begin.end());
            if (begin_vec.size() < rank) {
                size_t zeros_to_add = rank - begin_vec.size();
                begin_vec.insert(begin_vec.end(), zeros_to_add, 0);
            }
            std::vector<tensor::value_type> pads_begin(begin_vec.begin(), begin_vec.end());
            params.lt_sizes = convert_dim_vector(tensor(pads_format, pads_begin, 0));
        } else {
            params.begin_type = kernel_selector::base_params::ArgType::Input;

            auto begin_layout = impl_param.get_input_layout(input_offset);
            params.inputs.push_back(convert_data_tensor(begin_layout));
            input_offset += 1;
        }

        if (!(primitive->non_constant_input_mask & border::PAD_NON_CONST_INPUT::END)) {
            params.end_type = kernel_selector::base_params::ArgType::Constant;

            std::vector<int64_t> end_vec;
            end_vec.assign(primitive->pads_end.begin(), primitive->pads_end.end());
            if (end_vec.size() < rank) {
                size_t zeros_to_add = rank - end_vec.size();
                end_vec.insert(end_vec.end(), zeros_to_add, 0);
            }
            std::vector<tensor::value_type> pads_end(end_vec.begin(), end_vec.end());
            params.rb_sizes = convert_dim_vector(tensor(pads_format, pads_end, 0));
        } else {
            params.end_type = kernel_selector::base_params::ArgType::Input;

            auto end_layout = impl_param.get_input_layout(input_offset);
            params.inputs.push_back(convert_data_tensor(end_layout));
            input_offset += 1;
        }

        if (!(primitive->non_constant_input_mask & border::PAD_NON_CONST_INPUT::VALUE)) {
            params.pad_value_type = kernel_selector::base_params::ArgType::Constant;
            params.border_value = primitive->pad_value;
        } else {
            params.pad_value_type = kernel_selector::base_params::ArgType::Input;
            auto pad_value_layout = impl_param.get_input_layout(input_offset);
            params.inputs.push_back(convert_data_tensor(pad_value_layout));
        }

        switch (primitive->pad_mode) {
            case ov::op::PadMode::CONSTANT:
                params.b_type = kernel_selector::border_type::CONSTANT;
                break;
            case ov::op::PadMode::EDGE:
                params.b_type = kernel_selector::border_type::EDGE;
                break;
            case ov::op::PadMode::SYMMETRIC:
                params.b_type = kernel_selector::border_type::MIRROR;
                break;
            case ov::op::PadMode::REFLECT:
                params.b_type = kernel_selector::border_type::MIRROR_101;
                break;
            default:
                OPENVINO_ASSERT(false, "[GPU] Encountered unhandled enum case: PadMode during translation to kernel selector enumeration.");
        }

        return {params, optional_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
        update_kernels_list_to_skip();
    }
};

namespace detail {

attach_border_impl::attach_border_impl() {
    auto types = {data_types::f32, data_types::f16, data_types::i32, data_types::i8, data_types::u8};

    auto formats = {
        format::yxfb,
        format::bfyx,
        format::byxf,
        format::bfzyx,
        format::bfwzyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::b_fs_zyx_fsv16,
        format::bs_fs_yx_bsv4_fsv2,
        format::bs_fs_yx_bsv4_fsv4,
        format::bs_fs_yx_bsv8_fsv2,
        format::bs_fs_yx_bsv8_fsv4,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_zyx_bsv16_fsv16
    };

    implementation_map<border>::add(impl_types::ocl,
                                    shape_types::static_shape,
                                    typed_primitive_impl_ocl<border>::create<border_impl>,
                                    types,
                                    formats);

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx
    };

    implementation_map<border>::add(impl_types::ocl,
                                    shape_types::dynamic_shape,
                                    typed_primitive_impl_ocl<border>::create<border_impl>,
                                    types,
                                    dyn_formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::border_impl)
