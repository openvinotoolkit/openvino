// Copyright (C) 2018-2024 Intel Corporation
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
    using kernel_params_t = kernel_selector::border_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::border_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<border_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<border>();
        auto params = get_default_params<kernel_selector::border_params>(impl_param, is_shape_agnostic);

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
            params.lt_sizes = convert_dim_vector<int32_t>(tensor(pads_format, pads_begin, 0));
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
            params.rb_sizes = convert_dim_vector<int32_t>(tensor(pads_format, pads_end, 0));
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

        params.allow_negative_pad = primitive->allow_negative_pad;

        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }

        update_shapes(*_kernel_data.params, impl_param);
        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        const auto& prim_params = static_cast<const kernel_selector::border_params&>(*_kernel_data.params);
        if (prim_params.inputs[0].LogicalSize() == 0) {
            ob << true;
        } else {
            ob << false;
        }
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> zero_input;
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

protected:
    // WA for static impl deserialization
    bool zero_input = false;

    kernel_arguments_data get_arguments(const typed_primitive_inst<border>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);

        // In case of zero input shape and non-zero output (kernel execution is not skipped), we need to add fake input buffer
        // So as not to get an error during the argument setting stage
        if (instance.get_input_layout().count() == 0) {
            args.inputs[0] = instance.get_intermediates_memories().front();
        }

        return args;
    }

    std::vector<layout> get_internal_buffer_layouts_impl(const kernel_impl_params& /*params*/) const override {
        const auto& prim_params = static_cast<const kernel_selector::border_params&>(*_kernel_data.params);
        std::vector<layout> layouts;

        if ((_kernel_data.params == nullptr && zero_input) ||
            (_kernel_data.params != nullptr && prim_params.inputs[0].LogicalSize() == 0)) {
            layout any_layout = {data_types::u8, format::bfyx, {1, 1, 1, 1}};
            layouts.push_back(any_layout);
        }

        return layouts;
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
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::border)
