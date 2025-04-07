// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "quantize_inst.h"
#include "quantize/quantize_kernel_selector.h"
#include "quantize/quantize_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct quantize_impl : typed_primitive_impl_ocl<quantize> {
    using parent = typed_primitive_impl_ocl<quantize>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::quantize_kernel_selector;
    using kernel_params_t = kernel_selector::quantize_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::quantize_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<quantize_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic() && _kernel_data.kernelName.length() != 0) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<quantize>& instance) const override {
        kernel_arguments_data args;

        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }
        if (instance.get_typed_desc<quantize>()->scale_shift_opt) {
            if (instance.dependencies().size() == 9) {
                args.inputs.push_back(instance.dep_memory_ptr(5));
                args.inputs.push_back(instance.dep_memory_ptr(6));
                args.inputs.push_back(instance.dep_memory_ptr(7));
                args.inputs.push_back(instance.dep_memory_ptr(8));
            }
        }
        args.outputs = { instance.output_memory_ptr() };
        args.shape_info = instance.shape_info_memory_ptr();

        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        auto quantize_params = get_default_params<kernel_selector::quantize_params>(impl_param);
        const auto& arg = impl_param.prog->get_node(impl_param.desc->id).as<quantize>();

        quantize_params.levels = arg.get_levels();
        quantize_params.scale_shift_opt = arg.get_scale_shift_opt();
        quantize_params.has_post_scale = arg.get_need_post_scale();
        quantize_params.has_post_shift = arg.get_need_post_shift();
        quantize_params.has_pre_shift = arg.get_need_pre_shift();
        quantize_params.has_clamp = arg.get_need_clamp();
        quantize_params.has_min_clamp = arg.get_need_min_clamp();
        quantize_params.has_max_clamp = arg.get_need_max_clamp();

        quantize_params.per_tensor_input_range = arg.get_per_tensor_input_range();
        quantize_params.per_tensor_input_scale = arg.get_per_tensor_input_scale();
        quantize_params.per_tensor_input_shift = arg.get_per_tensor_input_shift();
        quantize_params.per_tensor_output_range = arg.get_per_tensor_output_range();
        quantize_params.per_tensor_output_scale = arg.get_per_tensor_output_scale();
        quantize_params.per_tensor_output_shift = arg.get_per_tensor_output_shift();

        quantize_params.in_lo = arg.get_input_lo_val();
        quantize_params.in_hi = arg.get_input_hi_val();
        quantize_params.in_scale = arg.get_input_scale_val();
        quantize_params.in_shift = arg.get_input_shift_val();
        quantize_params.out_lo = arg.get_output_lo_val();
        quantize_params.out_hi = arg.get_output_hi_val();
        quantize_params.out_scale = arg.get_output_scale_val();
        quantize_params.out_shift = arg.get_output_shift_val();

        for (size_t i = 1; i < arg.get_inputs_count(); i++) {
            quantize_params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[i]));
        }

        return quantize_params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }

        update_shapes(*_kernel_data.params, impl_param);
        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }
};

namespace detail {

attach_quantize_impl::attach_quantize_impl() {
    auto types = {
        data_types::f16,
        data_types::f32,
        data_types::i8,
        data_types::u8
    };

    auto formats = {
        format::bfyx,
        format::byxf,
        format::b_fs_yx_fsv4,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::fs_b_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv16_fsv32,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,

        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv32_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,

        format::bfwzyx,
        format::bfuwzyx,
        format::bfvuwzyx,
    };

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
        format::bfuwzyx,
        format::bfvuwzyx,
        format::b_fs_yx_fsv16,
    };

    auto keys = implementation_map<quantize>::combine(types, formats);
    keys.emplace(data_types::f16, format::yxfb);
    keys.emplace(data_types::f32, format::yxfb);

    implementation_map<quantize>::add(impl_types::ocl,
                                      shape_types::static_shape,
                                      typed_primitive_impl_ocl<quantize>::create<quantize_impl>,
                                      keys);

    implementation_map<quantize>::add(impl_types::ocl,
                                      shape_types::dynamic_shape,
                                      typed_primitive_impl_ocl<quantize>::create<quantize_impl>,
                                      types,
                                      dyn_formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::quantize_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::quantize)
