// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "eltwise_inst.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "eltwise/eltwise_kernel_base.h"

namespace cldnn {
namespace ocl {

struct eltwise_impl : typed_primitive_impl_ocl<eltwise> {
    using parent = typed_primitive_impl_ocl<eltwise>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::eltwise_kernel_selector;
    using kernel_params_t = kernel_selector::eltwise_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::eltwise_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<eltwise_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<eltwise>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);
        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<eltwise>();
        auto inputs_count = primitive->input.size();

        auto params = get_default_params<kernel_selector::eltwise_params>(impl_param, is_shape_agnostic);
        const auto mode = convert_to_eltwise_mode(primitive->mode);

        for (size_t i = 1; i < inputs_count; i++) {
            params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[i]));
        }

        if (inputs_count == 1) {
            params.operations.push_back({{kernel_selector::eltwise_params::InputType::Buffer(0)}, mode});
        } else {
            params.operations.push_back({{kernel_selector::eltwise_params::InputType::Buffer(0),
                                          kernel_selector::eltwise_params::InputType::Buffer(1)},
                                         mode});
        }

        for (uint32_t i = 2; i < static_cast<uint32_t>(inputs_count); i++) {
            params.operations.push_back({{kernel_selector::eltwise_params::InputType::Intermediate(i - 2),
                                          kernel_selector::eltwise_params::InputType::Buffer(i)},
                                         mode});
        }

        params.coefficients = primitive->coefficients;

        // WA to always match compiled dynamic kernel with dispatch data
        // W/O enforcing this option we may generate kernel for "broadcast" scneario due to umatched tensor dimensions
        // but in runtime dispatch data will be generated for non-broadcast case as shapes are actually same.
        if (impl_param.get_program().get_node(primitive->id).is_dynamic()) {
            params.broadcast = true;
        } else {
            for (size_t i = 0; i < params.inputs.size(); i++) {
                if (!params.inputs[i].SameDims(params.outputs[0])) {
                    std::vector<int32_t> input_size = impl_param.input_layouts[i].get_tensor().raw.vector();
                    std::vector<int32_t> output_size = impl_param.get_output_layout().get_tensor().raw.vector();
                    bool broadcast = false;
                    for (size_t d = 0; d < output_size.size(); d++) {
                        if (output_size[d] != 1 && input_size[d] == 1)
                            broadcast = true;
                    }
                    if (broadcast) {
                        params.broadcast = true;
                        break;
                    } else {
                        params.layoutBased = true;
                        break;
                    }
                }
            }
        }

        // stride
        if (!primitive->stride.empty()) {
            const auto& stride = primitive->stride;
            params.stride.resize(stride.size());
            for (size_t i = 0; i < primitive->stride.size(); i++) {
                params.stride[i] = {(uint32_t)stride[i].spatial[0],
                                    (uint32_t)stride[i].spatial[1],
                                    (uint32_t)stride[i].spatial[2]};
            }
        }

        // check if strides are the same
        if (!params.stride.empty()) {
            const auto& stride = params.stride[0];
            for (size_t i = 1; i < params.stride.size(); i++) {
                if (stride.x != params.stride[i].x || stride.y != params.stride[i].y)
                    params.layoutBased = true;
            }
        } else if (params.inputs.size() > 1 && (!params.inputs[0].SameDimsSizes(params.inputs[1]))) {
            params.broadcast = true;
        }

        // TODO [LOW PRECISION]: check if this parameter's really needed. Maybe data types are enough
        bool quantization = true;
        for (size_t i = 0; i < inputs_count; i++) {
            if (impl_param.input_layouts[i].data_type != data_types::u8 &&
                impl_param.input_layouts[i].data_type != data_types::i8) {
                quantization = false;
            }
        }
        params.int8_quantization = quantization;

        return params;
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        auto updated_impl_params = canonicalize_fused_shapes(impl_params);
        bool use_new_shape_infer = impl_params.prog->is_new_shape_infer();

        auto& output_layout = updated_impl_params.output_layouts[0];
        auto out_pshape = output_layout.get_partial_shape();
        output_layout.set_partial_shape(extend_shape_to_rank_from_end(out_pshape));

        for (auto& input_layout : updated_impl_params.input_layouts) {
            auto input_pshape = input_layout.get_partial_shape();
            if (!broadcastable(input_pshape, out_pshape, use_new_shape_infer)) {
                input_pshape = extend_shape_to_rank_from_begin(input_pshape, out_pshape.size());
            }
            input_layout.set_partial_shape(extend_shape_to_rank_from_end(input_pshape));
            input_layout.format = format::adjust_to_rank(input_layout.format, input_pshape.size());
        }

        return updated_impl_params;
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const override {
        return static_canonicalize_shapes(impl_params);
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

attach_eltwise_impl::attach_eltwise_impl() {
    auto dyn_types = {
        data_types::f32,
        data_types::f16,
        data_types::i8,
        data_types::u8,
        data_types::i16,
        data_types::u16,
        data_types::i32,
        data_types::u32,
        data_types::i64
    };

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
        format::bfuwzyx,
        format::bfvuwzyx,
    };

    implementation_map<eltwise>::add(impl_types::ocl,
                                     shape_types::dynamic_shape,
                                     typed_primitive_impl_ocl<eltwise>::create<eltwise_impl>,
                                     dyn_types,
                                     dyn_formats);

    implementation_map<eltwise>::add(impl_types::ocl, shape_types::static_shape, typed_primitive_impl_ocl<eltwise>::create<eltwise_impl>, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::i8, format::yxfb),
        std::make_tuple(data_types::u8, format::yxfb),
        std::make_tuple(data_types::i16, format::yxfb),
        std::make_tuple(data_types::u16, format::yxfb),
        std::make_tuple(data_types::u32, format::yxfb),
        std::make_tuple(data_types::i32, format::yxfb),
        std::make_tuple(data_types::i64, format::yxfb),

        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u16, format::bfyx),
        std::make_tuple(data_types::i16, format::bfyx),
        std::make_tuple(data_types::u32, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i64, format::bfyx),

        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::i8, format::byxf),
        std::make_tuple(data_types::u8, format::byxf),
        std::make_tuple(data_types::i16, format::byxf),
        std::make_tuple(data_types::u16, format::byxf),
        std::make_tuple(data_types::u32, format::byxf),
        std::make_tuple(data_types::i32, format::byxf),
        std::make_tuple(data_types::i64, format::byxf),

        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),

        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i16, format::bfzyx),
        std::make_tuple(data_types::u16, format::bfzyx),
        std::make_tuple(data_types::u32, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::i64, format::bfzyx),

        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::i8, format::bfwzyx),
        std::make_tuple(data_types::u8, format::bfwzyx),
        std::make_tuple(data_types::i16, format::bfwzyx),
        std::make_tuple(data_types::u16, format::bfwzyx),
        std::make_tuple(data_types::u32, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
        std::make_tuple(data_types::i64, format::bfwzyx),

        std::make_tuple(data_types::f32, format::bfuwzyx),
        std::make_tuple(data_types::f16, format::bfuwzyx),
        std::make_tuple(data_types::i8, format::bfuwzyx),
        std::make_tuple(data_types::u8, format::bfuwzyx),
        std::make_tuple(data_types::i16, format::bfuwzyx),
        std::make_tuple(data_types::u16, format::bfuwzyx),
        std::make_tuple(data_types::u32, format::bfuwzyx),
        std::make_tuple(data_types::i32, format::bfuwzyx),
        std::make_tuple(data_types::i64, format::bfuwzyx),

        std::make_tuple(data_types::f32, format::bfvuwzyx),
        std::make_tuple(data_types::f16, format::bfvuwzyx),
        std::make_tuple(data_types::i8, format::bfvuwzyx),
        std::make_tuple(data_types::u8, format::bfvuwzyx),
        std::make_tuple(data_types::i16, format::bfvuwzyx),
        std::make_tuple(data_types::u16, format::bfvuwzyx),
        std::make_tuple(data_types::u32, format::bfvuwzyx),
        std::make_tuple(data_types::i32, format::bfvuwzyx),
        std::make_tuple(data_types::i64, format::bfvuwzyx),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i64, format::b_fs_zyx_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i64, format::bs_fs_zyx_bsv16_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv16_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv16_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv16_fsv32),
        std::make_tuple(data_types::i32, format::bs_fs_zyx_bsv16_fsv32),
        std::make_tuple(data_types::i64, format::bs_fs_zyx_bsv16_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),

        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv2),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv2),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv2),
        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv2),

        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv8_fsv2),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv8_fsv2),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv8_fsv2),
        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv8_fsv2),

        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv16_fsv2),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv16_fsv2),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv16_fsv2),
        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv16_fsv2),

        std::make_tuple(data_types::i8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv4),

        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),

        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv32),

        std::make_tuple(data_types::f16, format::fs_b_yx_fsv32),

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

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv16_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv16_fsv32),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv16_fsv32),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv16_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv4_fsv4),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv8_fsv4),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv8_fsv2),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv4_fsv2),

        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv32_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv32_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv32_fsv32),
        std::make_tuple(data_types::i32, format::bs_fs_zyx_bsv32_fsv32),
        std::make_tuple(data_types::i64, format::bs_fs_zyx_bsv32_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv32_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv32_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv32_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv32_fsv16),
        std::make_tuple(data_types::i32, format::bs_fs_zyx_bsv32_fsv16),
        std::make_tuple(data_types::i64, format::bs_fs_zyx_bsv32_fsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::eltwise_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::eltwise)
