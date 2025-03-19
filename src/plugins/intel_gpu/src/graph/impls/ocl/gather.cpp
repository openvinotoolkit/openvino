// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "gather_inst.h"
#include "gather/gather_kernel_selector.h"
#include "gather/gather_kernel_ref.h"

namespace cldnn {
namespace ocl {
static kernel_selector::gather_axis convert_axis(int64_t axis, size_t rank) {
    if (axis == 0) {
        return kernel_selector::gather_axis::BATCH;
    } else if (axis == 1) {
        return kernel_selector::gather_axis::FEATURE;
    }

    if (rank <= 4) {
        switch (axis) {
            case 2: return kernel_selector::gather_axis::Y;
            case 3: return kernel_selector::gather_axis::X;
            case -1: return kernel_selector::gather_axis::Y;
            case -2: return kernel_selector::gather_axis::FEATURE;
            case -3: return kernel_selector::gather_axis::BATCH;
            default: OPENVINO_THROW("Unsupported gather axis: ", axis);
        }
    } else if (rank == 5) {
        switch (axis) {
            case 2: return kernel_selector::gather_axis::Z;
            case 3: return kernel_selector::gather_axis::Y;
            case 4: return kernel_selector::gather_axis::X;
            case -1: return kernel_selector::gather_axis::Y;
            case -2: return kernel_selector::gather_axis::Z;
            case -3: return kernel_selector::gather_axis::FEATURE;
            case -4: return kernel_selector::gather_axis::BATCH;
            default: OPENVINO_THROW("Unsupported gather axis: ", axis);
        }
    } else if (rank == 6) {
        switch (axis) {
            case 2: return kernel_selector::gather_axis::W;
            case 3: return kernel_selector::gather_axis::Z;
            case 4: return kernel_selector::gather_axis::Y;
            case 5: return kernel_selector::gather_axis::X;
            case -1: return kernel_selector::gather_axis::Y;
            case -2: return kernel_selector::gather_axis::Z;
            case -3: return kernel_selector::gather_axis::W;
            case -4: return kernel_selector::gather_axis::FEATURE;
            case -5: return kernel_selector::gather_axis::BATCH;
            default: OPENVINO_THROW("Unsupported gather axis: ", axis);
        }
    } else {
        OPENVINO_THROW("Unsupported gather axis: ", axis);
    }
}

struct gather_impl : typed_primitive_impl_ocl<gather> {
    using parent = typed_primitive_impl_ocl<gather>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::gather_kernel_selector;
    using kernel_params_t = kernel_selector::gather_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::gather_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<gather_impl, kernel_params_t>(*this);
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
    kernel_arguments_data get_arguments(const typed_primitive_inst<gather>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);
        const auto& desc = instance.get_typed_desc<gather>();

        if (desc->decompression_scale.is_valid())
            args.inputs.push_back(instance.dep_memory_ptr(2));

        if (desc->decompression_zero_point.is_valid())
            args.inputs.push_back(instance.dep_memory_ptr(3));

        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<gather>();
        auto params = get_default_params<kernel_selector::gather_params>(impl_param, is_shape_agnostic);

        auto input_layout = impl_param.get_input_layout(0);
        params.axis = convert_axis(primitive->axis, input_layout.get_rank());
        params.batch_dim = size_t(primitive->batch_dim);
        params.support_neg_ind = primitive->support_neg_ind;

        auto output_layout = impl_param.get_output_layout(0);
        auto in_rank = input_layout.get_partial_shape().size();
        auto out_rank = output_layout.get_partial_shape().size();

        if (in_rank > 4 && in_rank > out_rank) { // if in_rank <= 4, the dims are to be adjusted to 4 by convert_data_tensor
            auto output_shape = output_layout.get_partial_shape();
            ov::PartialShape new_output_shape({output_shape[0], output_shape[1]});
            for (size_t i = 0; i < in_rank - out_rank; ++i)
                new_output_shape.push_back(1);

            for (size_t i = 2; i < out_rank; ++i) {
                new_output_shape.push_back(output_shape[i]);
            }
            output_layout = layout(new_output_shape, output_layout.data_type, format::get_default_format(new_output_shape.size()));
        }

        params.outputs[0] = convert_data_tensor(output_layout);
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));

        bool commpressed = primitive->decompression_scale.is_valid();
        bool with_zp = primitive->decompression_zero_point.is_valid();
        if (commpressed) {
            params.compressed = true;
            params.decompression_scale = convert_data_tensor(impl_param.get_input_layout(2));
            if (with_zp) {
                params.has_decompression_zp = true;
                params.decompression_zero_point = convert_data_tensor(impl_param.get_input_layout(3));
            } else if (primitive->decompression_zero_point_scalar.has_value()) {
                params.has_decompression_zp = true;
                params.scalar_zp = true;
                params.zp_value = primitive->decompression_zero_point_scalar.value();
            }
        }

        return params;
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        auto updated_impl_params = canonicalize_fused_shapes(impl_params);
        const auto& prim = impl_params.typed_desc<gather>();

        auto input_pshape = updated_impl_params.input_layouts[0].get_partial_shape();
        auto& out_layout = updated_impl_params.output_layouts[0];
        auto output_pshape = out_layout.get_partial_shape();

        OPENVINO_ASSERT(input_pshape.size() <= output_pshape.size() || input_pshape.size() - output_pshape.size() == 1,
                        "[GPU] Gather output rank must be greater than or equal to the input rank, or less by one");

        if (input_pshape.size() > output_pshape.size()) {
            output_pshape.insert(output_pshape.begin() + prim->axis, ov::Dimension(1));
            out_layout.set_partial_shape(output_pshape);
            out_layout.format = format::adjust_to_rank(out_layout.format, output_pshape.size());
        }

        for (auto& input_layout : updated_impl_params.input_layouts) {
            input_layout.set_partial_shape(extend_shape_to_rank_from_end(input_layout.get_partial_shape()));
        }
        out_layout.set_partial_shape(extend_shape_to_rank_from_end(out_layout.get_partial_shape()));

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

attach_gather_impl::attach_gather_impl() {
    auto dyn_types = {
        data_types::f32,
        data_types::f16,
        data_types::i8,
        data_types::u8,
        data_types::i4,
        data_types::u4,
        data_types::i32
    };

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx
    };

    implementation_map<gather>::add(impl_types::ocl,
                                    shape_types::dynamic_shape,
                                    typed_primitive_impl_ocl<gather>::create<gather_impl>,
                                    dyn_types,
                                    dyn_formats);

    implementation_map<gather>::add(impl_types::ocl, shape_types::static_shape, typed_primitive_impl_ocl<gather>::create<gather_impl>, {
        std::make_tuple(data_types::f32, format::fyxb),
        std::make_tuple(data_types::f16, format::fyxb),
        std::make_tuple(data_types::i32, format::fyxb),
        std::make_tuple(data_types::i8, format::fyxb),
        std::make_tuple(data_types::u8, format::fyxb),

        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::i32, format::yxfb),
        std::make_tuple(data_types::i8, format::yxfb),
        std::make_tuple(data_types::u8, format::yxfb),

        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::i32, format::byxf),
        std::make_tuple(data_types::i8, format::byxf),
        std::make_tuple(data_types::u8, format::byxf),

        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i4, format::bfyx),
        std::make_tuple(data_types::u4, format::bfyx),

        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),

        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
        std::make_tuple(data_types::i8, format::bfwzyx),
        std::make_tuple(data_types::u8, format::bfwzyx),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::i32, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv4),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::i32, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv4_fsv2),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv4_fsv4),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv8_fsv2),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv8_fsv4),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv16_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv32),

        std::make_tuple(data_types::f32, format::fs_b_yx_fsv32),
        std::make_tuple(data_types::f16, format::fs_b_yx_fsv32),
        std::make_tuple(data_types::i32, format::fs_b_yx_fsv32),
        std::make_tuple(data_types::i8, format::fs_b_yx_fsv32),
        std::make_tuple(data_types::u8, format::fs_b_yx_fsv32),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::gather_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::gather)
