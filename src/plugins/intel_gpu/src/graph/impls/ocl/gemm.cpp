// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "gemm_inst.h"
#include "gemm/gemm_kernel_base.h"
#include "gemm/gemm_kernel_selector.h"

namespace cldnn {
namespace ocl {

struct gemm_impl : typed_primitive_impl_ocl<gemm> {
    using parent = typed_primitive_impl_ocl<gemm>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::gemm_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::gemm_params, kernel_selector::gemm_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::gemm_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gemm_impl>(*this);
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
    static size_t get_beam_table_id(std::shared_ptr<const gemm> primitive) {
        return primitive->input_size() == 3 ? 3 : 2;
    }

    kernel_arguments_data get_arguments(const typed_primitive_inst<gemm>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);
        const auto& desc = instance.get_typed_desc<gemm>();

        if (desc->indirect_a || desc->indirect_b) {
            args.inputs.push_back(instance.dep_memory_ptr(get_beam_table_id(desc)));
        }

        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<gemm>();

        auto params = get_default_params<kernel_selector::gemm_params>(impl_param, is_shape_agnostic);
        auto optional_params = get_default_optional_params<kernel_selector::gemm_optional_params>(impl_param.get_program());

        for (size_t i = 1; i < primitive->input_size(); ++i) {
            params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[i]));
        }

        params.alpha = primitive->alpha;
        params.beta = primitive->beta;
        params.transpose_input0 = primitive->transpose_input0;
        params.transpose_input1 = primitive->transpose_input1;
        params.input0_order = primitive->input0_order;
        params.input1_order = primitive->input1_order;
        params.output_order = primitive->output_order;

        params.indirect_input0 = primitive->indirect_a;
        params.indirect_input1 = primitive->indirect_b;
        if (primitive->indirect_a || primitive->indirect_b) {
            OPENVINO_ASSERT(impl_param.input_layouts.size() >= 3, "[GPU] Actual inputs count: ", impl_param.input_layouts.size());
            params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[get_beam_table_id(primitive)]));
        }

        bool is_quantized = true;
        for (auto& input : impl_param.input_layouts)
            is_quantized &= data_type_traits::is_quantized(input.data_type);

        if (is_quantized) {
            params.quantization = kernel_selector::QuantizationType::SYMMETRIC;
        } else {
            params.quantization = kernel_selector::QuantizationType::NONE;
        }
        return {params, optional_params};
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        const auto& primitive = impl_params.typed_desc<gemm>();
        auto updated_impl_params = canonicalize_fused_shapes(impl_params);

        updated_impl_params.input_layouts = gemm_inst::transform_input_layouts(primitive, impl_params.input_layouts);
        updated_impl_params.output_layouts[0] = gemm_inst::transform_output_layout(primitive, updated_impl_params.input_layouts, impl_params.output_layouts[0]);

        for (auto& input_layout : updated_impl_params.input_layouts) {
            input_layout.set_partial_shape(extend_shape_to_rank_from_begin(input_layout.get_partial_shape()));
        }

        auto& output_layout = updated_impl_params.output_layouts[0];
        output_layout.set_partial_shape(extend_shape_to_rank_from_begin(output_layout.get_partial_shape()));

        return updated_impl_params;
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const override {
        return static_canonicalize_shapes(impl_params);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
    }
};

namespace detail {

attach_gemm_impl::attach_gemm_impl() {
    const std::vector<data_types> types{data_types::f16,
                                        data_types::f32,
                                        data_types::i8,
                                        data_types::u8,
                                        data_types::i32};

    const std::vector<format::type> formats {
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,

        format::bfzyx,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,
        format::bs_fs_zyx_bsv32_fsv16,

        format::bfwzyx,
    };

    implementation_map<gemm>::add(impl_types::ocl, shape_types::static_shape, typed_primitive_impl_ocl<gemm>::create<gemm_impl>, types, formats);

    const std::vector<format::type> dyn_formats {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    implementation_map<gemm>::add(impl_types::ocl,
                                  shape_types::dynamic_shape,
                                  typed_primitive_impl_ocl<gemm>::create<gemm_impl>, types, dyn_formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::gemm_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::gemm)
