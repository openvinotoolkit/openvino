// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm/gemm_kernel_base.h"
#include "gemm/gemm_kernel_selector.h"
#include "gemm_inst.h"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include <algorithm>
#include "kernel_selector_helper.h"
#include "primitive_base.hpp"

namespace cldnn {
namespace ocl {

struct gemm_impl : typed_primitive_impl_ocl<gemm> {
    using parent = typed_primitive_impl_ocl<gemm>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::gemm_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::gemm_params, kernel_selector::gemm_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gemm_impl>(*this);
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<gemm>();
        auto get_gemm_input_layouts = [primitive](const std::vector<layout>& input_layouts, const layout& output_layout) {
            auto get_updated_input_shape = [&](const ov::Shape& input_shape, size_t input_rank, bool transpose, bool first_input) {
                ov::Shape updated_input_shape;

                if (input_rank == 1) {
                    updated_input_shape = { *std::max_element(input_shape.begin(), input_shape.end()) };
                } else {
                    updated_input_shape = ov::Shape(input_shape.begin(), input_shape.begin() + input_rank);
                }

                if (updated_input_shape.size() == 1) {
                    first_input ? updated_input_shape.insert(updated_input_shape.begin(), 1)
                                : updated_input_shape.insert(updated_input_shape.end(), 1);

                    if (transpose) {
                        std::swap(updated_input_shape[0], updated_input_shape[1]);
                    }
                }
                size_t ones_to_add = std::max(output_layout.get_shape().size(), static_cast<size_t>(4)) - updated_input_shape.size();
                updated_input_shape.insert(updated_input_shape.begin(), ones_to_add, 1ul);

                return updated_input_shape;
            };

            auto input0_shape = input_layouts[0].get_shape();
            auto input1_shape = input_layouts[1].get_shape();

            auto updated_input0_shape = get_updated_input_shape(input0_shape, primitive->input_rank, primitive->transpose_input0, true);
            auto updated_input1_shape = get_updated_input_shape(input1_shape, primitive->weight_rank, primitive->transpose_input1, false);

            std::vector<layout> layouts = input_layouts;
            layouts[0].set_partial_shape(updated_input0_shape);
            layouts[1].set_partial_shape(updated_input1_shape);

            if (input_layouts.size() == 3) {
                auto bias_shape = input_layouts[2].get_shape();
                auto updated_bias_shape = get_updated_input_shape(bias_shape, primitive->weight_rank, primitive->transpose_input1, false);
                layouts[2].set_partial_shape(updated_bias_shape);
            }

            return layouts;
        };

        auto get_gemm_output_layout = [primitive](const std::vector<layout>& input_layouts, const layout& output_layout) {
            auto updated_output_layout = output_layout;
            auto output_rank = output_layout.get_shape().size();
            if (output_rank < 4) {
                const auto& input0_layout = input_layouts[0];
                const auto& input1_layout = input_layouts[1];

                auto M = !primitive->transpose_input0 ? input0_layout.spatial(1) : input0_layout.spatial(0);
                auto N = !primitive->transpose_input1 ? input1_layout.spatial(0) : input1_layout.spatial(1);

                auto output_shape = input0_layout.get_shape();
                for (const auto& input_layout : input_layouts) {
                    auto input_shape = input_layout.get_shape();
                    for (size_t i = 0; i != input_shape.size(); ++i) {
                        output_shape[i] = std::max(output_shape[i], input_shape[i]);
                    }
                }

                auto get_spatial_idx = [](cldnn::format format, size_t spatial_idx) {
                    const size_t idx = (format::is_grouped(format) ? 3 : 2) + (format.spatial_num() - 1 - spatial_idx);
                    return idx;
                };

                output_shape[get_spatial_idx(updated_output_layout.format, 0)] = N;
                output_shape[get_spatial_idx(updated_output_layout.format, 1)] = M;
                updated_output_layout.set_partial_shape(output_shape);
            }
            return updated_output_layout;
        };

        const auto input_layouts = get_gemm_input_layouts(impl_param.input_layouts, impl_param.output_layouts[0]);
        const auto output_layout = get_gemm_output_layout(input_layouts, impl_param.output_layouts[0]);

        auto params = get_default_params<kernel_selector::gemm_params>(impl_param, 1);
        auto optional_params = get_default_optional_params<kernel_selector::gemm_optional_params>(impl_param.get_program());

        params.inputs.clear();
        for (size_t i = 0; i < primitive->input_size(); ++i) {
            params.inputs.push_back(convert_data_tensor(input_layouts[i]));
        }
        params.outputs[0] = convert_data_tensor(output_layout);

        params.alpha = primitive->alpha;
        params.beta = primitive->beta;
        params.transpose_input0 = primitive->transpose_input0;
        params.transpose_input1 = primitive->transpose_input1;

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

    implementation_map<gemm>::add(impl_types::ocl, typed_primitive_impl_ocl<gemm>::create<gemm_impl>, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::gemm_impl, cldnn::object_type::GEMM_IMPL)