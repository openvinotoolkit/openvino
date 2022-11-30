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
            auto get_updated_input_shape = [&](const ov::PartialShape& input_pshape, size_t input_rank, bool transpose, bool first_input) {
                ov::PartialShape updated_input_pshape;

                if (input_rank == 1) {
                    if (input_pshape.is_static()) {
                        auto input_shape = input_pshape.to_shape();
                        updated_input_pshape = ov::PartialShape{ static_cast<int64_t>(*std::max_element(input_shape.begin(), input_shape.end())) };
                    } else {
                        updated_input_pshape = ov::PartialShape::dynamic(input_rank);
                    }
                } else {
                    if (input_pshape.is_static()) {
                        OPENVINO_ASSERT(input_pshape.size() >= input_rank, "[GPU] Requested input rank in gemm primitive is greater than actual shape");
                        std::vector<ov::Dimension> dims(input_pshape.begin(), input_pshape.begin() + input_rank);
                        updated_input_pshape = ov::PartialShape(dims);
                    } else {
                        updated_input_pshape = input_pshape;
                    }
                }

                if (updated_input_pshape.size() == 1) {
                    first_input ? updated_input_pshape.insert(updated_input_pshape.begin(), 1)
                                : updated_input_pshape.insert(updated_input_pshape.end(), 1);

                    if (transpose) {
                        std::swap(updated_input_pshape[0], updated_input_pshape[1]);
                    }
                }
                size_t ones_to_add = std::max(output_layout.get_partial_shape().size(), static_cast<size_t>(4)) - updated_input_pshape.size();
                updated_input_pshape.insert(updated_input_pshape.begin(), ones_to_add, 1ul);

                return updated_input_pshape;
            };

            auto input0_pshape = input_layouts[0].get_partial_shape();
            auto input1_pshape = input_layouts[1].get_partial_shape();

            auto updated_input0_pshape = get_updated_input_shape(input0_pshape, primitive->input_rank, primitive->transpose_input0, true);
            auto updated_input1_pshape = get_updated_input_shape(input1_pshape, primitive->weight_rank, primitive->transpose_input1, false);

            std::vector<layout> layouts = input_layouts;
            layouts[0].set_partial_shape(updated_input0_pshape);
            layouts[1].set_partial_shape(updated_input1_pshape);

            if (input_layouts.size() == 3) {
                auto bias_pshape = input_layouts[2].get_partial_shape();
                auto updated_bias_pshape = get_updated_input_shape(bias_pshape, primitive->weight_rank, primitive->transpose_input1, false);
                layouts[2].set_partial_shape(updated_bias_pshape);
            }

            return layouts;
        };

        auto get_gemm_output_layout = [primitive](const std::vector<layout>& input_layouts, const layout& output_layout) {
            auto updated_output_layout = output_layout;
            auto output_rank = output_layout.get_partial_shape().size();
            if (output_rank < 4) {
                auto input0_pshape = input_layouts[0].get_partial_shape();
                auto input1_pshape = input_layouts[1].get_partial_shape();

                auto M = !primitive->transpose_input0 ? input0_pshape[input0_pshape.size() - 2] : input0_pshape[input0_pshape.size() - 1];
                auto N = !primitive->transpose_input1 ? input1_pshape[input1_pshape.size() - 1] : input1_pshape[input1_pshape.size() - 2];

                auto output_pshape = input_layouts[0].get_partial_shape();
                for (size_t i = 0; i != input_layouts.size(); ++i) {
                    auto input_pshape = input_layouts[i].get_partial_shape();
                    for (size_t j = 0; j != input_pshape.size(); ++j) {
                        ov::Dimension::merge(output_pshape[j], output_pshape[j], input_pshape[j]);
                    }
                }

                auto get_spatial_idx = [](cldnn::format format, size_t spatial_idx) {
                    const size_t idx = (format::is_grouped(format) ? 3 : 2) + (format.spatial_num() - 1 - spatial_idx);
                    return idx;
                };

                output_pshape[get_spatial_idx(updated_output_layout.format, 0)] = N;
                output_pshape[get_spatial_idx(updated_output_layout.format, 1)] = M;
                updated_output_layout.set_partial_shape(output_pshape);
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

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param);
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
