// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "mvn_inst.h"
#include "mvn/mvn_kernel_selector.h"
#include "mvn/mvn_kernel_base.h"

namespace cldnn {
namespace ocl {

struct mvn_impl : typed_primitive_impl_ocl<mvn> {
    using parent = typed_primitive_impl_ocl<mvn>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::mvn_kernel_selector;
    using kernel_params_t = kernel_selector::mvn_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::mvn_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<mvn_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic() && _kernel_data.kernelName.length() != 0) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<mvn>();
        auto params = get_default_params<kernel_selector::mvn_params>(impl_param, is_shape_agnostic);

        params.mvnMode = primitive->across_channels() ? kernel_selector::mvn_mode::ACROSS_CHANNELS
                                                      : kernel_selector::mvn_mode::WITHIN_CHANNELS;
        params.mvnNormalizeVariance = primitive->normalize_variance;
        params.epsilon = primitive->epsilon;

        params.mvnEpsMode = primitive->eps_inside_sqrt ? kernel_selector::mvn_eps_mode::INSIDE_SQRT
                                                       : kernel_selector::mvn_eps_mode::OUTSIDE_SQRT;
        return params;
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        auto updated_impl_params = canonicalize_fused_shapes(impl_params);
        const auto& prim = impl_params.typed_desc<mvn>();

        auto& input_layout = updated_impl_params.input_layouts[0];
        auto input_pshape = input_layout.get_partial_shape();
        auto input_rank = input_pshape.size();

        if (prim->requires_alignment(input_pshape)) {
            auto axes = prim->reduction_axes;
            auto min_it = std::min_element(axes.begin(), axes.end());
            auto min = min_it == axes.end() ? 1 : *min_it;

            auto new_rank = std::max<size_t>(4, input_rank);
            ov::PartialShape shape = ov::PartialShape::dynamic(new_rank);

            auto& output_layout = updated_impl_params.output_layouts[0];

            if (input_pshape.is_static()) {
                size_t flatten_axis = 0;
                // Change flatten axis if the format is single fsv.
                auto block_sizes = format::block_sizes(input_layout.format);
                if (block_sizes.size() == 1
                    && (input_pshape[block_sizes[0].first].get_length() % block_sizes[0].second == 0)
                    && (std::count(axes.begin(), axes.end(), block_sizes[0].first) == 0)
                    && block_sizes[0].first == 1) {
                    flatten_axis = 1;
                }

                for (size_t i = 0; i < new_rank; i++) {
                    shape[i] = 1;
                }

                // Split all dimensions into 2 parts:
                // 1. normalized dimensions which are flattened and written to the last dim
                // 2. not normalized dims which are flattened and written to the first dim
                for (size_t i = 0; i < input_rank; i++) {
                    shape[static_cast<int64_t>(i) < min ? flatten_axis : (new_rank - 1)] *= input_pshape[i];
                }
            }

            input_layout.set_partial_shape(shape);
            output_layout.set_partial_shape(shape);
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

attach_mvn_impl::attach_mvn_impl() {
    auto dyn_types = {
        data_types::f32,
        data_types::f16,
        data_types::i8,
        data_types::u8,
        data_types::i32
    };

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx
    };

    implementation_map<mvn>::add(impl_types::ocl,
                                 shape_types::dynamic_shape,
                                 typed_primitive_impl_ocl<mvn>::create<mvn_impl>,
                                 dyn_types,
                                 dyn_formats);

    implementation_map<mvn>::add(impl_types::ocl, typed_primitive_impl_ocl<mvn>::create<mvn_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),

        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),

        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),

        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv16_fsv16),

        // TODO: uncomment this code when fsv32 optimizations for MVN will be implemented
        /*std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv32),*/

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),

        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),

        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::mvn_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::mvn)
