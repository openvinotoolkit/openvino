// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "mvn.hpp"
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

std::unique_ptr<primitive_impl> MVNImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<mvn>());
    return typed_primitive_impl_ocl<mvn>::create<mvn_impl>(static_cast<const mvn_node&>(node), params);
}

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::mvn_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::mvn)
