// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "rope_inst.h"
#include "rope/rope_kernel_selector.h"
#include "rope/rope_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct rope_impl : typed_primitive_impl_ocl<rope> {
    using parent = typed_primitive_impl_ocl<rope>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::rope_kernel_selector;
    using kernel_params_t = kernel_selector::rope_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::rope_impl);

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<rope_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<rope>();
        auto params = get_default_params<kernel_selector::rope_params>(impl_param, is_shape_agnostic);

        params.head_cnt = primitive->config.head_cnt;
        params.head_size = primitive->config.head_size;
        params.rotary_ndims = primitive->config.rotary_ndims;
        params.gather_rank = primitive->gather_rank;

        params.slice_start = primitive->config.slice_start;
        params.slice_stop = primitive->config.slice_stop;

        params.axis = primitive->config.is_qwen || primitive->config.is_chatglm ? 2 : 3;
        params.num_of_inputs = primitive->config.is_chatglm || primitive->config.is_interleaved ? 2 : 3;

        if (params.gather_rank > 0) {
            params.num_of_inputs++;
        }

        params.is_qwen = primitive->config.is_qwen;
        params.is_chatglm = primitive->config.is_chatglm;
        params.support_2d_rope = primitive->config.support_2d_rope;
        params.transposed_input = primitive->config.input_trans0213;

        for (size_t i = 1; i < impl_param.input_layouts.size(); ++i) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }
        return params;
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        const auto& primitive = impl_params.typed_desc<rope>();

        if (primitive->config.is_chatglm || primitive->config.is_qwen) {
            return primitive_impl::static_canonicalize_shapes(impl_params);
        } else {
            auto updated_impl_params = canonicalize_fused_shapes(impl_params);

            std::set<size_t> canonicalize_from_begin = { 1, 2 };
            for (size_t i = 0; i < updated_impl_params.input_layouts.size(); ++i) {
                auto& input_layout = updated_impl_params.input_layouts[i];
                if (canonicalize_from_begin.count(i) != 0) {
                    input_layout.set_partial_shape(extend_shape_to_rank_from_begin(input_layout.get_partial_shape()));
                } else {
                    input_layout.set_partial_shape(extend_shape_to_rank_from_end(input_layout.get_partial_shape()));
                }
            }

            auto& output_layout = updated_impl_params.output_layouts[0];
            output_layout.set_partial_shape(extend_shape_to_rank_from_end(output_layout.get_partial_shape()));

            return updated_impl_params;
        }
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

attach_rope_impl::attach_rope_impl() {
    auto types = {
        data_types::f32,
        data_types::f16
    };

    auto formats = {
        format::bfyx
    };

    implementation_map<rope>::add(impl_types::ocl,
                                  shape_types::any,
                                  typed_primitive_impl_ocl<rope>::create<rope_impl>,
                                  types,
                                  formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::rope_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::rope)
