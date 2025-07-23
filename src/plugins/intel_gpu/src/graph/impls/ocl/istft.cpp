// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "istft/istft_kernel_base.h"
#include "istft/istft_kernel_selector.h"
#include "istft_inst.h"
#include "primitive_base.hpp"

namespace cldnn {
namespace ocl {

struct ISTFT_impl : typed_primitive_impl_ocl<ISTFT> {
    using parent = typed_primitive_impl_ocl<ISTFT>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::ISTFT_kernel_selector;
    using kernel_params_t = kernel_selector::ISTFT_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::ISTFT_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<ISTFT_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }

        update_shapes(*_kernel_data.params, impl_param);
        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<ISTFT>();
        auto params = get_default_params<kernel_selector::ISTFT_params>(impl_param, shape_agnostic);

        // Manually add all inputs except first one, since get_default_params does not handle it.
        for (size_t i = 1; i < impl_param.input_layouts.size(); ++i) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }

        params.center = primitive->center;
        params.normalized = primitive->normalized;

        return params;
    }

    // [NOTE]: Has to be added as a separete static function, since it is called via static dispatching in
    // typed_primitive_impl_ocl::create()..
    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        auto updated_impl_params = canonicalize_fused_shapes(impl_params);

        for (auto& input_layout : updated_impl_params.input_layouts) {
            input_layout.set_partial_shape(extend_shape_to_rank_from_begin(input_layout.get_partial_shape()));
        }

        for (auto& output_layout : updated_impl_params.output_layouts) {
            output_layout.set_partial_shape(extend_shape_to_rank_from_begin(output_layout.get_partial_shape()));
        }

        return updated_impl_params;
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const override {
        return static_canonicalize_shapes(impl_params);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, typed_primitive_inst<ISTFT>& instance) override {
        stream& stream = instance.get_network().get_stream();
        // This is needed to clear the output memory before executing the kernel for static shapes model.
        // Ref kernel assumes that output memory is already cleared.
        stream.enqueue_barrier();
        auto output_evt = instance.output_memory(0).fill(stream, {}, false);
        std::vector<event::ptr> ext_events(events);
        ext_events.push_back(output_evt);
        return parent::execute_impl(ext_events, instance);
    }
};

namespace detail {

attach_ISTFT_impl::attach_ISTFT_impl() {
    auto types = {data_types::i32, data_types::i64, data_types::f16, data_types::f32};

    auto formats = {format::bfyx};

    implementation_map<ISTFT>::add(impl_types::ocl,
                                   shape_types::any,
                                   typed_primitive_impl_ocl<ISTFT>::create<ISTFT_impl>,
                                   types,
                                   formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::ISTFT_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ISTFT)
