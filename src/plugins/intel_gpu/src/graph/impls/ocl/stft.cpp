// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"
#include "stft/stft_kernel_base.h"
#include "stft/stft_kernel_selector.h"
#include "stft_inst.h"

namespace cldnn {
namespace ocl {

struct STFT_impl : typed_primitive_impl_ocl<STFT> {
    using parent = typed_primitive_impl_ocl<STFT>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::STFT_kernel_selector;
    using kernel_params_t = kernel_selector::STFT_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::STFT_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<STFT_impl>(*this);
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
        const auto& primitive = impl_param.typed_desc<STFT>();
        auto params = get_default_params<kernel_selector::STFT_params>(impl_param, shape_agnostic);

        // Manually add all inputs except first one, since get_default_params does not handle it.
        for (size_t i = 1; i < impl_param.input_layouts.size(); ++i) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }

        params.transpose_frames = primitive->transpose_frames;
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
};

namespace detail {

attach_STFT_impl::attach_STFT_impl() {
    auto types = {data_types::i32, data_types::i64, data_types::f16, data_types::f32};

    auto formats = {format::bfyx};

    implementation_map<STFT>::add(impl_types::ocl,
                                  shape_types::any,
                                  typed_primitive_impl_ocl<STFT>::create<STFT_impl>,
                                  types,
                                  formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::STFT_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::STFT)
