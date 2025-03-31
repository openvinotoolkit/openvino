// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "rms_inst.h"
#include "rms/rms_kernel_selector.h"
#include "rms/rms_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct rms_impl : typed_primitive_impl_ocl<rms> {
    using parent = typed_primitive_impl_ocl<rms>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::rms_kernel_selector;
    using kernel_params_t = kernel_selector::rms_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::rms_impl);

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<rms_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic() && _kernel_data.kernelName.length() != 0) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    static void set_padding(const kernel_impl_params& impl_param, kernel_selector::rms_params* rms_params) {
        rms_params->dynamic_padding = impl_param.get_input_layout().data_padding.is_dynamic();
        if (rms_params->dynamic_padding && rms_params->slice_stride == 0) {
            const auto& input_layout = impl_param.get_input_layout();
            const auto& dynamic_pad_mask = input_layout.data_padding._dynamic_dims_mask;

            // To find out which item is in `dynamic_pad_mask`, you can iterate over it
            size_t mask_idx = -1;
            for (size_t i = 0; i < dynamic_pad_mask.size(); i++) {
                if (dynamic_pad_mask[i]) {
                    mask_idx = i;
                    break;
                }
            }
            OPENVINO_ASSERT(mask_idx != static_cast<size_t>(-1), "Dynamic pad mask is empty");
            const auto& pshape = input_layout.get_partial_shape().to_shape();
            size_t x_dim = 1;
            for (size_t i = mask_idx + 1; i < pshape.size(); i++) {
                x_dim *= pshape[i];
            }

            const auto& data_padding = input_layout.data_padding;
            const auto& lower_pads = data_padding._lower_size;
            const auto& upper_pads = data_padding._upper_size;
            rms_params->slice_start = lower_pads[mask_idx];
            rms_params->slice_stop = lower_pads[mask_idx] + pshape[mask_idx];
            rms_params->slice_stride = (lower_pads[mask_idx] + pshape[mask_idx] + upper_pads[mask_idx]) * x_dim;
        }
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<rms>();
        auto params = get_default_params<kernel_selector::rms_params>(impl_param, is_shape_agnostic);

        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        params.epsilon = primitive->epsilon;
        params.ov_input_rank = static_cast<int32_t>(impl_param.get_input_layout().get_partial_shape().size());
        params.dynamic_padding = impl_param.get_input_layout().data_padding.is_dynamic();
        set_padding(impl_param, &params);
        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }

        auto rms_params = static_cast<kernel_selector::rms_params*>(_kernel_data.params.get());
        set_padding(impl_param, rms_params);

        update_shapes(*_kernel_data.params, impl_param);
        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        return impl_params;
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const override {
        return static_canonicalize_shapes(impl_params);
    }
};

namespace detail {

attach_rms_impl::attach_rms_impl() {
    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i32
    };

    auto formats = {
        format::bfyx,
        format::bfzyx
    };

    implementation_map<rms>::add(impl_types::ocl,
                                 shape_types::any,
                                 typed_primitive_impl_ocl<rms>::create<rms_impl>,
                                 types,
                                 formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::rms_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::rms)
