// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_elt_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "lstm/lstm_elt_kernel_selector.h"
#include "lstm/lstm_elt_kernel_base.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct lstm_elt_impl : typed_primitive_impl_ocl<lstm_elt> {
    using parent = typed_primitive_impl_ocl<lstm_elt>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::lstm_elt_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::lstm_elt_params, kernel_selector::lstm_elt_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<lstm_elt_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<lstm_elt>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);

        args.cell = instance.cell_term() ? instance.cell_memory() : nullptr;
        args.outputs = { instance.output_memory_ptr() };

        return args;
    }

public:
    static std::unique_ptr<primitive_impl> create(const lstm_elt_node& arg, const kernel_impl_params& impl_param) {
        const auto& prim = arg.get_primitive();
        auto lstm_elt_params = get_default_params<kernel_selector::lstm_elt_params>(impl_param);
        auto lstm_elt_optional_params =
            get_default_optional_params<kernel_selector::lstm_elt_optional_params>(arg.get_program());

        if (arg.cell_term()) {
            const auto& cell_idx = 1;
            const auto& cell_layout = impl_param.input_layouts[cell_idx];
            lstm_elt_params.SetCell(convert_data_tensor(cell_layout));
            // TODO: make a generic function to get the direction
            if (cell_layout.spatial(1) > 1) {
                lstm_elt_params.cell_direction = arg.direction();
            }
        }

        if (!prim->activations.empty()) {
            auto a_sz = prim->activations.size();
            auto param_sz = prim->activation_params.size();
            if (param_sz) {
                CLDNN_ERROR_NOT_EQUAL(arg.id(),
                                      "number of activations",
                                      a_sz,
                                      "number of activation parameters",
                                      param_sz,
                                      "activations/parameters num mismatch");
            }
            for (size_t i = 0; i < a_sz; i++) {
                lstm_elt_params.activations.emplace_back(get_kernel_selector_activation_param(prim->activations[i]),
                                                         param_sz ? prim->activation_params[i].a : 0.0f,
                                                         param_sz ? prim->activation_params[i].b : 0.0f);
            }
        }

        if (prim->clip > 0.0f) {
            lstm_elt_params.activations.emplace_back(get_kernel_selector_activation_param(activation_func::clamp), -prim->clip, prim->clip);
        }

        lstm_elt_params.SetOffsetOrder(static_cast<int32_t>(arg.offset_order()));
        lstm_elt_params.clip = arg.clip();
        lstm_elt_params.input_forget = arg.input_forget();
        lstm_elt_params.direction = arg.direction();

        auto& kernel_selector = kernel_selector::lstm_elt_kernel_selector::Instance();
        auto best_kernel = kernel_selector.get_best_kernel(lstm_elt_params, lstm_elt_optional_params);

        return make_unique<lstm_elt_impl>(arg, best_kernel);
    }
};

namespace detail {

attach_lstm_elt_impl::attach_lstm_elt_impl() {
    implementation_map<lstm_elt>::add(impl_types::ocl, lstm_elt_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f32, format::fyxb),
        std::make_tuple(data_types::f16, format::fyxb),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::lstm_elt_impl)
