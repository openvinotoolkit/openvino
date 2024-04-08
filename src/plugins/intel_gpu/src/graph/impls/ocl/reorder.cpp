// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "reorder_inst.h"
#include "reorder/reorder_kernel_selector.h"
#include "reorder/reorder_kernel_base.h"
#include "reorder/reorder_weights_kernel_selector.h"

namespace cldnn {
namespace ocl {

struct reorder_impl : typed_primitive_impl_ocl<reorder> {
    using parent = typed_primitive_impl_ocl<reorder>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::reorder_kernel_selector;
    using kernel_params_t = kernel_selector::reorder_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::reorder_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<reorder_impl>(*this);
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
    kernel_arguments_data get_arguments(const reorder_inst& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);
        if (instance.has_node() && instance.has_mean()) {
            auto input = &instance.input_memory();
            auto input_layout = input->get_layout();

            if (input_layout.format == cldnn::format::nv12) {
                args.bias = instance.mean_nv12_memory();
            } else {
                args.bias = instance.mean_memory();
            }
        }
        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<reorder>();
        auto&& output_layout = impl_param.get_output_layout();
        auto params = get_default_params<kernel_selector::reorder_params>(impl_param, is_shape_agnostic);

        auto inputs_count = primitive->input.size();
        bool has_mean = !primitive->mean.empty();
        for (size_t i = 1; i < inputs_count; i++) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }
        if (impl_param.get_output_layout().data_padding) {
            params.has_padded_output = true;
        }

        params.surface_input = primitive->has_surface_input();

        if (has_mean) {
            if (impl_param.get_input_layout(0).format == cldnn::format::nv12) {
                const auto& mean_layout = impl_param.get_input_layout(2);
                params.mean = convert_data_tensor(mean_layout);
                params.mode = kernel_selector::mean_subtruct_mode::IN_BUFFER;
            } else {
                const auto mean_idx = 1;
                const auto& mean_layout = impl_param.get_input_layout(mean_idx);
                params.mean = convert_data_tensor(mean_layout);
                params.mode = kernel_selector::mean_subtruct_mode::IN_BUFFER;
            }
        } else if (primitive->subtract_per_feature.empty() == false) {
            params.mode = kernel_selector::mean_subtruct_mode::INSIDE_PARAMS;
            params.meanValues = primitive->subtract_per_feature;
        } else {
            params.mode = kernel_selector::mean_subtruct_mode::NONE;
        }

        if (params.mode != kernel_selector::mean_subtruct_mode::NONE) {
            switch (primitive->mean_mode) {
                case reorder_mean_mode::none:
                    params.mean_op = kernel_selector::mean_op::NONE;
                    break;
                case reorder_mean_mode::mul:
                    params.mean_op = kernel_selector::mean_op::MUL;
                    break;
                case reorder_mean_mode::subtract:
                    params.mean_op = kernel_selector::mean_op::SUB;
                    break;
                case reorder_mean_mode::div:
                    params.mean_op = kernel_selector::mean_op::DIV;
                    break;
                default: OPENVINO_ASSERT(false, "[GPU] Unsupported mean_mode value in primitive ", primitive->id);
            }
        }

        if (output_layout.format == format::winograd_2x3_s1_data) {
            params.winograd_input_offset_x = 0;
            params.winograd_input_offset_y = 0;
            params.winograd_nr_tiles_x = ceil_div(output_layout.spatial(0), 4);
        }

        params.winograd = impl_param.input_layouts[0].format.is_winograd() || output_layout.format.is_winograd();
        params.truncate = impl_param.typed_desc<reorder>()->truncate;

        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params, _kernel_data);
    }

    static std::unique_ptr<primitive_impl> create(const reorder_node& arg, const kernel_impl_params& impl_param) {
        bool is_reorder_weights = format::is_weights_format(impl_param.get_input_layout().format) ||
                                  format::is_weights_format(impl_param.get_output_layout().format);
        if (is_reorder_weights) {
            return create_reorder_weights(impl_param);
        } else {
            return typed_primitive_impl_ocl<reorder>::create<reorder_impl>(arg, impl_param);
        }
    }

    static std::unique_ptr<primitive_impl> create_reorder_weights(const kernel_impl_params& impl_param) {
        const auto& prim = impl_param.typed_desc<reorder>();
        const auto& weights_params = prim->weights_reorder_params;
        auto& kernel_selector = kernel_selector::ReorderWeightsKernelSelector::Instance();

        OPENVINO_ASSERT(weights_params != nullptr, "[GPU] Attempt to create reorder weights without weights params");

        OPENVINO_ASSERT(impl_param.get_input_layout().bytes_count() == weights_params->get_input_layout().bytes_count(),
                        "[GPU] Input layout doesn't match required reorder weights layout");

        kernel_selector::reorder_weights_params r_params;
        set_params(impl_param, r_params);

        r_params.input = convert_weights_tensor(weights_params->get_input_layout(), weights_params->get_grouped());
        r_params.output = convert_weights_tensor(weights_params->get_output_layout());
        r_params.layerID = impl_param.desc->id + "_reorder_weights";
        r_params.uniqueID = std::to_string(impl_param.unique_id) + "_weight";
        r_params.rotate_180 = weights_params->should_be_transposed();

        auto best_kernel = kernel_selector.get_best_kernel(r_params);

        return make_unique<reorder_impl>(best_kernel);
    }
};

namespace detail {

attach_reorder_impl::attach_reorder_impl() {
    implementation_map<reorder>::add(impl_types::ocl, shape_types::static_shape, reorder_impl::create, {});

    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
        data_types::i32,
        data_types::i64,
    };

    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };
    implementation_map<reorder>::add(impl_types::ocl, shape_types::dynamic_shape, reorder_impl::create, types, formats);

    WeightsReordersFactory::add(cldnn::impl_types::ocl, shape_types::static_shape, reorder_impl::create_reorder_weights);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::reorder_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::reorder)
