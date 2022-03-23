// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_inst.h"
#include "primitive_onednn_base.h"
#include "impls/implementation_map.hpp"

#include "kernel_selector_common.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

struct fully_connected_onednn : typed_primitive_onednn_impl<fully_connected, dnnl::inner_product_forward::desc> {
    using parent = typed_primitive_onednn_impl<fully_connected, dnnl::inner_product_forward::desc>;
    using parent::parent;

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<fully_connected_onednn>(*this);
    }

    bool validate_impl(const typed_primitive_inst<fully_connected>& instance) const override {
        bool res = true;

        auto outer_id = _outer.id();
        auto data_type = instance.node.input().get_output_layout().data_type;

        // Integer signed/unsigned is ok for fully connected
        CLDNN_ERROR_DATA_TYPES_MISMATCH_IGNORE_SIGN(outer_id,
                                                    "Input memory",
                                                    data_type,
                                                    "filter memory",
                                                    instance.weights_memory()->get_layout().data_type,
                                                    "");

        return res;
    }

    std::unordered_map<int, dnnl::memory> get_arguments(fully_connected_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args = parent::get_arguments(instance);

        {
            auto weights = instance.weights_memory();
            args.insert({DNNL_ARG_WEIGHTS, weights->get_onednn_memory(_pd.weights_desc(0))});
        }

        if (instance.bias_term()) {
            auto bias = instance.bias_memory();
            args.insert({DNNL_ARG_BIAS, bias->get_onednn_memory(_pd.weights_desc(1))});
        }

        return args;
    }

    static kernel_selector::WeightsReorderParams get_weights_reorder(const fully_connected_node& arg, const dnnl::primitive_desc& pd) {
        auto weights_layout = arg.get_dependency(1).get_output_layout();
        auto cldnn_prim = arg.get_primitive();
        const auto& bias_layout = arg.bias_term() ?  arg.bias().get_output_layout() : layout(data_types::f32, format::any, tensor());
        const auto& param_info = kernel_impl_params(arg.get_program(), cldnn_prim, arg.get_unique_id(),
                                                    arg.get_input_layouts(), arg.get_output_layout(),
                                                    arg.get_fused_primitives(),
                                                    arg.get_fused_activations_funcs(), arg.get_fused_activations_params(),
                                                    weights_layout, arg.bias_term(), bias_layout);
        kernel_selector::WeightsReorderParams weights_reorder_params;
        auto& reorderKS = kernel_selector::ReorderWeightsKernelSelctor::Instance();
        kernel_selector::reorder_weights_params r_params;

        cldnn::format out_fmt = onednn::find_format(pd.weights_desc(0));
        kernel_selector::WeightsLayout req_layout = to_weights_layout(out_fmt, false);

        // set engine info & forcing
        set_params(param_info, r_params);
        r_params.layerID = arg.id() + "_reorder_";
        r_params.input = convert_weights_tensor(weights_layout, false);
        r_params.output = r_params.input.TransformIgnorePadding(req_layout, r_params.input.GetDType(), 1, false);
        r_params.rotate_180 = false;

        kernel_selector::reorder_optional_params op;
        kernel_selector::KernelsData kernels_data = reorderKS.GetBestKernels(r_params, op);

        if (kernels_data.empty()) {
            throw std::runtime_error("No suitable kernel found for weights reorder from " +
                                      kernel_selector::toString(r_params.input.GetLayout()) + " to " +
                                      kernel_selector::toString(r_params.output.GetLayout()));
        }

        weights_reorder_params.engine = kernel_selector::WeightsReorderParams::Engine::GPU;
        weights_reorder_params.clKernel = std::make_shared<kernel_selector::clKernelData>(kernels_data[0].kernels[0]);
        weights_reorder_params.dest = r_params.output;

        return weights_reorder_params;
    }

    static std::shared_ptr<dnnl::inner_product_forward::desc> get_fully_connected_descriptor(const fully_connected_node& arg) {
        auto prim = arg.get_primitive();

        auto& input = arg.get_dependency(0);
        auto& weights = arg.get_dependency(1);
        auto input_layout = input.get_output_layout();
        auto output_layout = arg.get_output_layout();

        if (prim->input_size == 3) {
            combine_bf_with_first_spatial_dim(input_layout);
            combine_bf_with_first_spatial_dim(output_layout);
        }

        auto input_md = onednn::layout_to_memory_desc(input_layout, dnnl::memory::format_tag::undef, false);
        auto weights_md = onednn::layout_to_memory_desc(weights.get_output_layout(), dnnl::memory::format_tag::any);
        auto output_md = onednn::layout_to_memory_desc(output_layout, dnnl::memory::format_tag::ab, false);

        if (arg.bias_term()) {
            auto bias_md = onednn::layout_to_memory_desc(arg.get_dependency(2).get_output_layout(), dnnl::memory::format_tag::any, true);
            return std::make_shared<dnnl::inner_product_forward::desc>(
                dnnl::prop_kind::forward_inference,
                input_md,
                weights_md,
                bias_md,
                output_md);
        } else {
            return std::make_shared<dnnl::inner_product_forward::desc>(
                dnnl::prop_kind::forward_inference,
                input_md,
                weights_md,
                output_md);
        }
    }

public:
    static primitive_impl* create(const fully_connected_node& arg) {
        auto& engine = arg.get_program().get_engine();
        auto desc = get_fully_connected_descriptor(arg);
        auto attr = arg.get_onednn_primitive_attributes();
        dnnl::primitive_desc prim_desc{&desc->data, attr.get(), engine.get_onednn_engine(), nullptr};

        return new fully_connected_onednn(arg, desc, attr, prim_desc, get_weights_reorder(arg, prim_desc));
    }
};

namespace detail {

attach_fully_connected_onednn::attach_fully_connected_onednn() {
    implementation_map<fully_connected>::add(impl_types::onednn, fully_connected_onednn::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),

        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
    });
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn
