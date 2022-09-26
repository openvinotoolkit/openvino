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

        auto outer_id = instance.id();
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

    static kernel_selector::WeightsReorderParams get_weights_reorder(const kernel_impl_params& impl_params, const dnnl::primitive_desc& pd) {
        auto weights_layout = impl_params.get_input_layout(1);
        auto cldnn_prim = impl_params.typed_desc<fully_connected>();

        kernel_selector::WeightsReorderParams weights_reorder_params;
        auto& reorderKS = kernel_selector::ReorderWeightsKernelSelctor::Instance();
        kernel_selector::reorder_weights_params r_params;

        cldnn::format out_fmt = onednn::find_format(pd.weights_desc(0));
        kernel_selector::WeightsLayout req_layout = to_weights_layout(out_fmt, false);

        // set engine info & forcing
        set_params(impl_params, r_params);
        r_params.layerID = cldnn_prim->id + "_reorder_";
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

    static std::shared_ptr<dnnl::inner_product_forward::desc> get_fully_connected_descriptor( const kernel_impl_params& impl_params) {
        auto prim = impl_params.typed_desc<fully_connected>();

        auto input_layout = impl_params.get_input_layout(0);
        auto weights_layout = impl_params.get_input_layout(1);
        auto output_layout = impl_params.output_layout;

        if (prim->input_size == 3) {
            combine_bf_with_first_spatial_dim(input_layout);
            combine_bf_with_first_spatial_dim(output_layout);
        }

        auto input_md = onednn::layout_to_memory_desc(input_layout, dnnl::memory::format_tag::undef, false);
        auto weights_md = onednn::layout_to_memory_desc(weights_layout, dnnl::memory::format_tag::any);
        auto output_md = onednn::layout_to_memory_desc(output_layout, dnnl::memory::format_tag::ab, false);

        if (!prim->bias.empty()) {
            auto bias_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(2), dnnl::memory::format_tag::any, true);
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
    static primitive_impl* create(const fully_connected_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog.get_engine();
        auto desc = get_fully_connected_descriptor(impl_params);
        auto attr = arg.get_onednn_primitive_attributes();
        dnnl::primitive_desc prim_desc{&desc->data, attr.get(), engine.get_onednn_engine(), nullptr};

        return new fully_connected_onednn(engine, desc, attr, prim_desc, get_weights_reorder(impl_params, prim_desc));
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
