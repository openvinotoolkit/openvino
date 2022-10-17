// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deconvolution_inst.h"
#include "eltwise_inst.h"
#include "quantize_inst.h"
#include "primitive_onednn_base.h"
#include "impls/implementation_map.hpp"

#include "kernel_selector_common.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

struct deconvolution_onednn : typed_primitive_onednn_impl<deconvolution, dnnl::deconvolution_forward::desc> {
    using parent = typed_primitive_onednn_impl<deconvolution, dnnl::deconvolution_forward::desc>;
    using parent::parent;

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<deconvolution_onednn>(*this);
    }

    bool validate_impl(const typed_primitive_inst<deconvolution>& instance) const override {
        bool res = true;

        auto outer_id = instance.id();
        auto data_type = instance.node.input().get_output_layout().data_type;

        // Integer signed/unsigned is ok for convoluiton
        CLDNN_ERROR_DATA_TYPES_MISMATCH_IGNORE_SIGN(outer_id,
                                                    "Input memory",
                                                    data_type,
                                                    "filter memory",
                                                    instance.weights_memory(0)->get_layout().data_type,
                                                    "");

        return res;
    }

    std::unordered_map<int, dnnl::memory> get_arguments(deconvolution_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args = parent::get_arguments(instance);
        auto& engine = instance.get_network().get_engine();
        auto onednn_engine = engine.get_onednn_engine();

        {
            auto weights = instance.weights_memory(0);
            args.insert({DNNL_ARG_WEIGHTS, weights->get_onednn_memory(_pd.weights_desc(0))});
        }

        if (instance.bias_term()) {
            auto bias = instance.bias_memory(0);
            args.insert({DNNL_ARG_BIAS, bias->get_onednn_memory(_pd.weights_desc(1))});
        }

        return args;
    }

    static std::shared_ptr<dnnl::primitive_attr> get_primitive_attributes(const typed_program_node<deconvolution>& arg) {
        auto attrs = arg.get_onednn_primitive_attributes();

        return attrs;
    }

    static kernel_selector::WeightsReorderParams get_weights_reorder(const kernel_impl_params& impl_params, const dnnl::primitive_desc& pd) {
        kernel_selector::WeightsReorderParams weights_reorder_params;
        auto& reorderKS = kernel_selector::ReorderWeightsKernelSelctor::Instance();
        kernel_selector::reorder_weights_params r_params;

        auto cldnn_prim = impl_params.typed_desc<deconvolution>();
        auto weights_layout = impl_params.get_input_layout(1);
        auto grouped_weights = format::is_grouped(weights_layout.format) || cldnn_prim->grouped_weights_shape;
        cldnn::format out_fmt = onednn::find_format(pd.weights_desc(0), grouped_weights);
        kernel_selector::WeightsLayout reqLayout = to_weights_layout(out_fmt, cldnn_prim->grouped_weights_shape);

        set_params(impl_params, r_params);
        r_params.layerID = cldnn_prim->id + "_reorder_";
        r_params.input = convert_weights_tensor(weights_layout, cldnn_prim->grouped_weights_shape);
        r_params.output = r_params.input.TransformIgnorePadding(reqLayout, r_params.input.GetDType(), cldnn_prim->groups, false);
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

    static std::shared_ptr<dnnl::deconvolution_forward::desc> get_deconvolution_descriptor(const kernel_impl_params& impl_params) {
        auto prim = impl_params.typed_desc<deconvolution>();

        auto input_layout = impl_params.get_input_layout(0);
        auto weights_layout = impl_params.get_input_layout(1);
        auto output_layout = impl_params.output_layout;

        dnnl::memory::dims stride(prim->stride.begin(), prim->stride.end());
        dnnl::memory::dims dilation(input_layout.get_spatial_rank(), 1);
        dnnl::memory::dims pad_l(prim->pad.begin(), prim->pad.end());
        dnnl::memory::dims pad_r(prim->pad.begin(), prim->pad.end());

        auto input_md = onednn::layout_to_memory_desc(input_layout);
        auto weights_md = onednn::layout_to_memory_desc(weights_layout, dnnl::memory::format_tag::any);
        auto output_md = onednn::layout_to_memory_desc(output_layout);
        auto grouped_weights = format::is_grouped(weights_layout.format) || prim->grouped_weights_shape;

        for (size_t i = 0; i < dilation.size(); i++) {
            dilation[i]--;
            int weights_offset = (grouped_weights ? 3 : 2) + static_cast<int>(i);
            auto os = output_md.dims()[2 + i];
            auto is = input_md.dims()[2 + i];
            auto ks = weights_md.dims()[weights_offset];
            auto kernel_range = 1 + (ks - 1) * (dilation[i] + 1);
            pad_r[i] = (is - 1) * stride[i] - os + kernel_range - pad_l[i];
        }

        if (!prim->bias.empty()) {
            auto bias_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(2), dnnl::memory::format_tag::any, true);
            return std::make_shared<dnnl::deconvolution_forward::desc>(
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::deconvolution_direct,
                input_md,
                weights_md,
                bias_md,
                output_md,
                stride,
                dilation,
                pad_l,
                pad_r);
        } else {
            return std::make_shared<dnnl::deconvolution_forward::desc>(
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::deconvolution_direct,
                input_md,
                weights_md,
                output_md,
                stride,
                dilation,
                pad_l,
                pad_r);
        }
    }

public:
    static primitive_impl* create(const deconvolution_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog.get_engine();
        auto desc = get_deconvolution_descriptor(impl_params);
        auto attr = get_primitive_attributes(arg);
        dnnl::primitive_desc prim_desc{&desc->data, attr.get(), engine.get_onednn_engine(), nullptr};

        return new deconvolution_onednn(engine, desc, attr, prim_desc, get_weights_reorder(impl_params, prim_desc));
    }
};

namespace detail {

attach_deconvolution_onednn::attach_deconvolution_onednn() {
    std::vector<data_types> dt = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
    };
    std::vector<format::type> fmt = {
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::b_fs_zyx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_yx_bsv4_fsv4,
        format::bs_fs_yx_bsv8_fsv4,
        format::bs_fs_yx_bsv8_fsv2,
        format::bs_fs_yx_bsv4_fsv2,
    };
    implementation_map<deconvolution>::add(impl_types::onednn, deconvolution_onednn::create, dt, fmt);
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn
