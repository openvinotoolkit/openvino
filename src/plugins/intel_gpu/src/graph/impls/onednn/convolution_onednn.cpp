// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_inst.h"
#include "eltwise_inst.h"
#include "quantize_inst.h"
#include "primitive_onednn_base.h"
#include "impls/implementation_map.hpp"

#include "kernel_selector_common.h"

#include "utils.hpp"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

struct convolution_onednn : typed_primitive_onednn_impl<convolution, dnnl::convolution_forward::desc> {
    using parent = typed_primitive_onednn_impl<convolution, dnnl::convolution_forward::desc>;
    using parent::parent;

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<convolution_onednn>(*this);
    }

    bool validate_impl(const typed_primitive_inst<convolution>& instance) const override {
        bool res = true;

        auto outer_id = _outer.id();
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

    std::unordered_map<int, dnnl::memory> get_arguments(convolution_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args = parent::get_arguments(instance);
        auto attrs = instance.get_node().get_onednn_primitive_attributes();

        {
            auto weights = instance.weights_memory(0);
            args.insert({DNNL_ARG_WEIGHTS, weights->get_onednn_memory(_pd.weights_desc(0))});
        }

        if (instance.bias_term()) {
            auto bias = instance.bias_memory(0);
            args.insert({DNNL_ARG_BIAS, bias->get_onednn_memory(_pd.weights_desc(1))});
        }

        if (has_zero_points(DNNL_ARG_SRC, attrs)) {
            auto a_zp = instance.activations_zero_points_memory(0);
            dnnl::memory::desc desc = onednn::layout_to_memory_desc(a_zp->get_layout(), dnnl::memory::format_tag::a, true);
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, a_zp->get_onednn_memory(desc)});
        }

        if (has_zero_points(DNNL_ARG_WEIGHTS, attrs)) {
            auto w_zp = instance.weights_zero_points_memory(0);
            dnnl::memory::desc desc = onednn::layout_to_memory_desc(w_zp->get_layout(), dnnl::memory::format_tag::a, true);
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, w_zp->get_onednn_memory(desc)});
        }

        return args;
    }

    static std::shared_ptr<dnnl::primitive_attr> get_primitive_attributes(const typed_program_node<convolution>& arg) {
        auto attrs = arg.get_onednn_primitive_attributes();

        if (arg.activations_zero_points_term()) {
            auto& a_zp = arg.activations_zero_points();

            memory::ptr s32_mem;
            if (a_zp.get_output_layout().data_type == data_types::i8) {
                onednn::make_per_tensor_if_possible<data_type_to_type<data_types::i8>::type>(a_zp.as<data>());
                s32_mem = onednn::convert_zp_data_to_s32<data_type_to_type<data_types::i8>::type>(a_zp.as<data>().get_attached_memory_ptr());
            } else if (a_zp.get_output_layout().data_type == data_types::u8) {
                onednn::make_per_tensor_if_possible<data_type_to_type<data_types::u8>::type>(a_zp.as<data>());
                s32_mem = onednn::convert_zp_data_to_s32<data_type_to_type<data_types::u8>::type>(a_zp.as<data>().get_attached_memory_ptr());
            } else {
                throw std::runtime_error("Unsupported data type for activations zero points for oneDNN convolution");
            }
            a_zp.as<data>().attach_memory(s32_mem, false);

            int mask = a_zp.get_output_layout().count() > 1 ? 2 : 0;

            attrs->set_zero_points(DNNL_ARG_SRC, mask, {DNNL_RUNTIME_S32_VAL});
        }

        if (arg.weights_zero_points_term()) {
            throw std::runtime_error("Convolution oneDNN primitive doesn't support asymmetric weights quantization");

            // Commented out since oneDNN doesn't support asymmetric weights quantization
            // auto& w_zp = arg.weights_zero_points();
            // int mask = w_zp.get_output_layout().count() > 1 ? 2 : 0;
            // attrs->set_zero_points(DNNL_ARG_WEIGHTS, mask, {DNNL_RUNTIME_S32_VAL});
        }

        return attrs;
    }

    static kernel_selector::WeightsReorderParams get_weights_reorder(const convolution_node& arg, const dnnl::primitive_desc& pd) {
        kernel_selector::WeightsReorderParams weights_reorder_params;
        auto& reorderKS = kernel_selector::ReorderWeightsKernelSelctor::Instance();
        kernel_selector::reorder_weights_params r_params;

        auto cldnn_prim = arg.get_primitive();
        auto weights_layout = arg.get_dependency(1).get_output_layout();
        auto grouped_weights = format::is_grouped(weights_layout.format) || arg.get_primitive()->grouped_weights_shape;
        cldnn::format out_fmt = onednn::find_format(pd.weights_desc(0), grouped_weights);
        kernel_selector::WeightsLayout reqLayout = to_weights_layout(out_fmt, cldnn_prim->grouped_weights_shape);

        set_params(arg, r_params);
        r_params.layerID = arg.id() + "_reorder_";
        r_params.input = convert_weights_tensor(weights_layout, cldnn_prim->grouped_weights_shape);
        r_params.output = r_params.input.TransformIgnorePadding(reqLayout, r_params.input.GetDType(), arg.get_groups(), false);
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

    static std::shared_ptr<dnnl::convolution_forward::desc> get_convolution_descriptor(const convolution_node& arg) {
        auto prim = arg.get_primitive();

        auto& input = arg.get_dependency(0);
        auto& weights = arg.get_dependency(1);

        dnnl::memory::dims stride(prim->stride.begin(), prim->stride.end());
        dnnl::memory::dims dilation(prim->dilation.begin(), prim->dilation.end());
        dnnl::memory::dims pad_l(prim->pad.begin(), prim->pad.end());
        dnnl::memory::dims pad_r(prim->pad.begin(), prim->pad.end());

        auto input_md = onednn::layout_to_memory_desc(input.get_output_layout());
        auto weights_md = onednn::layout_to_memory_desc(weights.get_output_layout(), dnnl::memory::format_tag::any);
        auto output_md = onednn::layout_to_memory_desc(arg.get_output_layout());
        auto grouped_weights = format::is_grouped(weights.get_output_layout().format) || prim->grouped_weights_shape;

        for (size_t i = 0; i < dilation.size(); i++) {
            dilation[i]--;
            int weights_offset = (grouped_weights ? 3 : 2) + static_cast<int>(i);
            auto os = output_md.dims()[2 + i];
            auto is = input_md.dims()[2 + i];
            auto ks = weights_md.dims()[weights_offset];
            auto kernel_range = 1 + (ks - 1) * (dilation[i] + 1);
            pad_r[i] = (os - 1) * stride[i] - is + kernel_range - pad_l[i];
        }

        if (arg.bias_term()) {
            auto bias_md = onednn::layout_to_memory_desc(arg.get_dependency(2).get_output_layout(), dnnl::memory::format_tag::any, true);
            return std::make_shared<dnnl::convolution_forward::desc>(
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::convolution_direct,
                input_md,
                weights_md,
                bias_md,
                output_md,
                stride,
                dilation,
                pad_l,
                pad_r);
        } else {
            return std::make_shared<dnnl::convolution_forward::desc>(
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::convolution_direct,
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
    static primitive_impl* create(const convolution_node& arg) {
        auto& engine = arg.get_program().get_engine();
        auto desc = get_convolution_descriptor(arg);
        auto attr = get_primitive_attributes(arg);
        dnnl::primitive_desc prim_desc{&desc->data, attr.get(), engine.get_onednn_engine(), nullptr};

        return new convolution_onednn(arg, desc, attr, prim_desc, get_weights_reorder(arg, prim_desc));
    }
};

namespace detail {

attach_convolution_onednn::attach_convolution_onednn() {
    implementation_map<convolution>::add(impl_types::onednn, convolution_onednn::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),

        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),

        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::u8, format::byxf),
        std::make_tuple(data_types::i8, format::byxf),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv2),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv2),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv2),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv2),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv2),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv2),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv2),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv2),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv4),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv4),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv4),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv4),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv4),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv32),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv16_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv32_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv32_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv32_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv32_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv32_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv32_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv32_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv4_fsv4),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv8_fsv4),

        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv8_fsv4),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv8_fsv4),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv8_fsv4),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv8_fsv4),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv8_fsv2),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv8_fsv2),

        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv8_fsv2),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv8_fsv2),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv8_fsv2),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv8_fsv2),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv4_fsv2),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv4_fsv2),
    });
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn
