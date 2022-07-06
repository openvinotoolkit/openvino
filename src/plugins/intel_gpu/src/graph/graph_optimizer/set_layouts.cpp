// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "data_inst.h"
#include "mutable_data_inst.h"
#include "program_node.h"
#include "intel_gpu/runtime/engine.hpp"
#include "runtime/cldnn_itt.hpp"
#include <iostream>
#include <oneapi/dnnl/dnnl.hpp>
#include "impls/onednn/utils.hpp"
#include "to_string_utils.h"

using namespace cldnn;


// It is a code duplication from convolution_onednn.cpp
static std::shared_ptr<dnnl::convolution_forward::desc> get_convolution_descriptor(const convolution_node& arg) {
    auto prim = arg.get_primitive();

    auto& input = arg.get_dependency(0);
    auto& weights = arg.get_dependency(1);

    dnnl::memory::dims stride(prim->stride.begin(), prim->stride.end());
    dnnl::memory::dims dilation(prim->dilation.begin(), prim->dilation.end());
    dnnl::memory::dims pad_l(prim->pad.begin(), prim->pad.end());
    dnnl::memory::dims pad_r(prim->pad.begin(), prim->pad.end());

    auto input_md = onednn::layout_to_memory_desc(input.get_output_layout(), dnnl::memory::format_tag::any);
    auto weights_md = onednn::layout_to_memory_desc(weights.get_output_layout(), dnnl::memory::format_tag::any);
    auto output_md = onednn::layout_to_memory_desc(arg.get_output_layout(), dnnl::memory::format_tag::any);
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

void set_layouts::run(program& p) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "CLDNN::pass::SetLayouts");

    // TODO: if !supports_immad, return

    for (auto n : p.get_processing_order()) {
        if (!n->is_type<convolution>())  // only care for convolutions
            continue;
        auto& node = n->as<convolution>();

        auto& engine = p.get_engine();
        auto desc = get_convolution_descriptor(node);
        // XXX: did not handle attribute properly. especially for zero-point
        dnnl::primitive_desc prim_desc{&desc->data, nullptr, engine.get_onednn_engine(), nullptr};
        auto src_fmt = onednn::convert_data_format(onednn::get_format_by_desc(prim_desc.src_desc()));
        auto dst_fmt = onednn::convert_data_format(onednn::get_format_by_desc(prim_desc.dst_desc()));
        std::cout << "Mingyuki: " << node.id() << ": " << fmt_to_str(src_fmt) << " --> " << fmt_to_str(dst_fmt) << std::endl;
        node.set_required_input0(src_fmt);
        node.set_required_output(dst_fmt);
    }
}
