// Copyright (C) 2022-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace cldnn {
namespace onednn {

static std::shared_ptr<dnnl::convolution_forward::desc> get_convolution_descriptor(const kernel_impl_params& impl_params,
                                            dnnl::memory::format_tag tag_in_out = dnnl::memory::format_tag::undef) {
    auto prim = impl_params.typed_desc<convolution>();

    auto input_layout = impl_params.get_input_layout(0);
    auto weights_layout = impl_params.get_input_layout(1);
    auto output_layout = impl_params.output_layout;

    dnnl::memory::dims stride(prim->stride.begin(), prim->stride.end());
    dnnl::memory::dims dilation(prim->dilation.begin(), prim->dilation.end());
    dnnl::memory::dims pad_l(prim->pad.begin(), prim->pad.end());
    dnnl::memory::dims pad_r(prim->pad.begin(), prim->pad.end());

    auto input_md = onednn::layout_to_memory_desc(input_layout, tag_in_out);
    auto weights_md = onednn::layout_to_memory_desc(weights_layout, dnnl::memory::format_tag::any);
    auto output_md = onednn::layout_to_memory_desc(output_layout, tag_in_out);
    auto grouped_weights = format::is_grouped(weights_layout.format) || prim->grouped_weights_shape;

    // adjust_conv_dilation_pad(dilation, stride, pad_l, pad_r, input_md, output_md, weights_md, grouped_weights);
    for (size_t i = 0; i < dilation.size(); i++) {
        dilation[i]--;
        int weights_offset = (grouped_weights ? 3 : 2) + static_cast<int>(i);
        auto os = output_md.dims()[2 + i];
        auto is = input_md.dims()[2 + i];
        auto ks = weights_md.dims()[weights_offset];
        auto kernel_range = 1 + (ks - 1) * (dilation[i] + 1);
        pad_r[i] = (os - 1) * stride[i] - is + kernel_range - pad_l[i];
    }

    if (!prim->bias.empty()) {
        auto bias_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(2), dnnl::memory::format_tag::any, true);
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
} // namespace onednn
} // namespace cldnn
