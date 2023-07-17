// Copyright (C) 2022-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace cldnn {
namespace onednn {

static std::shared_ptr<dnnl::convolution_forward::primitive_desc> get_convolution_primitive_descriptor(const kernel_impl_params& impl_params,
                                            const dnnl::primitive_attr& attr = dnnl::primitive_attr(),
                                            dnnl::memory::format_tag tag_in_out = dnnl::memory::format_tag::undef) {
    auto& engine = impl_params.prog->get_engine();
    auto prim = impl_params.typed_desc<convolution>();

    auto input_layout = impl_params.get_input_layout(0);
    auto weights_layout = impl_params.get_input_layout(1);
    auto output_layout = impl_params.get_output_layout();

    dnnl::memory::dims stride(prim->stride.begin(), prim->stride.end());
    dnnl::memory::dims dilation(prim->dilation.begin(), prim->dilation.end());
    dnnl::memory::dims pad_l(prim->padding_begin.begin(), prim->padding_begin.end());
    dnnl::memory::dims pad_r(prim->padding_end.begin(), prim->padding_end.end());

    // issue: it could not find the implementation for 1d kernel GroupConvolution from onednn.
    // root-cause: 3d tensor of input/output is changed to 4d via ngraph.
    //             Creating conv description returns error if two inputs have same tensor of data input and weight.
    //     - original dims of IR
    //       input1: [  1, 280, 1200]      // [number of batches, number of channels, X]
    //       input2: [280,   1,    1, 67]  // [number of output channels, number of input channels, Y, X]
    //       output: [  1, 280, 1200]      // [number of batches, number of kernel output channels, X]
    //     - changed dims
    //       input1: [  1, 280, 1200,  1]
    //       input2: [280,   1,   67,  1]
    //       output: [  1, 280, 1200,  1]
    // WA: Weight tensor will be updated from 4d to 5d.
    auto grouped_weights = format::is_grouped(weights_layout.format) || prim->grouped_weights_shape;
    if (grouped_weights && (input_layout.get_rank() == weights_layout.get_rank())) {
        auto tensor = weights_layout.get_tensor();
        if (tensor.spatial[0] == 1 && tensor.spatial[1] != 1) {
            std::swap(tensor.spatial[0], tensor.spatial[1]);
            weights_layout.set_tensor(tensor);
        }
        weights_layout.format = format::get_default_format(weights_layout.get_rank() + 1, true, true);
    }

    auto input_md = onednn::layout_to_memory_desc(input_layout, tag_in_out);
    auto weights_md = onednn::layout_to_memory_desc(weights_layout, dnnl::memory::format_tag::any);
    auto output_md = onednn::layout_to_memory_desc(output_layout, tag_in_out);

    // adjust_conv_dilation_pad(dilation, stride, pad_l, pad_r, input_md, output_md, weights_md, grouped_weights);
    for (size_t i = 0; i < dilation.size(); i++) {
        dilation[i]--;
        int weights_offset = (grouped_weights ? 3 : 2) + static_cast<int>(i);
        auto os = output_md.get_dims()[2 + i];
        auto is = input_md.get_dims()[2 + i];
        auto ks = weights_md.get_dims()[weights_offset];
        auto kernel_range = 1 + (ks - 1) * (dilation[i] + 1);
        pad_r[i] = (os - 1) * stride[i] - is + kernel_range - pad_l[i];
    }

    // Extend conv parameters in case if spatials rank of output memory doesn't match size of parameters
    int64_t insert_count = static_cast<int64_t>(output_md.get_dims().size()) - 2 - stride.size();
    if (insert_count > 0) {
        stride.insert(stride.end(), insert_count, 1);
        dilation.insert(dilation.end(), insert_count, 0);
        pad_l.insert(pad_l.end(), insert_count, 0);
        pad_r.insert(pad_r.end(), insert_count, 0);
    }

    if (!prim->bias.empty()) {
        auto bias_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(2), dnnl::memory::format_tag::any, true);
        return std::make_shared<dnnl::convolution_forward::primitive_desc>(
            engine.get_onednn_engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct,
            input_md,
            weights_md,
            bias_md,
            output_md,
            stride,
            dilation,
            pad_l,
            pad_r,
            attr);
    } else {
        return std::make_shared<dnnl::convolution_forward::primitive_desc>(
            engine.get_onednn_engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct,
            input_md,
            weights_md,
            output_md,
            stride,
            dilation,
            pad_l,
            pad_r,
            attr);
    }
}
} // namespace onednn
} // namespace cldnn
