// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_conv_merged_dw_kernel.hpp"

#include <algorithm>

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "snippets/op/convolution_merged_dw_kernel.hpp"
//#include "snippets/op/loop.hpp"

using namespace Xbyak;

namespace ov {
namespace intel_cpu {

ConvolutionMergedDwKernelEmitter::ConvolutionMergedDwKernelEmitter(
        dnnl::impl::cpu::x64::jit_generator* h,
        dnnl::impl::cpu::x64::cpu_isa_t isa,
        const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    // TODO: backprop: do we need it???
    in_out_type_ = emitter_in_out_map::mixed;
    shouldPostIncrement = true;

    const auto &convolution = as_type_ptr<ngraph::snippets::op::ConvolutionMergedDwKernel>(n);
    assert(convolution != nullptr);

    //auto get_register_index = [](const std::shared_ptr<ngraph::Node> &node) {
    //    const auto &rt = node->get_rt_info();
    //    const auto it = rt.find("reginfo");
    //    if (it == rt.end()) {
    //        throw ov::Exception("reginfo is absent");
    //    }

    //    auto regs = it->second.as<std::vector<size_t>>();
    //    if (regs.size() != 1ul) {
    //        throw ov::Exception("registers count is not correct");
    //    }
    //    return regs[0];
    //};

    //{
    //    const auto loop = convolution->get_input_node_shared_ptr(0);
    //    if (!is_type<ngraph::snippets::op::Loop>(loop)) {
    //        throw ov::Exception("unexpected operation type on data");
    //    }
    //    const auto data = loop->get_input_node_shared_ptr(0);
    //    if (!is_type<ngraph::opset1::Parameter>(data)) {
    //        throw ov::Exception("unexpected operation type on data");
    //    }
    //    data_reg_index = get_register_index(data);
    //}

    const auto input_size = convolution->get_input_size();
    assert(input_size == 11ull);

    const auto weights = convolution->get_input_node_shared_ptr(input_size - 2ull);
    if (!ngraph::is_type<ngraph::opset1::Parameter>(weights)) {
        throw ov::Exception("unexpected operation type on weights");
    }

    const auto weights_shape = weights->get_shape();
    // TODO: uncomment
    //assert(weights_shape.size() == 7ull);
    filter_shape = { weights->get_shape()[3], weights->get_shape()[4] };

    //{
    //    const auto biases = convolution->get_input_node_shared_ptr(2);
    //    if (!is_type<ngraph::opset1::Parameter>(biases)) {
    //        throw ov::Exception("unexpected operation type on biases");
    //    }
    //    biases_reg_index = get_register_index(biases);
    //}

    pads_begin = convolution->get_pads_begin();
    pads_end = convolution->get_pads_end();
}

void ConvolutionMergedDwKernelEmitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out,
                            const std::vector<size_t>& pool,
                            const std::vector<size_t>& gpr,
                            const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_core) {
        emit_isa<dnnl::impl::cpu::x64::avx512_core>(in, out);
    } else {
        IE_THROW() << host_isa_;
        assert(!"unsupported isa");
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void ConvolutionMergedDwKernelEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    assert(in.size() >= 3ul);
    //assert(out.size() == 1ul);
    if (out.size() != 1ul) {
        std::cout << "ConvolutionMergedDwKernelEmitter::emit_isa: why we have more outputs?" << std::endl;
    }

    insert_marker(MARKER_CONVOLUTION_KERNEL);

    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
        Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;

    // TODO: hardcode: const auto h->allocate_register();
    const auto tmp = Vmm(9);
    const auto tmp2 = Vmm(10);
    const auto result = Vmm(static_cast<int>(out[0]));

    // TODO: biases are not mandatory
    const size_t data_size = in.size() - 2ull;
    const auto weight_gp = Reg64(in[in.size() - 2ull]);
    const auto biases_gp = Reg64(in[in.size() - 1ull]);

    // TODO: just to debug
    //h->uni_vmovups(tmp, h->ptr[weight_gp]);
    //h->uni_vmovups(tmp, h->ptr[biases_gp]);
    //h->uni_vmovups(tmp, h->ptr[weight_gp + 3 * 3 * 8ull * 4ull]);
    //h->uni_vmovups(tmp, h->ptr[biases_gp + 8 * 4]);

    std::vector<Vmm> data(data_size);
    for (auto i = 0ull; i < data_size; ++i) {
        data[i] = Vmm(in[i]);
    }

    auto result_as_input = false;
    for (auto i = 0ull; i < (in.size() - 2ull); ++i) {
        if (in[i] == out[0]) {
            result_as_input = true;
            break;
        }
    }
    if (result_as_input && (in[0] != out[0])) {
        throw ov::Exception("incorrect register assignment");
    }

    //if (!result_as_input) {
    //    h->uni_vmovups(result, h->ptr[biases_gp]);
    //} else {
    //    h->uni_vmovups(tmp, h->ptr[biases_gp]);
    //    h->uni_vaddps(result, result, tmp);
    //}

    assert(result_as_input);

    assert(filter_shape.size() == 2ull);
    const auto h_dim_max = filter_shape[0];
    const auto w_dim_max = filter_shape[1];

    // TODO: just to explore weights zero issue
    //const size_t h_offset = 8 * 3 * 3 * 4;
    //const size_t w_offset = 8 * 4;

    //h->uni_vmovups(tmp, h->ptr[weight_gp + h_offset * 0 + w_offset * 0]);
    //h->uni_vmovups(tmp, h->ptr[weight_gp + h_offset * 0 + w_offset * 1]);
    //h->uni_vmovups(tmp, h->ptr[weight_gp + h_offset * 0 + w_offset * 2]);
    //h->uni_vmovups(tmp, h->ptr[weight_gp + h_offset * 1 + w_offset * 0]);
    //h->uni_vmovups(tmp, h->ptr[weight_gp + h_offset * 1 + w_offset * 1]);
    //h->uni_vmovups(tmp, h->ptr[weight_gp + h_offset * 1 + w_offset * 2]);
    //h->uni_vmovups(tmp, h->ptr[weight_gp + h_offset * 2 + w_offset * 0]);
    //h->uni_vmovups(tmp, h->ptr[weight_gp + h_offset * 2 + w_offset * 1]);
    //h->uni_vmovups(tmp, h->ptr[weight_gp + h_offset * 2 + w_offset * 2]);

    //for (auto h_dim = 0ull; h_dim < h_dim_max; ++h_dim) {
    //    for (auto w_dim = 0ull; w_dim < w_dim_max; ++w_dim) {
    //        //h->uni_vmovups(tmp, h->ptr[weight_gp + (w_dim + h_dim * h_dim_max) * 8 * 8 * 4ull]);
    //        h->uni_vmovups(tmp, h->ptr[weight_gp + (w_dim + h_dim * h_dim_max) * 8ull * 4ull]);
    //        if ((w_dim == 0ull) && (h_dim == 0ull)) {
    //            h->uni_vmulps(result, tmp, data[w_dim + h_dim * 3ull]);
    //        } else {
    //            h->uni_vfmadd231ps(result, tmp, data[w_dim + h_dim * 3ull]);
    //        }
    //    }
    //}

    for (auto h_dim = 0ull; h_dim < h_dim_max; ++h_dim) {
        for (auto w_dim = 0ull; w_dim < w_dim_max; ++w_dim) {
            //h->uni_vmovups(tmp, h->ptr[weight_gp + (w_dim + h_dim * h_dim_max) * 8 * 8 * 4ull]);
            h->uni_vmovups(tmp, h->ptr[weight_gp + (w_dim + h_dim * h_dim_max) * 8ull * 4ull]);
            if ((w_dim == 0ull) && (h_dim == 0ull)) {
                // TODO: like as oneDNN
                h->uni_vmovups(tmp2, result);
                h->uni_vmovups(result, h->ptr[biases_gp]);
                h->uni_vfmadd231ps(result, tmp, tmp2);
            } else {
                h->uni_vfmadd231ps(result, tmp, data[w_dim + h_dim * 3ull]);
            }
        }
    }

    //if (result_as_input) {
    //    h->uni_vmovups(tmp, h->ptr[biases_gp]);
    //    h->uni_vaddps(result, result, tmp);
    //}

    h->add(biases_gp, dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen);
    h->add(weight_gp, dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen);

    insert_marker(MARKER_CONVOLUTION_KERNEL);
}

}   // namespace intel_cpu
}   // namespace ov
