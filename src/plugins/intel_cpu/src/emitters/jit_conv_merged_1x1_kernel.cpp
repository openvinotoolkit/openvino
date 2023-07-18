// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_conv_merged_1x1_kernel.hpp"

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/opsets/opset1.hpp>
//#include "snippets/op/loop.hpp"

using namespace Xbyak;

namespace ov {
namespace intel_cpu {

ConvolutionMerged1x1KernelEmitter::ConvolutionMerged1x1KernelEmitter(
        dnnl::impl::cpu::x64::jit_generator* h,
        dnnl::impl::cpu::x64::cpu_isa_t isa,
        const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    // TODO: backprop: do we need it???
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
    shouldPostIncrement = true;

    // TODO: temporary limitation
    auto shape = n->input(0).get_shape();
    if (shape.size() != 5ull) {
        throw ov::Exception("unexpected shape size");
    }

    data_spatial_shape = {shape[2], shape[3]};

    //const auto &convolution = as_type_ptr<ngraph::snippets::op::Convolution1x1Kernel>(n);
    //assert(convolution != nullptr);

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

    //{
    //    const auto weights = convolution->get_input_node_shared_ptr(1);
    //    if (!is_type<ngraph::opset1::Parameter>(weights)) {
    //        throw ov::Exception("unexpected operation type on weights");
    //    }
    //    weights_reg_index = get_register_index(weights);
    //    weights_shape = weights->get_shape();
    //}

    //{
    //    const auto biases = convolution->get_input_node_shared_ptr(2);
    //    if (!is_type<ngraph::opset1::Parameter>(biases)) {
    //        throw ov::Exception("unexpected operation type on biases");
    //    }
    //    biases_reg_index = get_register_index(biases);
    //}
}

void ConvolutionMerged1x1KernelEmitter::emit_impl(const std::vector<size_t>& in,
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

namespace {
size_t get_value_offset(
        const size_t val_index,
        const size_t ch_index,
        const size_t h_dim,
        const size_t w_dim,
        const size_t filters_count,
        const size_t vlen) {
    return (val_index * 8 + (ch_index % 8) + (ch_index / 8) * h_dim * w_dim * 8) * 4;
}
} // namespace

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void ConvolutionMerged1x1KernelEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    assert(in.size() == 3ul);
    if (out.size() != 1ul) {
        std::cout << "ConvolutionKernelEmitter::emit_isa: why we have more outputs?" << std::endl;
    }

    insert_marker(MARKER_CONVOLUTION_KERNEL);

    int data_reg_index = in[0];
    int weights_reg_index = in[1];
    int biases_reg_index = in[2];

    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;

    const auto data_gp = Reg64(data_reg_index);
    const auto weight_gp = Reg64(weights_reg_index);
    const auto biases_gp = Reg64(biases_reg_index);

    // TODO: workaround: hardcoded values
    auto data = Vmm(15);
    auto weights = Vmm(12);

    std::vector<Vmm> accums(out.size());
    h->uni_vmovups(accums[0], h->ptr[biases_gp]);
    for (auto i = 1ull; i < out.size(); ++i) {
        accums[i] = Vmm(out[i]);
        h->uni_vmovups(accums[i], accums[0]);
    }

    const auto h_dim_max = data_spatial_shape[0];
    const auto w_dim_max = data_spatial_shape[1];

    for (auto in_ch = 0ull; in_ch < 16ull; ++in_ch) {
        h->uni_vmovups(weights, h->ptr[weight_gp + in_ch * 32ull]);
        for (auto h_dim = 0ull; h_dim < 3ull; ++h_dim) {
            for (auto w_dim = 0ull; w_dim < 3ull; ++w_dim) {
                const auto value_offset = get_value_offset(
                        w_dim + w_dim_max * h_dim,
                        in_ch,
                        h_dim_max,
                        w_dim_max,
                        16ull,
                        dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen);
                h->uni_vbroadcastss(data, h->ptr[data_gp + value_offset]);
                h->uni_vfmadd231ps(accums[h_dim * 3ull + w_dim], weights, data);
            }
        }
    }

    h->add(data_gp, dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen);
    h->add(biases_gp, dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen);
    h->add(weight_gp, dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen);

    insert_marker(MARKER_CONVOLUTION_KERNEL);
}

}   // namespace intel_cpu
}   // namespace ov
