// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_snippets_tpp_emitters.hpp"

#include <cpu/x64/jit_generator.hpp>

#include "snippets/lowered/port_connector.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "transformations/snippets/tpp/op/brgemm.hpp"
#include "libxsmm.h"
#include "transformations/snippets/tpp/op/eltwise.hpp"

using namespace InferenceEngine;
using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

using jit_generator = dnnl::impl::cpu::x64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::x64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;
using ExpressionPort = snippets::lowered::ExpressionPort;

namespace {
size_t get_leading_dim(ExpressionPort port) {
        auto get_shape = [](ExpressionPort port) {
            bool has_buffer = false;
            std::vector<size_t> shape;
            for (const auto& p : port.get_connected_ports()) {
                std::cerr << p.get_expr()->get_node()->get_friendly_name() << "\n";
                if (const auto& buf = ov::as_type_ptr<snippets::op::Buffer>(p.get_expr()->get_node())) {
                    OPENVINO_ASSERT(!has_buffer, "Only one Buffer can be connected to a TPP op");
                    has_buffer = true;
                    shape = buf->get_allocation_shape();
                }
            }
            return has_buffer ? shape : port.get_descriptor_ptr()->get_shape();
        };
        const auto& shape = get_shape(port);
        const auto& layout = port.get_descriptor_ptr()->get_layout();
        OPENVINO_ASSERT(layout.back() == layout.size() - 1 && layout.size() == shape.size(),
                "BrgemmTppEmitter detected invalid layout values: check that this shape + layout combination is schedulable");
        const auto dim = [&]() -> size_t {
                switch (port.get_type()) {
                // Input shape is original, so we need to correctly read this data by order
                // Example:
                //      Original shape (shape) = [1, 49, 2, 23]
                //      Layout (transpose order) = [2, 0, 1, 3]
                //      Transposed shape = [2, 1, 49, 23]
                //      The leading dimension is equal to stride of shape[layout[3]] = 2 x 23
                case ExpressionPort::Type::Input :
                    return layout[layout.size() - 2]; // `1` in example
                // Output shape is already transposed, we need to correctly write the data with original shape by the order
                // Example:
                //      Original transposed shape (shape) = [49, 2, 7, 39]
                //      Layout (transpose order) = [2, 0, 1, 3]
                //      Before leading dimension with index 3 there is dimension with index 2 in planar layout.
                //      Since we have non-planar layout, we have to find this before LD dim in transposed order.
                //      In layout 2nd idx is first element, it means, that the leading dimension is equal to stride of shape[0]
                case ExpressionPort::Type::Output :
                    return std::distance(layout.cbegin(), std::find(layout.cbegin(), layout.cend(), layout.size() - 2)); // 0 in the example: shape[0] = 49
                default:
                    OPENVINO_THROW("Unsupported Expression port type");
            }
        }();
        return std::accumulate(shape.cbegin() + dim + 1, shape.cend(), 1, std::multiplies<size_t>());
}
} // namespace



size_t BrgemmTppEmitter::get_in_leading_dim(const VectorDims& shape, const std::vector<size_t>& layout) {
    // Input shape is original, so we need to correctly read this data by order
    // Example:
    //      Original shape (shape) = [1, 49, 2, 23]
    //      Layout (transpose order) = [2, 0, 1, 3]
    //      Transposed shape = [2, 1, 49, 23]
    //      The leading dimension is equal to stride of shape[layout[3]] = 2 x 23
    OPENVINO_ASSERT(layout.back() == layout.size() - 1 && layout.size() == shape.size(),
                    "BrgemmTppEmitter detected invalid layout values: check that this shape + layout combination is schedulable");
    const auto idx = layout[layout.size() - 2];  // `1` in example
    return std::accumulate(shape.cbegin() + idx + 1, shape.end(), 1, std::multiplies<size_t>());
}
size_t BrgemmTppEmitter::get_out_leading_dim(const VectorDims& shape, const std::vector<size_t>& layout) {
    // Output shape is already transposed, we need to correctly write the data with original shape by the order
    // Example:
    //      Original transposed shape (shape) = [49, 2, 7, 39]
    //      Layout (transpose order) = [2, 0, 1, 3]
    //      Before leading dimension with index 3 there is dimension with index 2 in planar layout.
    //      Since we have non-planar layout, we have to find this before LD dim in transposed order.
    //      In layout 2nd idx is first element, it means, that the leading dimension is equal to stride of shape[0]
    OPENVINO_ASSERT(layout.back() == layout.size() - 1 && layout.size() == shape.size(),
                    "BrgemmTppEmitter detected invalid layout values: check that this shape + layout combination is schedulable");
    const auto idx = layout.size() - 2; // 2 in the example
    const auto dim = std::distance(layout.cbegin(), std::find(layout.cbegin(), layout.cend(), idx)); // 0 in the example: shape[0] = 49
    return std::accumulate(shape.cbegin() + dim + 1, shape.cend(), 1, std::multiplies<size_t>()); // shape[1] x shape[2] x shape[3] = 2 x 7 x 39
}

BrgemmTppEmitter::BrgemmTppEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto& brgemm_node = as_type_ptr<intel_cpu::tpp::op::BrgemmTPP>(expr->get_node());
    OPENVINO_ASSERT(!brgemm_node->is_dynamic(), "Snippets don't support code generation for dynamic Brgemm");

    const auto& input_0_desc = expr->get_input_port_descriptor(0);
    const auto& input_1_desc = expr->get_input_port_descriptor(1);
    const auto& output_desc = expr->get_output_port_descriptor(0);

    std::vector<size_t> leading_dimensions {brgemm_node->get_input_stride(0),
                                            brgemm_node->get_input_stride(1),
                                            brgemm_node->get_output_stride(0)};


    auto brg0Prc = brgemm_node->get_input_element_type(0);
    auto brg1Prc = brgemm_node->get_input_element_type(1);
    bool brgWithAMX = brgemm_node->is_amx();

    io_data_size = {brg0Prc.size(), brg1Prc.size()};
    io_data_size.push_back(brgemm_node->get_output_element_type(0).size());

    const auto& output_subtensor = output_desc->get_subtensor();
    const auto& input_0_subtensor = input_0_desc->get_subtensor();
    const auto& input_1_subtensor = input_1_desc->get_subtensor();

    OPENVINO_ASSERT(*(output_subtensor.rbegin() + 1) == *(input_0_subtensor.rbegin() + 1),
                    "Brgemm has different M dimension subtensors on input0 and output");
    OPENVINO_ASSERT(*output_subtensor.rbegin() == *input_1_subtensor.rbegin(),
                    "Brgemm has different N dimension subtensors on input1 and output");
    OPENVINO_ASSERT(*input_0_subtensor.rbegin() == *(input_1_subtensor.rbegin() + 1),
                    "Brgemm has different K dimension subtensors on input0 and input1");

    m_brgCtx.M = *(output_subtensor.rbegin() + 1);
    m_brgCtx.N = *output_subtensor.rbegin();
    m_brgCtx.K = *input_0_subtensor.rbegin();
    m_brgCtx.LDA = leading_dimensions[0];
    m_brgCtx.LDB = leading_dimensions[1];
    m_brgCtx.LDC = leading_dimensions[2];
    m_brgCtx.dt_in0 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(brg0Prc));
    m_brgCtx.dt_in1 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(brg1Prc));
    m_brgCtx.beta = brgemm_node->get_beta();
    // Note: all special values of dimensions are assigned starting from SIZE_MAX (and downwards).
    // We need to make sure that M, N and K are ordinary dims, and not special values
    const auto max_meaningful_dim = SIZE_MAX - 100;
    OPENVINO_ASSERT(m_brgCtx.M < max_meaningful_dim && m_brgCtx.N < max_meaningful_dim && m_brgCtx.K < max_meaningful_dim,
                    "BrgemmTppEmitter: Invalid M, N or K dim detected");

    unsigned int is_f32_gemm = m_brgCtx.dt_in0 == m_brgCtx.dt_in1 && m_brgCtx.dt_in0 == dnnl_data_type_t::dnnl_f32 ? 1 : 0;
    unsigned int is_bf16_gemm =  m_brgCtx.dt_in0 == m_brgCtx.dt_in1 && m_brgCtx.dt_in0 == dnnl_data_type_t::dnnl_bf16 ? 1 : 0;
    unsigned int is_i8_gemm = ((m_brgCtx.dt_in0 == dnnl_data_type_t::dnnl_u8) || (m_brgCtx.dt_in0 == dnnl_data_type_t::dnnl_s8)) ? 1 : 0;
    unsigned int isKvnniDiv = is_f32_gemm > 0 ? 1 :
                              is_bf16_gemm > 0 ? (m_brgCtx.K % 2 == 0 ? 1 : 0) :
                              is_i8_gemm > 0 ? (m_brgCtx.K % 4 == 0 ? 1 : 0) :
                              0;
    OPENVINO_ASSERT(isKvnniDiv, "Unsupported parameter combination for BrgemmTpp kernel configuration");
    initBrgemmXsmm(m_brgCtx, m_brgKernelsXsmm, m_brgKernelsXsmmTileCfg, brgWithAMX);

    m_load_offset_a = brgemm_node->get_offset_a();
    m_load_offset_b = brgemm_node->get_offset_b();
    m_store_offset_c = brgemm_node->get_offset_c();
}

std::set<std::vector<element::Type>> BrgemmTppEmitter::get_supported_precisions(const std::shared_ptr<ngraph::Node>& node) {
    using BrgemmTPP = ov::intel_cpu::tpp::op::BrgemmTPP;
    const auto brgemm = as_type_ptr<BrgemmTPP>(node);
    OPENVINO_ASSERT(brgemm, "BrgemmTppEmitter::get_supported_precisions() expects BrgemmCPU node");
    switch (brgemm->get_type()) {
        case BrgemmTPP::Type::Floating:
            return {{element::f32, element::f32}};
        case BrgemmTPP::Type::WithDataRepacking:
            return {{element::u8, element::i8},
                    {element::bf16, element::bf16}};
        case BrgemmTPP::Type::WithCompensations:
            return {{element::i8, element::i8, element::f32}};
        case BrgemmTPP::Type::AMX:
            return {{element::i8, element::i8, element::u8},
                    {element::u8, element::i8, element::u8},
                    {element::bf16, element::bf16, element::u8}};
        default:
            OPENVINO_THROW("BrgemmTppEmitter got BrgemmCPU node with unsupported type");
    }
}

void BrgemmTppEmitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OPENVINO_ASSERT(in.size() == 2, "BrgemmTPPEmitter expects 2 input regs, got" + std::to_string(in.size()));
    OPENVINO_ASSERT(out.size() == 1, "BrgemmTPPEmitter expects 1 output reg, got" + std::to_string(out.size()));
}

libxsmm_datatype BrgemmTppEmitter::dnnl_to_xsmm_dtype(dnnl_data_type_t dnnl_dtype) {
    switch (dnnl_dtype) {
        case dnnl_data_type_t::dnnl_f32 : return LIBXSMM_DATATYPE_F32;
        case dnnl_data_type_t::dnnl_bf16 : return LIBXSMM_DATATYPE_BF16;
        // todo: is this always correct? Contact libxsmm devs
        case dnnl_data_type_t::dnnl_s8 : return LIBXSMM_DATATYPE_I8;
        case dnnl_data_type_t::dnnl_u8 : return LIBXSMM_DATATYPE_U8;
        default:
            // todo: not sure we need this implicit dtype, but check
            OPENVINO_THROW("Attempt to convert unsupported dnnl data type");
            return LIBXSMM_DATATYPE_IMPLICIT;
    }
}

void BrgemmTppEmitter::initBrgemmXsmm(brgemmCtx& ctx, libxsmm_gemmfunction& brgKernel, libxsmm_gemmfunction& brgKernelTileCfg, bool use_amx) {
  unsigned int is_f32_gemm = ((ctx.dt_in0 == dnnl_data_type_t::dnnl_f32) && (ctx.dt_in1 == dnnl_data_type_t::dnnl_f32)) ? 1 : 0;
  auto l_flags = (is_f32_gemm > 0) ? LIBXSMM_GEMM_FLAGS('N', 'N')
                                   : LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') |
                                     LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG |
                                     LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG;

  OPENVINO_ASSERT(ctx.beta == 0 || ctx.beta == 1, "BrgemmTppEmitter detected unsupported beta value: " + std::to_string(ctx.beta));
  if (ctx.beta == 0)
     l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
  auto l_flags_cfg = (is_f32_gemm > 0) ? LIBXSMM_GEMM_FLAGS('N', 'N') :
    LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG;
  auto dtype0 = dnnl_to_xsmm_dtype(ctx.dt_in1);
  auto dtype1 = dnnl_to_xsmm_dtype(ctx.dt_in0);
  auto comp_dtype = (dtype0 == LIBXSMM_DATATYPE_I8 || dtype0 == LIBXSMM_DATATYPE_U8) ? LIBXSMM_DATATYPE_I32 : LIBXSMM_DATATYPE_F32;
  auto out_dtype = (comp_dtype == LIBXSMM_DATATYPE_I32) ? LIBXSMM_DATATYPE_I32 : LIBXSMM_DATATYPE_F32;
  if (dtype0 == LIBXSMM_DATATYPE_U8) {
    dtype0 = LIBXSMM_DATATYPE_I8;
    l_flags |= LIBXSMM_GEMM_FLAG_A_UNSIGNED;
  }
  if (dtype1 == LIBXSMM_DATATYPE_U8) {
    dtype1 = LIBXSMM_DATATYPE_I8;
    l_flags |= LIBXSMM_GEMM_FLAG_B_UNSIGNED;
  }
  auto l_shape = libxsmm_create_gemm_shape(ctx.N, ctx.M, ctx.K, ctx.LDB, ctx.LDA, ctx.LDC, dtype0, dtype1, out_dtype, comp_dtype);
  auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  brgKernel = libxsmm_dispatch_gemm_v2(l_shape, l_flags, l_prefetch_flags);
  OPENVINO_ASSERT(brgKernel, "LIBXSMM BrgemmTppEmitter cannot create brgemm kernel due to invalid params");

  ctx.is_with_amx = use_amx;
  ctx.is_with_comp = ctx.dt_in0 == dnnl_data_type_t::dnnl_s8 && !ctx.is_with_amx;
  if (use_amx) {
    brgKernelTileCfg = libxsmm_dispatch_gemm_v2(l_shape, l_flags_cfg, l_prefetch_flags);
    OPENVINO_ASSERT(brgKernelTileCfg, "LIBXSMM BrgemmTppEmitter cannot create brgemm tile config kernel due to invalid params");
  }
}

void BrgemmTppEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    OPENVINO_ASSERT(host_isa_ == cpu::x64::avx512_core, "BrgemmTppEmitter requires at least avx512_core instruction set");
    Xbyak::Reg64 input_0(static_cast<int>(in[0]));
    Xbyak::Reg64 input_1(static_cast<int>(in[1]));
    Xbyak::Reg64 output_0(static_cast<int>(out[0]));
    emit_brgemm_kernel_call_libxsmm(input_0, input_1, output_0);
}

void BrgemmTppEmitter::emit_brgemm_kernel_call_libxsmm(Reg64 addr_A, Reg64 addr_B, Reg64 addr_C) const {
    if (m_brgCtx.is_with_amx) {
        Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->rax,
                                         h->rcx, h->rdx, h->rdi, h->rsi, h->rbp, h->rbx};
        size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);

        h->sub(h->rsp, n_gprs_to_save * gpr_size);
        for (size_t i = 0; i < n_gprs_to_save; ++i)
            h->mov(h->ptr[h->rsp + i * gpr_size], gprs_to_save[i]);

        // save function address in gpr to pass in call instruction
        h->mov(h->rbp, reinterpret_cast<uintptr_t>(libxsmm_amx_tile_configure));
        h->mov(abi_param1, reinterpret_cast<uintptr_t>(&m_brgKernelsXsmmTileCfg));
        // align stack on 16-byte as ABI requires
        // note that RBX must not be changed by the callee
        h->mov(h->rbx, h->rsp);
        h->and_(h->rbx, 0xf);
        h->sub(h->rsp, h->rbx);

        h->call(h->rbp);

        h->add(h->rsp, h->rbx);
        // restore gpr registers
        for (int i = n_gprs_to_save - 1; i >= 0; --i)
            h->mov(gprs_to_save[i], h->ptr[h->rsp + i * gpr_size]);
        h->add(h->rsp, n_gprs_to_save * gpr_size);
    }

    internal_call_preamble();

    h->mov(h->rbp, reinterpret_cast<uintptr_t>(kernel_execute_libxsmm));
    // todo: several of addr_{A, B, C} could be also abi_paramX, so one of them could be corrupted
    //  if moving directly h->uni_vmovq(abi_paramX, adr_X). Save them to vector regs to avoid corruption.
    //  It's likely that a more efficient solution exists.
    h->uni_vmovq(Xmm(0), addr_A);
    h->uni_vmovq(Xmm(1), addr_B);
    h->uni_vmovq(Xmm(2), addr_C);

    // todo: Windows ABI : requires different num of arguments passed in regs and on the stack. Need to align.
    const auto data_ptr_reg = [&](Xmm xmm, Xbyak::Reg64 reg, size_t bytes_offset) {
        h->uni_vmovq(reg, xmm);
        if (bytes_offset) h->add(reg, bytes_offset);
    };
    h->mov(abi_param1, reinterpret_cast<uintptr_t>(m_brgKernelsXsmm));
    data_ptr_reg(Xmm(0), abi_param2, m_load_offset_a);
    data_ptr_reg(Xmm(1), abi_param3, m_load_offset_b);
    data_ptr_reg(Xmm(2), abi_param4, m_store_offset_c);

#ifdef _WIN32
    // Before function call we should allocate stack area for
    //  - register parameters - ABI parameters (shadow space)
    //  - stack parameters - remaining parameters
    const size_t num_args_passed_on_stack = 3;  // count of function parameters
    size_t abi_param_count = sizeof(abi_param_regs) / sizeof(abi_param_regs[0]);
    h->sub(h->rsp, num_args_passed_on_stack * gpr_size);
#endif

    internal_call_rsp_align();
    h->call(h->rbp);
    internal_call_rsp_restore();

#ifdef _WIN32
    h->add(h->rsp, num_args_passed_on_stack * gpr_size);
#endif
    internal_call_postamble();
}

void BrgemmTppEmitter::kernel_execute_libxsmm(libxsmm_gemmfunction brg_kernel,
                                   void *A, void *B, void *C) {
    libxsmm_gemm_param gemm_p;
    gemm_p.a.primary = reinterpret_cast<void*>(B);
    gemm_p.b.primary = reinterpret_cast<void*>(A);
    gemm_p.c.primary = reinterpret_cast<void*>(C);
    assert(brg_kernel);
    brg_kernel(&gemm_p);
}

void BrgemmTppEmitter::libxsmm_amx_tile_configure(libxsmm_gemmfunction cfg_kernel) {
    libxsmm_gemm_param gemm_p;
    gemm_p.a.primary = reinterpret_cast<void*>(NULL);
    gemm_p.b.primary = reinterpret_cast<void*>(NULL);
    gemm_p.c.primary = reinterpret_cast<void*>(NULL);
    assert(cfg_kernel);
    cfg_kernel(&gemm_p);
}
libxsmm_blasint BinaryEltwiseTppEmitter::get_broadcasted_dim(libxsmm_blasint dim0, libxsmm_blasint dim1, std::pair<bool, bool>& bcast_flags) {
    if (dim0 == dim1) {
        bcast_flags = {false, false};
        return dim0;
    } else if (dim1 == 1) {
        bcast_flags = {false, true};
        return dim0;
    } else if (dim0 == 1) {
        bcast_flags = {true, false};
        return dim1;
    }
    OPENVINO_THROW("Invalid dimensions passed to get_broadcast_flags");
    return -1;
}

BinaryEltwiseTppEmitter::BinaryEltwiseTppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                                                 dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                 const ov::snippets::lowered::ExpressionPtr& expr)
                                                 : jit_emitter(h, isa) {
    using PortDescriptorPtr = snippets::lowered::PortDescriptorPtr;
    const auto& node = expr->get_node();
    const auto& tpp_node = std::dynamic_pointer_cast<tpp::op::BinaryEltwiseTPP>(node);
    OPENVINO_ASSERT(tpp_node, "BinaryEltwiseTppEmitter invoked with invalid node type");
    const auto& dtype_in0 = ov_to_xsmm_dtype(node->get_input_element_type(0));
    const auto& dtype_in1 = ov_to_xsmm_dtype(node->get_input_element_type(1));
    const auto& dtype_out0 = ov_to_xsmm_dtype(node->get_output_element_type(0));
    const auto& dtype_comp = ov_to_xsmm_dtype(ov::element::Type_t::f32);
    io_offsets[0] = tpp_node->get_input_offset(0);
    io_offsets[1] = tpp_node->get_input_offset(1);
    io_offsets[2] = tpp_node->get_output_offset(0);

    const auto ld_in0 = tpp_node->get_input_stride(0);
    const auto ld_in1 = tpp_node->get_input_stride(1);
    const auto ld_out0 = tpp_node->get_output_stride(0);

    const std::vector<PortDescriptorPtr> port_desc_input = {expr->get_input_port_descriptor(0), expr->get_input_port_descriptor(1)};

    auto get_projected_subtensor = [expr](const PortDescriptorPtr& desc){
        const auto& shape = desc->get_shape();
        auto subtensor = desc->get_subtensor();
        OPENVINO_ASSERT(subtensor.size() <= shape.size(), "Subtersor can't have more dimensins than a shape");
        auto shape_it = shape.rbegin();
        for (auto sub_it = subtensor.rbegin(); sub_it != subtensor.rend(); sub_it++, shape_it++) {
            if (*shape_it == 1)
                *sub_it = 1;
            OPENVINO_ASSERT(*sub_it <= *shape_it, "Subtensor element can't be larger than a shape element");
        }
        return subtensor;
    };

    const auto& subtensor_in0 = get_projected_subtensor(port_desc_input[0]);
    const auto& subtensor_in1 = get_projected_subtensor(port_desc_input[1]);

    const auto N_in0 = static_cast<libxsmm_blasint>(*subtensor_in0.rbegin());
    const auto M_in0 = static_cast<libxsmm_blasint>(*++subtensor_in0.rbegin());
    const auto N_in1 = static_cast<libxsmm_blasint>(*subtensor_in1.rbegin());
    const auto M_in1 = static_cast<libxsmm_blasint>(*++subtensor_in1.rbegin());

    // TODO: move LDA params to node fields and set them in a pass (validation pass?)
    auto get_lda = [&](std::set<snippets::lowered::ExpressionPort> connected_ports) {
        size_t LDA = 0;
        for (const auto& port : connected_ports) {
            if (const auto& buf = ov::as_type_ptr<snippets::op::Buffer>(port.get_expr()->get_node())) {
                OPENVINO_ASSERT(LDA == 0, "Only one Buffer can be connected to TPP Eltwise");
                LDA = buf->get_allocation_shape().back();
            } else if (ov::is_type<ov::op::v0::Parameter>(port.get_expr()->get_node()) ||
                       ov::is_type<ov::op::v0::Result>(port.get_expr()->get_node())) {
                const size_t new_LDA = port.get_descriptor_ptr()->get_shape().back();
                OPENVINO_ASSERT(LDA == 0 || LDA == new_LDA, "Incompatible leading dimensions detected");
                LDA = new_LDA;
            }
        }
        return LDA;
    };

    std::pair<bool, bool> n_bcast_flags, m_bcast_flags;
    const auto N = get_broadcasted_dim(N_in0, N_in1, n_bcast_flags);
    const auto M = get_broadcasted_dim(M_in0, M_in1, m_bcast_flags);

    if (m_bcast_flags.first && n_bcast_flags.first) {
        libxsmm_cfg.flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0;
    } else if (m_bcast_flags.first) {
        libxsmm_cfg.flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
    } else  if (n_bcast_flags.first) {
        libxsmm_cfg.flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0;
    }
    if (m_bcast_flags.second && n_bcast_flags.second) {
        libxsmm_cfg.flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1;
    } else if (m_bcast_flags.second) {
        libxsmm_cfg.flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1;
    } else  if (n_bcast_flags.second) {
        libxsmm_cfg.flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1;
    }
    libxsmm_cfg.op_type = tpp_node->get_op_type();
    // Note: libxsmm implies column-major layout, so we have to swap M and N here
    libxsmm_cfg.shape = libxsmm_create_meltw_binary_shape(N, M, ld_in0, ld_in1, ld_out0, dtype_in0, dtype_in1, dtype_out0, dtype_comp);
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void BinaryEltwiseTppEmitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

std::set<std::vector<element::Type>> BinaryEltwiseTppEmitter::get_supported_precisions(const std::shared_ptr<ngraph::Node>& node) {
    // todo: check what precisions are natively supported by tpp (without additional converts)
    return {{element::f32, element::f32}};
}

void BinaryEltwiseTppEmitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OPENVINO_ASSERT(in.size() == 2, "BinaryEltwiseTppEmitter expects 2 input registers, got " + std::to_string(in.size()));
    OPENVINO_ASSERT(out.size() == 1, "BinaryEltwiseTppEmitter expects 1 output register, got " + std::to_string(in.size()));
}

void BinaryEltwiseTppEmitter::execute_libxsmm_kernel(libxsmm_meltwfunction_binary eltwise_kernel,
                                                    void *in0, void *in1, void *out0) {
    // todo: how do we initialize the libxsmm_meltw_binary_param.op field?
    //  In general, how to use the libxsmm_matrix_arg type? What is the purpose of these primary/secondary/ternary fields?
    libxsmm_meltw_binary_param binary_param;
    binary_param.op.primary = nullptr;
    binary_param.in0.primary = in0;
    binary_param.in1.primary = in1;
    binary_param.out.primary = out0;
    assert(eltwise_kernel);
    eltwise_kernel(&binary_param);
}

void BinaryEltwiseTppEmitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    internal_call_preamble();
    auto in0_ptr = Reg64(static_cast<int>(in[0]));
    auto in1_ptr = Reg64(static_cast<int>(in[1]));
    auto out0_ptr = Reg64(static_cast<int>(out[0]));

    // save function address in gpr to pass in call instruction
    h->mov(h->rbp, reinterpret_cast<uintptr_t>(execute_libxsmm_kernel));
    // todo: several of in/out ptr could be also abi_paramX, so one of them could be corrupted
    //  if moving directly h->uni_vmovq(abi_paramX, adr_X). Save them to vector regs to avoid corruption.
    //  It's likely that a more efficient solution exists.
    h->uni_vmovq(Xmm(0), in0_ptr);
    h->uni_vmovq(Xmm(1), in1_ptr);
    h->uni_vmovq(Xmm(2), out0_ptr);

    // todo: Windows ABI : requires different num of arguments passed in regs and on the stack. Need to align.
    const auto data_ptr_reg = [&](Xmm xmm, Xbyak::Reg64 reg, size_t bytes_offset) {
        h->uni_vmovq(reg, xmm);
        if (bytes_offset) h->add(reg, bytes_offset);
    };
    // Note: libxsmm hides memory management from the user, so we don't have to store pointer to compiled kernel to keep it alive.
    // libxsmm will keep the pointer alive until the end of program execution (it doesn't matter whether we save the pointer in the emitter or not)
    auto libxsmm_kernel = libxsmm_dispatch_meltw_binary_v2(libxsmm_cfg.op_type, libxsmm_cfg.shape, libxsmm_cfg.flags);
    if (!libxsmm_kernel)
        std::cerr << "fail";
    OPENVINO_ASSERT(libxsmm_kernel, "Failed to dispatch libxsmm_kernel in BinaryEltwiseTppEmitter");
    h->mov(abi_param1, reinterpret_cast<uintptr_t>(libxsmm_kernel));
    data_ptr_reg(Xmm(0), abi_param2, io_offsets[0]);
    data_ptr_reg(Xmm(1), abi_param3, io_offsets[1]);
    data_ptr_reg(Xmm(2), abi_param4, io_offsets[2]);

    // todo: add shadow space handling for WIN32
    internal_call_rsp_align();
    h->call(h->rbp);
    internal_call_rsp_restore();

    internal_call_postamble();
}

libxsmm_datatype BinaryEltwiseTppEmitter::ov_to_xsmm_dtype(ov::element::Type_t elemet_type) {
    switch (elemet_type) {
        case ov::element::Type_t::f32 : return LIBXSMM_DATATYPE_F32;
        case ov::element::Type_t::bf16 : return LIBXSMM_DATATYPE_BF16;
        case ov::element::Type_t::i8 : return LIBXSMM_DATATYPE_I8;
        case ov::element::Type_t::u8 : return LIBXSMM_DATATYPE_U8;
        default:
            OPENVINO_THROW("Attempt to convert unsupported ov data type");
            return LIBXSMM_DATATYPE_IMPLICIT;
    }
}

}  // namespace intel_cpu
}  // namespace ov
