// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_snippets_tpp_emitters.hpp"

#include <cpu/x64/jit_generator.hpp>

#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/port_connector.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
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

size_t BrgemmTppEmitter::getBrgIdx(size_t kIdx, size_t nIdx) {
        return kIdx * BRGEMM_N_KERNEL_NUM + nIdx;
}
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
    m_brgCtxs.fill(brgemmCtx());
    for (size_t i = 0; i < m_brgKernelsXsmm.size(); i++) {
      m_brgKernelsXsmm[i].gemm = nullptr;
      m_brgKernelsXsmmTileCfg[i].gemm = nullptr;
    }
    //todo: remove this debug print
    std::cerr << "BrgemmTppEmitter invoked\n" << std::flush;
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto& brgemm_node = as_type_ptr<intel_cpu::tpp::op::BrgemmTPP>(expr->get_node());
    OPENVINO_ASSERT(!brgemm_node->is_dynamic(), "Snippets don't support code generation for dynamic Brgemm");
//    const auto brgemm_copy = brgemm_node->is_with_data_repacking() ? brgemm_node->get_brgemm_copy() : nullptr;

    std::vector<size_t> leading_dimensions;
    std::vector<std::vector<size_t>> io_layouts;

     auto get_layout = [](const std::vector<size_t>& layout, const snippets::VectorDims& io_shape) {
        if (!layout.empty()) return layout;
        std::vector<size_t> default_layout(io_shape.size());
        std::iota(default_layout.begin(), default_layout.end(), 0);
        return default_layout;
    };

    auto init_in_scheduling_params = [&](const snippets::lowered::PortDescriptorPtr& input) {
        io_layouts.push_back(get_layout(input->get_layout(), input->get_shape()));
        leading_dimensions.push_back(get_in_leading_dim(input->get_shape(), io_layouts.back()));
    };
    auto init_out_scheduling_params = [&](const snippets::lowered::PortDescriptorPtr& output) {
        io_layouts.push_back(get_layout(output->get_layout(), output->get_shape()));
        leading_dimensions.push_back(get_out_leading_dim(output->get_shape(), io_layouts.back()));
    };
    init_in_scheduling_params(expr->get_input_port_descriptor(0));
//    if (brgemm_node->is_with_data_repacking()) {
//        io_layouts.push_back(std::vector<size_t>{});
//        leading_dimensions.push_back(0);
//    } else {
        init_in_scheduling_params(expr->get_input_port_descriptor(1));
//    }
    init_out_scheduling_params(expr->get_output_port_descriptor(0));

    const auto& A_shape = expr->get_input_port_descriptor(0)->get_shape();
    const auto& A_layout = io_layouts[0];
    const auto& C_shape = expr->get_output_port_descriptor(0)->get_shape();
    const auto& C_layout = io_layouts[2];

    // We need find original M,N,K having layouts and ordered shapes
    // Layout:  0, 1, 2, 3   =>   New layout: 0, 2, 1, 3
    // Shape:   1, 3, 5, 9   =>   New Shape:  1, 5, 3, 9
    // To find original 2nd dimension, we should find index of position value `2` in new layout
    // and get dimension from new shape by this index
    auto get_ordered_idx = [](const std::vector<size_t>& layout, size_t idx) {
        return std::distance(layout.begin(), std::find(layout.begin(), layout.end(), idx));
    };

    m_K = A_shape[get_ordered_idx(A_layout, A_layout.size() - 1)];
    m_M = brgemm_node->get_input_count(0);
    m_N = C_shape[get_ordered_idx(C_layout, C_layout.size() - 1)];

//    if (brgemm_node->is_with_data_repacking())
//        leading_dimensions[1] = rnd_up(m_N, brgemm_copy->get_n_block_size());

    auto brg0Prc = InferenceEngine::details::convertPrecision(brgemm_node->get_input_element_type(0));
    auto brg1Prc = InferenceEngine::details::convertPrecision(brgemm_node->get_input_element_type(1));
    bool brgWithAMX = brgemm_node->is_amx();

    io_data_size = {brg0Prc.size(), brg1Prc.size()};
//    if (brgemm_node->get_input_size() == 3)
//        io_data_size.push_back(brgemm_node->get_input_element_type(2).size());
    io_data_size.push_back(brgemm_node->get_output_element_type(0).size());

//    m_with_comp = brgemm_node->is_with_compensations();
//    m_with_scratch = brgemm_node->is_with_scratchpad();
    m_with_scratch = false;
    m_with_comp = false;

    m_N_blk = brgemm_node->get_n_block_size();
    m_K_blk = brgemm_node->get_k_block_size();
    m_N_tail = m_N % m_N_blk;
    m_K_tail = m_K % m_K_blk;

    m_N_blk_loop = m_N >= 2 * m_N_blk;
    m_K_blk_loop = m_K >= 3 * m_K_blk;
//    OPENVINO_ASSERT((!brgemm_node->is_with_data_repacking()) || (!m_N_blk_loop && !m_K_blk_loop),
//                    "BrgemmTppEmitter doesn't support blocking by K, N dimensions when data repacking is needed!");

    auto N = [&](size_t n) {
        switch (n) {
            case 0: return m_N_blk;
            case 1: return m_N_tail;
            default: OPENVINO_THROW("BrgemmTppEmitter detected unsupported N value");
        }
    };
    auto K = [&](size_t k) {
        switch (k) {
            case 0: return m_K_blk;
            case 1: return m_K >= 2 * m_K_blk ? m_K_blk : 0;
            case 2: return m_K_tail;
            default:  OPENVINO_THROW("BrgemmTppEmitter detected unsupported K value");
        }
    };

    bool has_K_kernel = false;
    for (size_t k = 0; k < BRGEMM_K_KERNEL_NUM; k++) {
        bool has_N_kernel = false;
        for (size_t n = 0; n < BRGEMM_N_KERNEL_NUM; n++) {
            const size_t kernel_idx = getBrgIdx(k, n);
            auto& brgemmCtx = m_brgCtxs[kernel_idx];

            brgemmCtx.M = m_M;
            brgemmCtx.N = N(n);
            brgemmCtx.K = K(k);
            brgemmCtx.LDA = leading_dimensions[0];
            brgemmCtx.LDB = leading_dimensions[1];
            brgemmCtx.LDC = leading_dimensions[2];
            brgemmCtx.dt_in0 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::IEPrecisionToDataType(brg0Prc));
            brgemmCtx.dt_in1 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::IEPrecisionToDataType(brg1Prc));
            brgemmCtx.beta = has_K_kernel ? 1 : 0;

            if (brgemmCtx.N == 0 || brgemmCtx.N > m_N ||
                brgemmCtx.K == 0 || brgemmCtx.K > m_K)
                continue;

            unsigned int is_f32_gemm = ((brgemmCtx.dt_in0 == dnnl_data_type_t::dnnl_f32) && (brgemmCtx.dt_in1 == dnnl_data_type_t::dnnl_f32)) ? 1 : 0;
            unsigned int is_bf16_gemm = ((brgemmCtx.dt_in0 == dnnl_data_type_t::dnnl_bf16) && (brgemmCtx.dt_in1 == dnnl_data_type_t::dnnl_bf16)) ? 1 : 0;
            unsigned int is_i8_gemm = ((brgemmCtx.dt_in0 == dnnl_data_type_t::dnnl_u8) || (brgemmCtx.dt_in0 == dnnl_data_type_t::dnnl_s8)) ? 1 : 0;
            unsigned int isKvnniDiv = (is_f32_gemm > 0) ? 1
              : ((is_bf16_gemm > 0) ? ((brgemmCtx.K % 2 == 0) ? 1 : 0) : ((is_i8_gemm > 0) ? ((brgemmCtx.K % 4 == 0) ? 1 : 0) : 0));
            OPENVINO_ASSERT((m_with_comp == 0) && (isKvnniDiv > 0), "Unsupported parameter combination for BrgemmTpp kernel configuration");
            initBrgemmXsmm(brgemmCtx, &m_brgKernelsXsmm[kernel_idx], &m_brgKernelsXsmmTileCfg[kernel_idx], brgWithAMX);
            has_N_kernel = true;
        }
        if (has_N_kernel)
            has_K_kernel = true;
    }

    m_load_offset_a = brgemm_node->get_offset_a();
    m_load_offset_b = brgemm_node->get_offset_b();
    m_store_offset_c = brgemm_node->get_offset_c();
    if (m_with_scratch)
        m_load_offset_scratch = brgemm_node->get_offset_scratch();
}

std::set<std::vector<element::Type>> BrgemmTppEmitter::get_supported_precisions(const std::shared_ptr<ngraph::Node>& node) {
    const auto brgemm = as_type_ptr<ov::intel_cpu::BrgemmCPU>(node);
    OPENVINO_ASSERT(brgemm, "BrgemmTppEmitter::get_supported_precisions() expects BrgemmCPU node");
    switch (brgemm->get_type()) {
        case BrgemmCPU::Type::Floating:
            return {{element::f32, element::f32}};
        case BrgemmCPU::Type::WithDataRepacking:
            return {{element::u8, element::i8},
                    {element::bf16, element::bf16}};
        case BrgemmCPU::Type::WithCompensations:
            return {{element::i8, element::i8, element::f32}};
        case BrgemmCPU::Type::AMX:
            return {{element::i8, element::i8, element::u8},
                    {element::u8, element::i8, element::u8},
                    {element::bf16, element::bf16, element::u8}};
        default:
            OPENVINO_THROW("BrgemmTppEmitter got BrgemmCPU node with unsupported type");
    }
}

void BrgemmTppEmitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    std::set<size_t> unique_ids{in[0], in[1], out[0]};
    size_t unique_ids_count = 3;
    auto add_reg_to_unique_ids = [&](const size_t reg_number) {
        unique_ids.insert(reg_number);
        unique_ids_count++;
    };

    if (m_N_blk_loop || m_K_blk_loop) {
        if (aux_gpr_idxs.size() < static_cast<size_t>(m_N_blk_loop) + static_cast<size_t>(m_K_blk_loop))
            IE_THROW() << "BRGEMM Emitter requires extra gpr which was not allocated";
        if (m_N_blk_loop)
            add_reg_to_unique_ids(aux_gpr_idxs[0]);
        if (m_K_blk_loop)
            add_reg_to_unique_ids(aux_gpr_idxs[m_N_blk_loop]);
    }
    if (m_with_scratch) {
        if (in.size() != 3)
            IE_THROW() << "BRGEMM Emitter expects 3 inputs if there are compensations/wsp";
        add_reg_to_unique_ids(in[2]);
    }
    if (unique_ids.size() != unique_ids_count) {
        IE_THROW() << "BRGEMM Emitter expects that all input/output registers are unique";
    }
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

void BrgemmTppEmitter::initBrgemmXsmm(brgemmCtx& ctx, libxsmm_xmmfunction *brgKernel, libxsmm_xmmfunction *brgKernelTileCfg, bool use_amx) {
  unsigned int is_f32_gemm = ((ctx.dt_in0 == dnnl_data_type_t::dnnl_f32) && (ctx.dt_in1 == dnnl_data_type_t::dnnl_f32)) ? 1 : 0;
  auto l_flags_init = (is_f32_gemm > 0) ? LIBXSMM_GEMM_FLAGS('N', 'N')
                                        : LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') |
                                          LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG |
                                          LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG;
  auto l_flags = (ctx.beta == 0.0f) ? l_flags_init | LIBXSMM_GEMM_FLAG_BETA_0 : l_flags_init;
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
  brgKernel->gemm = libxsmm_dispatch_gemm_v2(l_shape, l_flags, l_prefetch_flags);
  OPENVINO_ASSERT(brgKernel->gemm, "LIBXSMM BrgemmTppEmitter cannot create brgemm kernel due to invalid params");

  ctx.is_with_amx = use_amx;
  ctx.is_with_comp = ctx.dt_in0 == dnnl_data_type_t::dnnl_s8 && !ctx.is_with_amx;
  if (use_amx) {
    brgKernelTileCfg->gemm = libxsmm_dispatch_gemm_v2(l_shape, l_flags_cfg, l_prefetch_flags);
    OPENVINO_ASSERT(brgKernelTileCfg->gemm, "LIBXSMM BrgemmTppEmitter cannot create brgemm tile config kernel due to invalid params");
  }
}

size_t BrgemmTppEmitter::aux_gprs_count() const {
    return m_N_blk_loop + m_K_blk_loop;
}

void BrgemmTppEmitter::emit_N_blocking_loops(size_t k_kernel_id,
                                          const Xbyak::Reg64& input_0, const Xbyak::Reg64& input_1,
                                          const Xbyak::Reg64& input_2, const Xbyak::Reg64& output_0,
                                          const Xbyak::Reg64& work_amount_N) const {
    // Blocked N loop
    size_t kernel_idx = getBrgIdx(k_kernel_id, 0);
    if (m_brgKernelsXsmm[kernel_idx].gemm) {
        const auto& brgemmCtx = m_brgCtxs[kernel_idx];
        Label N_loop_begin;
        if (m_N_blk_loop) {
            h->mov(work_amount_N, m_N);
            h->L(N_loop_begin);
        }
        OPENVINO_ASSERT(m_with_comp == 0 && m_brgKernelsXsmm[kernel_idx].gemm, "Invalid configuration for Tpp emitter detected");
        if (m_with_comp == 0 && m_brgKernelsXsmm[kernel_idx].gemm) {
          emit_brgemm_kernel_call_libxsmm(&m_brgKernelsXsmm[kernel_idx], &m_brgKernelsXsmmTileCfg[kernel_idx], brgemmCtx, input_0, input_1, input_2, output_0);
        }
        // We don't need to increment pointers if we cover full N dimension in one kernel call
        if (m_N_blk_loop || m_N_tail != 0) {
            h->add(output_0, brgemmCtx.N * io_data_size.back());
            h->add(input_1, brgemmCtx.N * io_data_size[1]);
            if (m_with_scratch && m_with_comp)
                h->add(input_2, brgemmCtx.N * io_data_size[2]);
        }

        if (m_N_blk_loop) {
            h->sub(work_amount_N, brgemmCtx.N);
            h->cmp(work_amount_N, brgemmCtx.N);
            h->jge(N_loop_begin);
        }
    }
    // N loop tail
    kernel_idx = getBrgIdx(k_kernel_id, 1);
    const auto& brgemmCtxTail = m_brgCtxs[kernel_idx];
    if (m_brgKernelsXsmm[kernel_idx].gemm) {
      emit_brgemm_kernel_call_libxsmm(&m_brgKernelsXsmm[kernel_idx], &m_brgKernelsXsmmTileCfg[kernel_idx], brgemmCtxTail, input_0, input_1, input_2, output_0);
    }

    if (m_N_blk_loop || m_N_tail != 0) {
        h->sub(input_1, (m_N - m_N_tail) * io_data_size[1]);
        h->sub(output_0, (m_N - m_N_tail) * io_data_size.back());
        if (m_with_scratch && m_with_comp)
            h->sub(input_2, (m_N - m_N_tail) * io_data_size[2]);
    }
}

void BrgemmTppEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    if (host_isa_ == cpu::x64::avx512_core) {
        Xbyak::Reg64 input_0(static_cast<int>(in[0]));
        Xbyak::Reg64 input_1(static_cast<int>(in[1]));
        Xbyak::Reg64 input_2(static_cast<int>(0));  // scratch. Default reg index is 0 if there isn't scratch
        Xbyak::Reg64 output_0(static_cast<int>(out[0]));
        Xbyak::Reg64 work_amount_N(m_N_blk_loop ? static_cast<int>(aux_gpr_idxs[0]) : 0);
        Xbyak::Reg64 work_amount_K(m_K_blk_loop ? static_cast<int>(aux_gpr_idxs[m_N_blk_loop]) : 0);
        h->add(input_0, m_load_offset_a);
        h->add(input_1, m_load_offset_b);
        h->add(output_0, m_store_offset_c);
        if (m_with_scratch) {
            input_2 = Xbyak::Reg64(static_cast<int>(in[2]));
            h->add(input_2, m_load_offset_scratch);
        }

        // fills kernel_idx with the first idx of non-empty K kernel or returns false
        auto get_K_kernel_idx = [&](size_t k_kernel_id, size_t& kernel_idx) {
            for (size_t n = 0; n < BRGEMM_N_KERNEL_NUM; n++) {
                const auto idx = getBrgIdx(k_kernel_id, n);
                if (m_brgKernelsXsmm[idx].gemm) {
                    kernel_idx = idx;
                    return true;
                }
            }
            return false;
        };
        // Blocked K loop
        const auto k_tail_id = BRGEMM_K_KERNEL_NUM - 1;
        size_t total_K_work_amount = m_K;
        size_t kernel_idx = SIZE_MAX;
        for (size_t k_blocked_id = 0; k_blocked_id < k_tail_id; k_blocked_id++) {
            if (get_K_kernel_idx(k_blocked_id, kernel_idx)) {
                const auto& brgemmCtx = m_brgCtxs[kernel_idx];
                Label K_loop_begin;
                // Note: we never emit loop for the first blocked kernel, since it always executed only once.
                // The purpose of the first blocked K kernel is to initializes output, because it has beta = 0
                if (k_blocked_id == 0) {
                    total_K_work_amount -= brgemmCtx.K;
                } else if (m_K_blk_loop) {
                    h->mov(work_amount_K, total_K_work_amount);
                    h->L(K_loop_begin);
                }

                emit_N_blocking_loops(k_blocked_id, input_0, input_1, input_2, output_0, work_amount_N);
                h->add(input_0, brgemmCtx.K * io_data_size[0]);
                h->add(input_1, (brgemmCtx.K * brgemmCtx.LDB) * io_data_size[1]);
                if (m_K_blk_loop && k_blocked_id) {
                    h->sub(work_amount_K, brgemmCtx.K);
                    h->cmp(work_amount_K, brgemmCtx.K);
                    h->jge(K_loop_begin);
                }
            }
        }
        // K loop tail
        if (get_K_kernel_idx(k_tail_id, kernel_idx)) {
            emit_N_blocking_loops(k_tail_id, input_0, input_1, input_2, output_0, work_amount_N);
        }

        h->sub(input_0, m_load_offset_a + (m_K - m_K_tail) * io_data_size[0]);
        h->sub(input_1, m_load_offset_b + (m_K - m_K_tail) * m_brgCtxs[0].LDB * io_data_size[1]);
        if (m_with_scratch)
            h->sub(input_2, m_load_offset_scratch);
        h->sub(output_0, m_store_offset_c);
    } else {
        IE_THROW() << "BrgemmTppEmitter requires at least avx512_core instruction set";
    }
}

void BrgemmTppEmitter::emit_brgemm_kernel_call_libxsmm(const libxsmm_xmmfunction *xsmm_func, const libxsmm_xmmfunction *xsmm_tile_cfg, const brgemmCtx& ctx,
                                            Reg64 addr_A, Reg64 addr_B, Reg64 scratch, Reg64 addr_C,
                                            const size_t in0_kernel_offset, const size_t in1_kernel_offset,
                                            const size_t in2_kernel_offset, const size_t out0_kernel_offset) const {
    if (ctx.is_with_amx) {
        Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->rax,
                                         h->rcx, h->rdx, h->rdi, h->rsi, h->rbp, h->rbx};
        size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);

        h->sub(h->rsp, n_gprs_to_save * gpr_size);
        for (size_t i = 0; i < n_gprs_to_save; ++i)
            h->mov(h->ptr[h->rsp + i * gpr_size], gprs_to_save[i]);

        // save function address in gpr to pass in call instruction
        const auto& overload = static_cast<void (*)(libxsmm_xmmfunction*)>(libxsmm_amx_tile_configure);
        h->mov(h->rbp, reinterpret_cast<uintptr_t>(overload));
        h->mov(abi_param1, reinterpret_cast<uintptr_t>(xsmm_tile_cfg));
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

    // save function address in gpr to pass in call instruction
    const auto& brgemm_kernel_overload = static_cast<void (*)(libxsmm_xmmfunction*,
                                                              void*,
                                                              void*,
                                                              void*)>(kernel_execute_libxsmm);
    h->mov(h->rbp, reinterpret_cast<uintptr_t>(brgemm_kernel_overload));
    // todo: several of addr_{A, B, C} could be also abi_paramX, so one of them could be corrupted
    //  if moving directly h->uni_vmovq(abi_paramX, adr_X). Save them to vector regs to avoid corruption.
    //  It's likely that a more efficient solution exists.
    h->uni_vmovq(Xmm(0), addr_A);
    h->uni_vmovq(Xmm(1), addr_B);
    h->uni_vmovq(Xmm(2), addr_C);
#if 0
    if (m_with_scratch)
        h->uni_vmovq(Xmm(3), scratch);
#endif
    // todo: Windows ABI : requires different num of arguments passed in regs and on the stack. Need to align.
    const auto data_ptr_reg = [&](Xmm xmm, Xbyak::Reg64 reg, size_t bytes_offset) {
        h->uni_vmovq(reg, xmm);
        if (bytes_offset) h->add(reg, bytes_offset);
    };
    h->mov(abi_param1, reinterpret_cast<uintptr_t>(xsmm_func));
    data_ptr_reg(Xmm(0), abi_param2, in0_kernel_offset);
    data_ptr_reg(Xmm(1), abi_param3, in1_kernel_offset);
    data_ptr_reg(Xmm(2), abi_param4, out0_kernel_offset);

#ifdef _WIN32
    // Before function call we should allocate stack area for
    //  - register parameters - ABI parameters (shadow space)
    //  - stack parameters - remaining parameters
    const size_t num_args_passed_on_stack = 6;  // count of function brgemm_kernel_overload() parameters
    size_t abi_param_count = sizeof(abi_param_regs) / sizeof(abi_param_regs[0]);
    h->sub(h->rsp, num_args_passed_on_stack * gpr_size);

    // Push the remaining parameters on the stack
    if (m_with_scratch) {
        h->uni_vmovq(h->qword[h->rsp + (abi_param_count + 0) * gpr_size], Xmm(3));
        if (in2_kernel_offset) h->add(h->qword[h->rsp + (abi_param_count + 0) * gpr_size], in2_kernel_offset);
    } else {
        h->mov(h->qword[h->rsp + (abi_param_count + 0) * gpr_size], reinterpret_cast<uintptr_t>(nullptr));
    }
    h->mov(abi_not_param1, static_cast<int>(m_with_comp));
    h->mov(h->qword[h->rsp + (abi_param_count + 1) * gpr_size], abi_not_param1);
#else
#if 0
    if (m_with_scratch) {
        data_ptr_reg(Xmm(3), abi_param5, in2_kernel_offset);
    } else {
        h->mov(abi_param5, reinterpret_cast<uintptr_t>(nullptr));
    }
    h->mov(abi_param6, static_cast<int>(m_with_comp));
#endif
#endif

    internal_call_rsp_align();
    h->call(h->rbp);
    internal_call_rsp_restore();

#ifdef _WIN32
    h->add(h->rsp, num_args_passed_on_stack * gpr_size);
#endif
    internal_call_postamble();
}

void BrgemmTppEmitter::kernel_execute_libxsmm(libxsmm_xmmfunction *brg_kernel,
                                   void *A, void *B, void *C) {
    libxsmm_gemm_param gemm_p;
    gemm_p.a.primary = reinterpret_cast<void*>(B);
    gemm_p.b.primary = reinterpret_cast<void*>(A);
    gemm_p.c.primary = reinterpret_cast<void*>(C);
    assert(brg_kernel);
    (*brg_kernel).gemm(&gemm_p);
}

void BrgemmTppEmitter::libxsmm_amx_tile_configure(libxsmm_xmmfunction *cfg_kernel) {
    libxsmm_gemm_param gemm_p;
    gemm_p.a.primary = reinterpret_cast<void*>(NULL);
    gemm_p.b.primary = reinterpret_cast<void*>(NULL);
    gemm_p.c.primary = reinterpret_cast<void*>(NULL);
    assert(cfg_kernel);
    (*cfg_kernel).gemm(&gemm_p);
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
    const auto& node = expr->get_node();
    const auto& tpp_node = std::dynamic_pointer_cast<tpp::op::BinaryEltwiseTPP>(node);
    OPENVINO_ASSERT(tpp_node, "BinaryEltwiseTppEmitter invoked with invalid node type");
    const auto& dtype_in0 = ov_to_xsmm_dtype(node->get_input_element_type(0));
    const auto& dtype_in1 = ov_to_xsmm_dtype(node->get_input_element_type(1));
    const auto& dtype_out0 = ov_to_xsmm_dtype(node->get_output_element_type(0));
    // todo: how to derive dtype_comp in a general case?
    const auto& dtype_comp = ov_to_xsmm_dtype(ov::element::Type_t::f32);


    const auto& shape_in0 = expr->get_input_port_descriptor(0)->get_shape();
    const auto& shape_in1 = expr->get_input_port_descriptor(1)->get_shape();
    const libxsmm_blasint ld_in0 = static_cast<libxsmm_blasint>(*++shape_in0.rbegin());
    const libxsmm_blasint ld_in1 = static_cast<libxsmm_blasint>(*++shape_in1.rbegin());

    std::pair<bool, bool> n_bcast_flags, m_bcast_flags;
    const auto N = get_broadcasted_dim(static_cast<libxsmm_blasint>(shape_in0.back()),
                                       static_cast<libxsmm_blasint>(shape_in1.back()),
                                       n_bcast_flags);
    const auto M = get_broadcasted_dim(ld_in0, ld_in1, m_bcast_flags);

    const libxsmm_blasint ld_out0 = N;
    libxsmm_bitfield libxsmm_flags{LIBXSMM_MELTW_FLAG_BINARY_NONE};
    if (n_bcast_flags.first && m_bcast_flags.first) {
        libxsmm_flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0;
    } else if (n_bcast_flags.first) {
        libxsmm_flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
    } else  if (m_bcast_flags.first) {
        libxsmm_flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0;
    }
    if (n_bcast_flags.second && m_bcast_flags.second) {
        libxsmm_flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1;
    } else if (n_bcast_flags.second) {
        libxsmm_flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1;
    } else  if (m_bcast_flags.second) {
        libxsmm_flags |= LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1;
    }
    const auto& libxsmm_op_type = tpp_node->get_op_type();
    const auto& libxsmm_shape = libxsmm_create_meltw_binary_shape(M, N, ld_in0, ld_in1, ld_out0, dtype_in0, dtype_in1, dtype_out0, dtype_comp);
    libxsmm_kernel = libxsmm_dispatch_meltw_binary_v2(libxsmm_op_type, libxsmm_shape, libxsmm_flags);
    OPENVINO_ASSERT(libxsmm_kernel, "Failed to dispatch libxsmm_kernel in BinaryEltwiseTppEmitter");
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

void BinaryEltwiseTppEmitter::execute_libxsmm_kernel(libxsmm_meltwfunction_binary *eltwise_kernel,
                                                    void *in0, void *in1, void *out0) {
    // todo: how do we initialize the libxsmm_meltw_binary_param.op field?
    //  In general, how to use the libxsmm_matrix_arg type? What is the purpose of these primary/secondary/ternary fields?
    libxsmm_meltw_binary_param binary_param;
    binary_param.in0.primary = reinterpret_cast<void*>(in0);
    binary_param.in1.primary = reinterpret_cast<void*>(in1);
    binary_param.out.primary = reinterpret_cast<void*>(out0);
    assert(eltwise_kernel);
    (*eltwise_kernel)(&binary_param);
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
    h->mov(abi_param1, reinterpret_cast<uintptr_t>(libxsmm_kernel));
//    data_ptr_reg(Xmm(0), abi_param2, in0_kernel_offset);
//    data_ptr_reg(Xmm(1), abi_param3, in1_kernel_offset);
//    data_ptr_reg(Xmm(2), abi_param4, out0_kernel_offset);
    data_ptr_reg(Xmm(0), abi_param2, 0);
    data_ptr_reg(Xmm(1), abi_param3, 0);
    data_ptr_reg(Xmm(2), abi_param4, 0);

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
