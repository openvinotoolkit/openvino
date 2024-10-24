// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm.hpp"

#include <cpu/x64/amx_tile_configure.hpp>

#include "common/utils.hpp"
#include "dnnl_extension_utils.h"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/insert_specific_iterations.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#include "onednn/dnnl.h"

#define DIM_CAST(X) static_cast<dnnl_dim_t>(X)
#define DTYPE_CAST(X) static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(X))

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace {
size_t init_hash(dnnl_data_type_t dt_in0, dnnl_data_type_t dt_in1, bool is_with_amx,
                 bool is_with_comp, bool is_c_pre_scale, dnnl::impl::cpu::x64::cpu_isa_t isa) {
    size_t seed = 0;
#define HASH(X) seed = hash_combine(seed, X)
    HASH(dt_in0); HASH(dt_in1);
    HASH(is_with_amx); HASH(is_with_comp); HASH(is_c_pre_scale);
    HASH(isa);
#undef HASH
    return seed;
}
} // namespace

namespace ov {
namespace intel_cpu {
BrgemmKernelConfig::BrgemmKernelConfig(const element::Type& in0_dtype, const element::Type& in1_dtype,
                                       bool is_with_amx, bool is_with_comp, bool is_c_pre_scale,
                                       dnnl::impl::cpu::x64::cpu_isa_t primitive_isa) :
                                       m_static_params(std::make_shared<StaticParams>(in0_dtype, in1_dtype,
                                                                                      is_with_amx, is_with_comp, is_c_pre_scale,
                                                                                      primitive_isa)) {
    m_hash = compute_hash();
}

bool BrgemmKernelConfig::is_completed() const {
    return !utils::one_of(0, m_M, m_N, m_K, m_LDA, m_LDB, m_LDC) || is_empty();
}

bool BrgemmKernelConfig::operator==(const BrgemmKernelConfig& rhs) const {
#define EQ(X) X == rhs.X
    return EQ(m_hash) && EQ(m_beta) &&
           EQ(m_M) && EQ(m_N) && EQ(m_K) &&
           EQ(m_LDA) && EQ(m_LDB) && EQ(m_LDC) &&
           (EQ(m_static_params.get()) || *m_static_params == *(rhs.m_static_params));
#undef EQ
}

void BrgemmKernelConfig::update(dnnl_dim_t M, dnnl_dim_t N, dnnl_dim_t K, dnnl_dim_t LDA, dnnl_dim_t LDB, dnnl_dim_t LDC, float beta) {
    // If M is zero, it means that Brgemm won't be executed (in Loop with work_amount = 0, for example)
    // To process this case, we have to make this Config as empty (nullify runtime parameters)
    if (utils::one_of(0, M, N, K)) {
        m_M = 0; m_N = 0; m_K = 0;
        m_LDA = 0; m_LDB = 0; m_LDC = 0;
        m_beta = 0;
    } else {
        m_M = M; m_N = N; m_K = K;
        m_LDA = LDA; m_LDB = LDB; m_LDC = LDC;
        m_beta = beta;
    }
    m_hash = compute_hash();
}

bool BrgemmKernelConfig::is_empty() const {
    return everyone_is(0, m_M, m_N, m_K, m_LDA, m_LDB, m_LDC, m_beta);
}

BrgemmKernelConfig::operator amx_tile_config_t() const {
    amx_tile_config_t res;
    res.M = m_M; res.N = m_N; res.K = m_K;
    return  res;
}

BrgemmKernelConfig::StaticParams::StaticParams(const element::Type& in0_dtype, const element::Type& in1_dtype,
                                               bool is_with_amx, bool is_with_comp, bool is_c_pre_scale,
                                               dnnl::impl::cpu::x64::cpu_isa_t primitive_isa) :
                                               dt_in0(DTYPE_CAST(in0_dtype)), dt_in1(DTYPE_CAST(in1_dtype)),
                                               is_with_amx(is_with_amx), is_with_comp(is_with_comp), is_c_pre_scale(is_c_pre_scale),
                                               isa(primitive_isa),
                                               hash(init_hash(dt_in0, dt_in1, is_with_amx, is_with_comp, is_c_pre_scale, isa)) {
}

bool BrgemmKernelConfig::StaticParams::operator==(const StaticParams& rhs) const {
#define EQ(X) X == rhs.X
    return EQ(hash) && EQ(dt_in0) && EQ(dt_in1)&& EQ(is_with_amx) && EQ(is_with_comp) && EQ(isa);
#undef EQ
}
size_t BrgemmKernelConfig::compute_hash() const {
    size_t seed = m_static_params->hash;
#define HASH(X) seed = hash_combine(seed, X)
    HASH(m_M); HASH(m_N); HASH(m_K);
    HASH(m_LDA); HASH(m_LDB); HASH(m_LDC);
    HASH(m_beta);
#undef HASH
    return seed;
}

#ifdef SNIPPETS_DEBUG_CAPS
#define PRINT(X) ss << #X  << " = " << X << "\n"
std::string BrgemmKernelConfig::StaticParams::to_string() const {
    std::stringstream ss;
    PRINT(dt_in0); PRINT(dt_in1);
    PRINT(is_with_amx); PRINT(is_with_comp);
    PRINT(isa);
    return ss.str();
}

std::string BrgemmKernelConfig::to_string() const {
    std::stringstream ss;
    ss << m_static_params->to_string() << "\n";
    PRINT(m_M); PRINT(m_N); PRINT(m_K);
    PRINT(m_LDA); PRINT(m_LDB); PRINT(m_LDC);
    PRINT(m_beta);
    return ss.str();
}
#undef PRINT
#endif

template <cpu_isa_t isa>
struct jit_c_pre_scale_kernel_f32 : public jit_c_pre_scale_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_c_pre_scale_kernel_f32)

    explicit jit_c_pre_scale_kernel_f32(jit_c_pre_scale_params jcp) : jit_c_pre_scale_kernel(jcp), jit_generator(jit_name()) {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }
    // src and dst are the same. shape agnostic.
    void generate() override {
        load_pool_gpr_idxs = {static_cast<size_t>(reg_tmp1_64.getIdx()), static_cast<size_t>(reg_tmp2_64.getIdx())};
        store_pool_gpr_idxs = {static_cast<size_t>(reg_tmp1_64.getIdx())};
        store_pool_vec_idxs = {static_cast<size_t>(vmm_zero.getIdx())};
        load_emitter = std::unique_ptr<jit_load_emitter>(new jit_load_emitter(this, isa, jcp_.data_prc, ov::element::f32, vector_step));
        load_emitter_tail = std::unique_ptr<jit_load_emitter>(new jit_load_emitter(this, isa, jcp_.data_prc, ov::element::f32, 1));
        store_emitter = std::unique_ptr<jit_store_emitter>(new jit_store_emitter(this, isa, ov::element::f32, jcp_.data_prc, vector_step));
        store_emitter_tail = std::unique_ptr<jit_store_emitter>(new jit_store_emitter(this, isa, ov::element::f32, jcp_.data_prc, 1));

        this->preamble();
        mov(reg_data, ptr[reg_params + GET_OFF_BRGEMM_C_PRE_SCALE_ARGS(data_ptr)]);
        mov(reg_scale, ptr[reg_params + GET_OFF_BRGEMM_C_PRE_SCALE_ARGS(scale_ptr)]);
        mov(reg_work_amount_M, ptr[reg_params + GET_OFF_BRGEMM_C_PRE_SCALE_ARGS(work_amount_M)]);

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        Xbyak::Label m_loop_label;
        Xbyak::Label m_loop_end_label;
        L(m_loop_label);
        {
            cmp(reg_work_amount_M, 0);
            jle(m_loop_end_label, T_NEAR);

            mov(reg_work_amount_N, ptr[reg_params + GET_OFF_BRGEMM_C_PRE_SCALE_ARGS(work_amount_N)]); // reset N for each M
            uni_vbroadcastss(vmm_scale, ptr[reg_scale]);

            Xbyak::Label n_loop_label;
            Xbyak::Label n_loop_end_label;
            L(n_loop_label);
            {
                cmp(reg_work_amount_N, vector_step);
                jl(n_loop_end_label, T_NEAR);

                load_emitter->emit_code({static_cast<size_t>(reg_data.getIdx())},
                                        {static_cast<size_t>(vmm_data.getIdx())}, {}, {load_pool_gpr_idxs});
                uni_vfmadd231ps(vmm_data, vmm_data, vmm_scale);
                store_emitter->emit_code({static_cast<size_t>(vmm_data.getIdx())}, {static_cast<size_t>(reg_data.getIdx())},
                                         {store_pool_vec_idxs}, {store_pool_gpr_idxs});

                add(reg_data, vector_step * jcp_.data_prc.size());
                sub(reg_work_amount_N, vector_step);
                jmp(n_loop_label, T_NEAR);
            }
            L(n_loop_end_label);

            // tail n
            Xbyak::Label n_loop_tail_label;
            Xbyak::Label n_loop_tail_end_label;
            L(n_loop_tail_label);
            {
                cmp(reg_work_amount_N, 0);
                jle(n_loop_tail_end_label, T_NEAR);

                load_emitter_tail->emit_code({static_cast<size_t>(reg_data.getIdx())},
                                             {static_cast<size_t>(vmm_data.getIdx())}, {}, {load_pool_gpr_idxs});
                uni_vfmadd231ps(vmm_data, vmm_data, vmm_scale);
                store_emitter_tail->emit_code({static_cast<size_t>(vmm_data.getIdx())}, {static_cast<size_t>(reg_data.getIdx())},
                                              {store_pool_vec_idxs}, {store_pool_gpr_idxs});

                add(reg_data, 1 * jcp_.data_prc.size());
                sub(reg_work_amount_N, 1);
                jmp(n_loop_tail_label, T_NEAR);
            }
            L(n_loop_tail_end_label);

            add(reg_scale, 1 * jcp_.scale_prc.size());
            sub(reg_work_amount_M, 1);
            jmp(m_loop_label, T_NEAR);
        }
        L(m_loop_end_label);

        this->postamble();

        load_emitter->emit_data();
        load_emitter_tail->emit_data();
        store_emitter->emit_data();
        store_emitter_tail->emit_data();
    }

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;
    const int vlen = cpu_isa_traits<isa>::vlen;
    const int vector_step = vlen / sizeof(float);
    Xbyak::Reg64 reg_params = abi_param1;
    Xbyak::Reg64 reg_data = r8;
    Xbyak::Reg64 reg_scale = r9;
    Xbyak::Reg64 reg_work_amount_M = r10;
    Xbyak::Reg64 reg_work_amount_N = r11;
    Xbyak::Reg64 reg_tmp1_64 = r14;
    Xbyak::Reg64 reg_tmp2_64 = r15;

    Vmm vmm_scale = Vmm(1);
    Vmm vmm_data = Vmm(2);
    Vmm vmm_zero = Vmm(3);

    std::unique_ptr<jit_load_emitter> load_emitter;
    std::unique_ptr<jit_load_emitter> load_emitter_tail;
    std::unique_ptr<jit_store_emitter> store_emitter;
    std::unique_ptr<jit_store_emitter> store_emitter_tail;

    std::vector<size_t> store_pool_gpr_idxs;
    std::vector<size_t> store_pool_vec_idxs;
    std::vector<size_t> load_pool_gpr_idxs;
};

BrgemmKernelExecutor::BrgemmKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmKernelConfig config) :
        CPUKernelExecutor<BrgemmKernelConfig, BrgemmCompiledKernel>(std::move(kernel_cache), std::move(config)) { }


std::shared_ptr<BrgemmCompiledKernel> BrgemmKernelExecutor::compile_kernel(const BrgemmKernelConfig& config) const {
    std::shared_ptr<BrgemmCompiledKernel> compiled_kernel = std::make_shared<BrgemmCompiledKernel>();

    // Brgemm is not executable - nothing to compile
    if (config.is_empty())
        return compiled_kernel;

    // std::cout << "config.get_LDA():" << config.get_LDA() << std::endl;
    // std::cout << "config.get_LDB():" << config.get_LDB() << std::endl;
    // std::cout << "config.get_LDC():" << config.get_LDC() << std::endl;

    cpu::x64::brgemm_desc_t desc;
    auto status = brgemm_desc_init(&desc, config.get_isa(), cpu::x64::brgemm_strd,
                                   config.get_dt_in0(), config.get_dt_in1(),
                                   false, false, cpu::x64::brgemm_row_major, 1.f,
                                   config.get_beta(),
                                   config.get_LDA(), config.get_LDB(), config.get_LDC(),
                                   config.get_M(), config.get_N(), config.get_K(), nullptr);
    OV_CPU_JIT_EMITTER_ASSERT(status == dnnl_success, "Cannot initialize brgemm descriptor due to invalid params");

    if (config.is_with_amx()) {
        status = brgemm_init_tiles(desc, compiled_kernel->palette);
        OV_CPU_JIT_EMITTER_ASSERT(status == dnnl_success, "Cannot initialize brgemm tiles due to invalid params");
    }

    cpu::x64::brgemm_kernel_t* kernel_ = nullptr;
    status = brgemm_kernel_create(&kernel_, desc);
    OV_CPU_JIT_EMITTER_ASSERT(status == dnnl_success, "Cannot create brgemm kernel due to invalid params");
    compiled_kernel->compiled_kernel = std::unique_ptr<brgemm_kernel_t>(kernel_);

    // compile C pre scale kernel
    if (config.is_c_pre_scale()) {
        jit_c_pre_scale_params jcp;
        jcp.data_prc = ov::element::f32;
        jcp.scale_prc = ov::element::f32;
        if (cpu::x64::avx512_core == config.get_isa()) {
            compiled_kernel->c_pre_scale_kernel.reset(new jit_c_pre_scale_kernel_f32<cpu::x64::avx512_core>(jcp));
        } else if (cpu::x64::avx2 == config.get_isa()) {
            compiled_kernel->c_pre_scale_kernel.reset(new jit_c_pre_scale_kernel_f32<cpu::x64::avx2>(jcp));
        }
    }

    return compiled_kernel;
}
float BrgemmKernelExecutor::get_beta(const ov::snippets::lowered::LoopManagerPtr& loop_manager, int loop_id,
                                     const ov::snippets::lowered::ExpandedLoopInfoPtr& current_expanded_loop_info) {
    // Find all Expanded loops with the same Unified loop information -> they were decomposed from this Unified Loop.
    // Note that LoopInfo are normalized and sorted (due to NormalizedLoopIDs pass).
    // It means that previous executed Loops have Loop ID less the current Loop ID.
    // - If there is executed Loop (work_amount > 0) and evaluated before the current -> the current Brgemm should have `beta = 1`.
    // - If there is not this Loop -> the current executed Brgemm should have `beta = 0`.
    if (loop_id > 0) {
        const auto& current_unified_loop_info = current_expanded_loop_info->get_unified_loop_info();
        // Check the previous Loops
        --loop_id;
        while (loop_id >= 0) {
            const auto& expanded_loop_info = loop_manager->get_loop_info<ov::snippets::lowered::ExpandedLoopInfo>(loop_id);
            if (expanded_loop_info->get_unified_loop_info() != current_unified_loop_info)
                return 0;
            if (expanded_loop_info->get_work_amount() > 0) {
                // there is previous executed Brgemm with `beta = 0` -> the current Brgemm should have `beta = 1`
                return 1;
            }
            --loop_id;
        }
    }
    return 0;
}
void BrgemmKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                         const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                         BrgemmKernelConfig& config) const {
    const auto& input_pds = expr->get_input_port_descriptors();
    const auto& output_pds = expr->get_output_port_descriptors();
    OV_CPU_JIT_EMITTER_ASSERT((input_pds.size() == 2 || input_pds.size() == 3) && output_pds.size() == 1,
                              "Invalid number of in/out port descriptors");

    const auto in0_shape = snippets::utils::get_planar_vdims(input_pds[0]->get_shape(), input_pds[0]->get_layout());
    const auto in1_shape = snippets::utils::get_planar_vdims(input_pds[1]->get_shape(), input_pds[1]->get_layout());
    auto in0_subtensor = input_pds[0]->get_subtensor();
    auto in1_subtensor = input_pds[1]->get_subtensor();

    // Need to update M, K, N
    // 1. If the original value in subtensor is `FULL_DIM`, it means that
    //    Brgemm block should process full tensor by this dim -> take dimension from shape
    // 2. Otherwise, Brgemm block processes part of the tensor by this dim
    //    (there is blocking by this dimension) -> take from Loop increment

    auto M = *++in0_subtensor.rbegin();
    auto K = *in0_subtensor.rbegin();
    auto N = *in1_subtensor.rbegin();

    size_t loop_idx = 0;
    const auto& loop_ids = expr->get_loop_ids();
    const auto& loop_manager = linear_ir->get_loop_manager();
    auto get_loop_info = [&](){
        OPENVINO_ASSERT(loop_idx < loop_ids.size(), "Loop is missed");
        return loop_manager->get_loop_info<ov::snippets::lowered::ExpandedLoopInfo>(loop_ids[loop_idx++]);
    };

    /* ------- Dimension M ----------*/
    if (ov::snippets::utils::is_full_dim_value(M)) {
        M = *++in0_shape.rbegin();
    } else {
        const auto& current_expanded_loop_info = get_loop_info();
        const auto& in_ports = current_expanded_loop_info->get_input_ports();
        const auto& out_ports = current_expanded_loop_info->get_output_ports();
        // Quick validation check: Should we check that port is really Brgemm port?
        // If BrgemmCopyB in the Loop by M -> first input port will be BrgemmCopyB with `incremented=false`
        // to avoid extra checks, we validate only first input port
        // Note: We check `is_incremented` attribute only for not incremented ports because
        //       this `is_incremented = true` can be changed by `CleanRepeatedDataPointerShifts` optimization
        auto check_port = [&](const ov::snippets::lowered::LoopPort& p) { return p.dim_idx == 1; };
        OPENVINO_ASSERT(in_ports.size() > 1 && std::all_of(in_ports.cbegin(), in_ports.cend(), check_port) &&
                        out_ports.size() == 1 && check_port(out_ports.back()),
                        "Incorrect Loop by Brgemm dimension M");
        M = current_expanded_loop_info->get_increment();
        input_pds[0]->set_subtensor_dim(1, M);
        output_pds[0]->set_subtensor_dim(1, M);
    }

    /* ------- Dimension N ----------*/
    if (ov::snippets::utils::is_full_dim_value(N)) {
        N = *in1_shape.rbegin();
    } else {
        const auto& current_expanded_loop_info = get_loop_info();
        // const auto& in_ports = current_expanded_loop_info->get_input_ports();
        // const auto& out_ports = current_expanded_loop_info->get_output_ports();
        // Quick validation check: Should we check that port is really Brgemm port?
        // Note: We check `is_incremented` attribute only for not incremented ports because
        //       this `is_incremented = true` can be changed by `CleanRepeatedDataPointerShifts` optimization
        // auto check_port = [&](const ov::snippets::lowered::LoopPort& p) { return p.dim_idx == 0; };
        // std::cout << "in_ports.size():" << in_ports.size() << std::endl;
        // std::cout << "in_ports.front().is_incremented:" << in_ports.front().is_incremented << std::endl;
        // for (size_t i = 0; i < in_ports.size(); i++) {
        //     std::cout << "in_ports_idx:" << in_ports[i].dim_idx << std::endl;
        // }
        // std::cout << "out_ports.size():" << out_ports.size() << std::endl;
        // std::cout << "out_ports_idx:" << out_ports[0].dim_idx << std::endl;
        // Not valide check when block is fused more ops
        // OPENVINO_ASSERT((in_ports.size() == 2 || in_ports.size() == 3) &&
        //                 !in_ports.front().is_incremented && std::all_of(in_ports.cbegin(), in_ports.cend(), check_port) &&
        //                 out_ports.size() == 1 && check_port(out_ports.back()),
        //                 "Incorrect Loop by Brgemm dimension N");
        N = current_expanded_loop_info->get_increment();
        input_pds[1]->set_subtensor_dim(0, N);
        output_pds[0]->set_subtensor_dim(0, N);
    }
    const auto& brgemm_node = as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_node, "Got invalid node type in update_config");

    /* ------- Dimension K ----------*/
    // 1. If Brgemm block processes full dimension K -> `beta = 0`
    // 2. If Brgemm block processes part of the dimension K (there is blocking), need to find
    //    the most first executed Brgemm Block in Loops which iterate through dimension K (work_amount > 0).
    //    First of them will have `beta = 0`, other - `beta = 1`
    float beta = 0;
    if (ov::snippets::utils::is_full_dim_value(K)) {
        K = *in0_shape.rbegin();
    } else {
        const auto& current_expanded_loop_info = get_loop_info();
        // const auto& in_ports = current_expanded_loop_info->get_input_ports();
        // const auto& out_ports = current_expanded_loop_info->get_output_ports();
        // std::cout << "K_in_ports.size():" << in_ports.size() << std::endl;
        // for (size_t i = 0; i < in_ports.size(); i++) {
        //     std::cout << "K_in_ports_idx:" << in_ports[i].dim_idx << std::endl;
        // }
        // std::cout << "K_out_ports.size():" << out_ports.size() << std::endl;
        // std::cout << "K_out_ports_idx:" << out_ports[0].dim_idx << std::endl;
        // std::cout << "K_out_ports.front().is_incremented:" << out_ports.front().is_incremented << std::endl;
        // Not valide check when block is fused more ops
        // Quick validation check: Should we check that port is really Brgemm port?
        // Note: We check `is_incremented` attribute only for not incremented ports because
        //       this `is_incremented = true` can be changed by `CleanRepeatedDataPointerShifts` optimization
        // OPENVINO_ASSERT(in_ports.size() == 2 && in_ports.front().dim_idx == 0 && in_ports.back().dim_idx == 1 &&
        //                 out_ports.size() == 1 && !out_ports.front().is_incremented,
        //                 "Incorrect Loop by Brgemm dimension K");
        K = current_expanded_loop_info->get_increment();
        input_pds[0]->set_subtensor_dim(0, K);
        input_pds[1]->set_subtensor_dim(1, K);
        if (K > 0)
            beta = get_beta(loop_manager, static_cast<int>(loop_ids.back()), current_expanded_loop_info);
    }

    auto LDA = DIM_CAST(snippets::utils::get_dim_stride(expr->get_input_port(0)));
    // input1 could from buffer by blocks, set LDA based on buffer shape
    auto parent_expr = expr->get_input_port_connector(0)->get_source().get_expr();
    if (ov::is_type<snippets::op::Buffer>(parent_expr->get_node())) {
        LDA = parent_expr->get_output_port_descriptor(0)->get_shape().back();
    }
    auto LDB = DIM_CAST(snippets::utils::get_dim_stride(expr->get_input_port(1)));
    // In case of data repacking LDB is chosen in accordance with repacking buffer size
    if (with_repacking(brgemm_node->get_type()))
        LDB = brgemm_utils::repacking::compute_out_leading_dim(N, brgemm_node->get_input_element_type(1));
    auto LDC = DIM_CAST(snippets::utils::get_dim_stride(expr->get_output_port(0)));
    // ouput could be buffer by blocks, set LDA based on buffer shape
    auto child_expr = expr->get_output_port_connector(0)->get_consumers().begin()->get_expr();
    if (ov::is_type<snippets::op::Buffer>(child_expr->get_node())) {
        LDC = child_expr->get_input_port_descriptor(0)->get_shape().back();
    }

    config.update(DIM_CAST(M), DIM_CAST(N), DIM_CAST(K), LDA, LDB, LDC, beta);
}

void BrgemmKernelExecutor::execute(const BrgemmKernelExecutor* executor, call_args* args) {
    auto kernel = executor->get_kernel();
    const auto& config = static_cast<const BrgemmKernelConfig&>(executor->get_config());
    OV_CPU_JIT_EMITTER_ASSERT(kernel, "has nullptr compiler kernel or invalid config");

    if (config.is_c_pre_scale()) {
        jit_c_pre_scale_call_args c_pre_scale_args;
        c_pre_scale_args.data_ptr = args->C;
        c_pre_scale_args.scale_ptr = args->C_pre_scale;
        c_pre_scale_args.work_amount_M = config.get_M();
        c_pre_scale_args.work_amount_N = config.get_N();

        (*kernel->c_pre_scale_kernel)(&c_pre_scale_args);
    }

    const auto tile_config = args->amx_tile_config;
    if (config.is_with_amx() && tile_config && !config.compatible(tile_config)) {
        *tile_config = static_cast<amx_tile_config_t>(config);
        cpu::x64::amx_tile_configure(kernel->palette);
    }

    cpu::x64::brgemm_kernel_params_t brgemm_p;

    brgemm_p.batch = nullptr;  // default value
    brgemm_p.ptr_A = args->A;
    brgemm_p.ptr_B = args->B;
    brgemm_p.ptr_C = args->C;
    brgemm_p.ptr_D = args->C;
    brgemm_p.ptr_buf = args->scratch;
    brgemm_p.ptr_bias = nullptr;
    brgemm_p.do_post_ops = static_cast<size_t>(config.is_with_comp());
    brgemm_p.do_apply_comp = static_cast<size_t>(config.is_with_comp());
    brgemm_p.skip_accm = 0;
    brgemm_p.BS = 1;  // default value
    OV_CPU_JIT_EMITTER_ASSERT(kernel->compiled_kernel, "has nullptr kernel");
    (*kernel->compiled_kernel)(&brgemm_p);
}

#ifdef SNIPPETS_DEBUG_CAPS
BrgemmKernelReferenceExecutor::BrgemmKernelReferenceExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmKernelConfig config) :
        BrgemmKernelExecutor(std::move(kernel_cache), std::move(config)) {
}

std::shared_ptr<BrgemmCompiledKernel> BrgemmKernelReferenceExecutor::compile_kernel(const BrgemmKernelConfig& c) const {
    const auto& res = std::make_shared<BrgemmCompiledKernel>();
    res->compiled_kernel.reset(new brgemm_ref_kernel(c));
    return res;
}

brgemm_ref_kernel::brgemm_ref_kernel(BrgemmKernelConfig c) : m_config(std::move(c)) {
    OV_CPU_JIT_EMITTER_ASSERT(!m_config.is_with_comp() && !m_config.is_with_amx(),
                              "brgemm_ref_kernel doesn't currently support compensations or amx");
    OV_CPU_JIT_EMITTER_ASSERT(m_config.get_dt_in0() == m_config.get_dt_in1() &&
                              m_config.get_dt_in0() == dnnl_data_type_t::dnnl_f32,
                              "brgemm_ref_kernel currently supports only fp32 inputs");
}

void brgemm_ref_kernel::operator()(dnnl::impl::cpu::x64::brgemm_kernel_params_t* args) const {
    auto A = reinterpret_cast<const float*>(args->ptr_A);
    auto B = reinterpret_cast<const float*>(args->ptr_B);
    auto C = reinterpret_cast<float*>(args->ptr_C);
    for (dnnl_dim_t m = 0; m < m_config.get_M(); m++) {
        for (dnnl_dim_t n = 0; n < m_config.get_N(); n++, B++) {
            C[n] = 0;
            for (dnnl_dim_t k = 0; k < m_config.get_K(); k++)
                C[n] += A[k] * B[k * m_config.get_LDB()];
        }
        B -= m_config.get_N();
        A += m_config.get_LDA();
        C += m_config.get_LDC();
    }
}
#endif

}   // namespace intel_cpu
}   // namespace ov
