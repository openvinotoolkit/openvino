// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_copy_b.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "emitters/plugin/x64/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

#define DTYPE_CAST(X) static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(X))

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

BrgemmCopyBKernelConfig::BrgemmCopyBKernelConfig(const element::Type& src_dt, const element::Type& wei_dt, cpu_isa_t isa,
                                                 bool is_with_comp, bool is_transposed_B, dnnl_dim_t wei_N_blk)
    : m_static_params(std::make_shared<StaticParams>(src_dt, wei_dt, isa, is_with_comp, is_transposed_B, wei_N_blk)) {
    m_hash = compute_hash();
}

bool BrgemmCopyBKernelConfig::is_completed() const {
    return !utils::one_of(0, m_N, m_K, m_copy_B_wei_stride, m_LDB) || is_empty();
}

bool BrgemmCopyBKernelConfig::is_empty() const {
    return everyone_is(0, m_N, m_N_blk, m_K, m_K_blk, m_copy_B_wei_stride, m_LDB);
}

bool BrgemmCopyBKernelConfig::operator==(const BrgemmCopyBKernelConfig& rhs) const {
#define EQ(X) X == rhs.X
    return EQ(m_hash) && EQ(m_N) && EQ(m_N_blk) && EQ(m_K) && EQ(m_K_blk) && EQ(m_LDB) && EQ(m_copy_B_wei_stride) &&
           (EQ(m_static_params.get()) || *m_static_params == *(rhs.m_static_params));
#undef EQ
}

void BrgemmCopyBKernelConfig::update(dnnl_dim_t N, dnnl_dim_t N_blk, dnnl_dim_t K, dnnl_dim_t K_blk, dnnl_dim_t copy_B_wei_stride, dnnl_dim_t LDB) {
    // If one of the dims is zero, it means that BrgemmCopyB won't be executed (in Loop with work_amount = 0, for example)
    // To process this case, we have to make this Config as empty (nullify runtime parameters)
    if (utils::one_of(0, N, K)) {
        m_N = 0; m_N_blk = 0;
        m_K = 0; m_K_blk = 0;
        m_copy_B_wei_stride = 0; m_LDB = 0;
    } else {
        m_N = N; m_N_blk = N_blk;
        m_K = K; m_K_blk = K_blk;
        m_copy_B_wei_stride = copy_B_wei_stride; m_LDB = LDB;
    }
    m_hash = compute_hash();
}

size_t BrgemmCopyBKernelConfig::compute_hash() const {
    size_t seed = m_static_params->hash;
#define HASH(X) seed = hash_combine(seed, X)
    HASH(m_N); HASH(m_N_blk);
    HASH(m_K); HASH(m_K_blk);
    HASH(m_copy_B_wei_stride); HASH(m_LDB);
#undef HASH
    return seed;
}

BrgemmCopyBKernelConfig::StaticParams::StaticParams(const element::Type& src_type, const element::Type& wei_type, cpu_isa_t isa,
                                                    bool is_with_comp, bool is_transposed_B, dnnl_dim_t wei_n_blk)
    : src_dt(DTYPE_CAST(src_type)), wei_dt(DTYPE_CAST(wei_type)), isa(isa),
      is_with_comp(is_with_comp), is_transposed_B(is_transposed_B), wei_N_blk(wei_n_blk),
      hash(init_hash(src_dt, wei_dt, isa, is_with_comp, is_transposed_B, wei_N_blk)) {}

bool BrgemmCopyBKernelConfig::StaticParams::operator==(const StaticParams& rhs) const {
#define EQ(X) X == rhs.X
    return EQ(hash) && EQ(src_dt) && EQ(wei_dt)&& EQ(isa) && EQ(is_with_comp) && EQ(is_transposed_B) && EQ(wei_N_blk);
#undef EQ
}

size_t BrgemmCopyBKernelConfig::StaticParams::init_hash(const dnnl_data_type_t& src_dt, const dnnl_data_type_t& wei_dt, cpu_isa_t isa,
                                                        bool is_with_comp, bool is_transposed_B, dnnl_dim_t wei_N_blk) {
    size_t seed = 0;
#define HASH(X) seed = hash_combine(seed, X)
    HASH(src_dt); HASH(wei_dt); HASH(isa);
    HASH(is_with_comp); HASH(is_transposed_B); HASH(wei_N_blk);
#undef HASH
    return seed;
}

#ifdef SNIPPETS_DEBUG_CAPS
#define PRINT(X) ss << #X  << " = " << X << "\n"
std::string BrgemmCopyBKernelConfig::to_string() const {
    std::stringstream ss;
    ss << m_static_params->to_string() << "\n";
    PRINT(m_hash); PRINT(m_N); PRINT(m_N_blk);
    PRINT(m_K); PRINT(m_K_blk); PRINT(m_LDB); PRINT(m_copy_B_wei_stride);
    return ss.str();
}
std::string BrgemmCopyBKernelConfig::StaticParams::to_string() const {
    std::stringstream ss;
    PRINT(src_dt); PRINT(wei_dt); PRINT(isa);
    PRINT(is_with_comp); PRINT(is_transposed_B); PRINT(wei_N_blk);
    return ss.str();
}
#undef PRINT
#endif

BrgemmCopyBKernel::BrgemmCopyBKernel() : jit_generator(jit_name()), ker_(nullptr) {}

BrgemmCopyBKernel::BrgemmCopyBKernel(const BrgemmCopyBKernelConfig& conf)
    : jit_generator(jit_name()), is_with_comp(conf.is_with_comp()), is_transpose(conf.is_transposed_B()),
      wei_data_size(dnnl_data_type_size(conf.get_wei_dt())), vnni_factor(data_type_vnni_granularity(conf.get_wei_dt())),
      K(conf.get_K()), N_blk(conf.get_N_blk()), wei_N_blk(conf.get_wei_N_blk()), wei_N_tail(conf.get_wei_N_tail()), ker_(nullptr) {
    init_brgemm_copy_b_kernel(dnnl_brgemm_copy_b_kernel, conf);
    OV_CPU_JIT_EMITTER_ASSERT(dnnl_brgemm_copy_b_kernel, "Kernel is missed!");
}

status_t BrgemmCopyBKernel::create_kernel() {
    const auto code = jit_generator::create_kernel();
    OV_CPU_JIT_EMITTER_ASSERT(code == status::success, "Failed to create kernel");
    ker_ = (decltype(ker_))jit_ker();
    return code;
}

void BrgemmCopyBKernel::operator()(const call_args* args) const {
    OV_CPU_JIT_EMITTER_ASSERT(ker_, "Kernel is nullptr");
    ker_(args);
}

void BrgemmCopyBKernel::init_brgemm_copy_b_kernel(std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t>& kernel,
                                                  const BrgemmCopyBKernelConfig& conf) const {
    matmul::brgemm_matmul_conf_t brgCopyKernelConf;
    brgCopyKernelConf.src_dt = conf.get_src_dt();
    brgCopyKernelConf.wei_dt = conf.get_wei_dt();
    brgCopyKernelConf.orig_wei_dt = brgCopyKernelConf.wei_dt;
    brgCopyKernelConf.wei_n_blk = static_cast<int>(conf.get_wei_N_blk());
    // Note: 2D format tags are used just to force the needed OneDNN primitive creation.
    // However, the generated primitive can be also applied to tensors with other ranks
    brgCopyKernelConf.wei_tag = conf.is_transposed_B() ? dnnl_ba : dnnl_ab;
    brgCopyKernelConf.transposed_B = conf.is_transposed_B();
    brgCopyKernelConf.copy_B_wei_stride = conf.get_copy_B_wei_stride();
    brgCopyKernelConf.LDB = conf.get_LDB();
    brgCopyKernelConf.N = conf.get_N();
    brgCopyKernelConf.N_tail = conf.get_wei_N_tail();
    brgCopyKernelConf.N_blk = conf.get_wei_N_blk();
    brgCopyKernelConf.K = conf.get_K_blk();
    brgCopyKernelConf.K_blk = conf.get_K_blk();
    brgCopyKernelConf.N_chunk_elems = brgCopyKernelConf.N_blk;
    brgCopyKernelConf.b_dt_sz = DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.wei_dt));
    brgCopyKernelConf.tr_b_dt_sz = DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.wei_dt));

    brgCopyKernelConf.req_wei_vnni_downconvert = false;

    brgCopyKernelConf.isa = conf.get_isa();
    brgCopyKernelConf.s8s8_compensation_required = conf.is_with_comp();

    brgCopyKernelConf.has_zero_point_a = false;
    brgCopyKernelConf.has_zero_point_b = false;
    brgCopyKernelConf.src_zp_type = dnnl::impl::cpu::x64::none;

    OV_CPU_JIT_EMITTER_ASSERT(matmul::create_brgemm_matmul_copy_b(kernel, &brgCopyKernelConf) == dnnl_success,
                              "cannot create kernel due to invalid params");
}

void BrgemmCopyBKernel::generate() {
    preamble();

    mov(src_reg, ptr[abi_param1 + GET_OFF_BRGEMM_COPY_B_ARGS(src)]);
    mov(tr_src_reg, ptr[abi_param1 + GET_OFF_BRGEMM_COPY_B_ARGS(tr_src)]);
    if (is_with_comp)
        mov(comp_reg, ptr[abi_param1 + GET_OFF_BRGEMM_COPY_B_ARGS(compensation_ptr)]);

    size_t start_in = 0;
    size_t start_out = 0;
    size_t start_comp = 0;

    auto add_ptr_increments = [&](size_t current_N) {
        start_in += is_transpose ? K * current_N * wei_data_size : current_N * wei_data_size;
        start_out += current_N * vnni_factor * wei_data_size;
        start_comp += is_with_comp ? current_N * sizeof(int32_t) : 0;
    };

    // OneDNN requires tail handling before main iterations
    if (wei_N_tail != 0) {
        emit_brgemm_copy_b_kernel_call(wei_N_tail, K, start_in, start_out, start_comp);
        add_ptr_increments(wei_N_tail);
    }

    for (auto nb = wei_N_tail; nb < N_blk; nb += wei_N_blk) {
        emit_brgemm_copy_b_kernel_call(wei_N_blk, K, start_in, start_out, start_comp);
        add_ptr_increments(wei_N_blk);
    }

    postamble();
}

void BrgemmCopyBKernel::emit_brgemm_copy_b_kernel_call(size_t N, size_t K, size_t offset_in, size_t offset_out, size_t offset_comp) {
    EmitABIRegSpills spill(this);
    spill.preamble();

    const auto add_offset = [&](Xbyak::Reg64 reg, size_t bytes_offset) {
        if (bytes_offset) add(reg, bytes_offset);
    };

    // save function address in gpr to pass in call instruction
    const auto& kernel_overload = static_cast<void (*)(matmul::jit_brgemm_matmul_copy_b_t*, const void*, const void*, const void*, size_t, size_t)>(execute);
    mov(rbp, reinterpret_cast<uintptr_t>(kernel_overload));
    mov(abi_param1, reinterpret_cast<uintptr_t>(dnnl_brgemm_copy_b_kernel.get()));

    add_offset(src_reg, offset_in); // abi_param2
    add_offset(tr_src_reg, offset_out); // abi_param3
    if (is_with_comp)  // abi_param4
        add_offset(comp_reg, offset_comp);
    else
        mov(comp_reg, reinterpret_cast<uintptr_t>(nullptr));

#ifdef _WIN32
    // Note: ABI requires that the remaining parameters (except the first for) are pushed to the stack in right-to-left order
    //  Shadow space will be allocated inside internal_call_rsp_align()
    push(K);
    push(N);
#else
    mov(abi_param5, N);
    mov(abi_param6, K);
#endif

    spill.rsp_align();
    call(rbp);
    spill.rsp_restore();

#ifdef _WIN32
    static constexpr int gpr_size = 8;
    add(rsp, gpr_size * 2);
#endif

    spill.postamble();
}

void BrgemmCopyBKernel::execute(matmul::jit_brgemm_matmul_copy_b_t* kernel, const void* src, const void* dst, const void* comp, size_t N, size_t K) {
    auto ctx = matmul::jit_brgemm_matmul_copy_b_t::ctx_t();
    ctx.current_N_blk = N;
    ctx.src = src;
    ctx.tr_src = dst;
    ctx.compensation_ptr = comp;
    ctx.zp_a_compensation_ptr = nullptr;
    ctx.zp_a_neg_value_ptr = nullptr;
    ctx.current_K_start = 0;
    ctx.current_K_iters = K;

    OV_CPU_JIT_EMITTER_ASSERT(kernel, "Kernel hasn't been created");
    (*kernel)(&ctx);
}

BrgemmCopyBKernelExecutor::BrgemmCopyBKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmCopyBKernelConfig config)
    : CPUKernelExecutor<BrgemmCopyBKernelConfig, BrgemmCopyBKernel>(std::move(kernel_cache), std::move(config)) { }

std::shared_ptr<BrgemmCopyBKernel> BrgemmCopyBKernelExecutor::compile_kernel(const BrgemmCopyBKernelConfig& config) const {
    std::shared_ptr<BrgemmCopyBKernel> compiled_kernel = std::make_shared<BrgemmCopyBKernel>();
    // BrgemmCopyB is not executable - nothing to compile
    if (!config.is_empty()) {
        compiled_kernel = std::make_shared<BrgemmCopyBKernel>(config);
        OV_CPU_JIT_EMITTER_ASSERT(compiled_kernel, "compiled kernel is nullptr");
        compiled_kernel->create_kernel();
    }

    return compiled_kernel;
}

void BrgemmCopyBKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                              const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                              BrgemmCopyBKernelConfig& config) const {
    const auto& input_desc = expr->get_input_port_descriptor(0);
    const auto& output_desc = expr->get_output_port_descriptor(0);

    // Need to update K, N
    // 1. If the original value in subtensor is `FULL_DIM`, it means that
    //    BrgemmCopyB block should process full tensor by this dim -> take dimension from shape
    // 2. Otherwise, BrgemmCopyB block processes part of the tensor by this dim
    //    (there is blocking by this dimension) -> take from Loop increment

    const auto planar_shape = ov::snippets::utils::get_planar_vdims(expr->get_input_port(0));
    const auto& in_subtensor = input_desc->get_subtensor();

    size_t loop_idx = 0;
    const auto& loop_ids = expr->get_loop_ids();
    const auto& loop_manager = linear_ir->get_loop_manager();

    auto init = [&](size_t& dim, size_t& blk, size_t idx) {
        OPENVINO_ASSERT(idx < planar_shape.size() && idx < in_subtensor.size(), "Index must be less than shape/subtensor rank!");
        dim = *(planar_shape.rbegin() + idx);
        blk = *(in_subtensor.rbegin() + idx);
        if (ov::snippets::utils::is_full_dim_value(blk)) {
            blk = dim;
        } else {
            OPENVINO_ASSERT(loop_idx < loop_ids.size(), "Loop is missed");
            const auto& current_expanded_loop_info = loop_manager->get_loop_info<ov::snippets::lowered::ExpandedLoopInfo>(loop_ids[loop_idx++]);
            blk = current_expanded_loop_info->get_increment();
            input_desc->set_subtensor_dim(idx, blk);
            output_desc->set_subtensor_dim(idx, blk);
            OV_CPU_JIT_EMITTER_ASSERT(blk <= dim, "BrgemmCopyB has incompatible subtensor dimensions");
        }
    };

    size_t K_dim, K_blk, N_dim, N_blk;
    //  Dimension K
    init(K_dim, K_blk, 1);
    //  Dimension N
    init(N_dim, N_blk, 0);

    const auto& brg_weight_etype = expr->get_node()->get_input_element_type(0);
    const auto LDB = brgemm_utils::repacking::compute_out_leading_dim(N_dim, brg_weight_etype);
    const auto copy_B_wei_stride = ov::snippets::utils::get_dim_stride(expr->get_input_port(0), config.is_transposed_B() ? 0 : 1) * brg_weight_etype.size();

    config.update(N_dim, N_blk, K_dim, K_blk, copy_B_wei_stride, LDB);
}

void BrgemmCopyBKernelExecutor::execute(const BrgemmCopyBKernelExecutor* executor, BrgemmCopyBKernel::call_args* args) {
    auto kernel = executor->get_kernel();
    OV_CPU_JIT_EMITTER_ASSERT(kernel, "has nullptr kernel");
    OV_CPU_JIT_EMITTER_ASSERT(args, "has nullptr call args");
    (*kernel)(args);
}

}   // namespace intel_cpu
}   // namespace ov
