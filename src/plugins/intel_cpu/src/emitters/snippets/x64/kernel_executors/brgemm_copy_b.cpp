// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_copy_b.hpp"

#include "emitters/plugin/x64/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

#define DTYPE_CAST(X) static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(X))

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

BrgemmCopyBKernelConfig::BrgemmCopyBKernelConfig(const element::Type& src_dt, const element::Type& wei_dt, dnnl_format_tag_t format,
                                                 cpu_isa_t isa, bool is_with_comp, bool is_transposed_B, dnnl_dim_t wei_N_blk)
    : m_static_params(std::make_shared<StaticParams>(src_dt, wei_dt, format, isa, is_with_comp, is_transposed_B, wei_N_blk)) {
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
    // If M is zero, it means that BrgemmCopyB won't be executed (in Loop with work_amount = 0, for example)
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

BrgemmCopyBKernelConfig::StaticParams::StaticParams(const element::Type& src_type, const element::Type& wei_type, dnnl_format_tag_t format,
                                                    cpu_isa_t isa, bool is_with_comp, bool is_transposed_B, dnnl_dim_t wei_n_blk)
    : src_dt(DTYPE_CAST(src_type)), wei_dt(DTYPE_CAST(wei_type)), format(format), isa(isa),
      is_with_comp(is_with_comp), is_transposed_B(is_transposed_B), wei_N_blk(wei_n_blk),
      hash(init_hash(src_dt, wei_dt, format, isa, is_with_comp, is_transposed_B, wei_N_blk)) {}

bool BrgemmCopyBKernelConfig::StaticParams::operator==(const StaticParams& rhs) const {
#define EQ(X) X == rhs.X
    return EQ(hash) && EQ(src_dt) && EQ(wei_dt)&& EQ(format) && EQ(isa) && EQ(is_with_comp) && EQ(is_transposed_B) && EQ(wei_N_blk);
#undef EQ
}

size_t BrgemmCopyBKernelConfig::StaticParams::init_hash(const dnnl_data_type_t& src_dt, const dnnl_data_type_t& wei_dt, dnnl_format_tag_t format,
                                                        cpu_isa_t isa, bool is_with_comp, bool is_transposed_B, dnnl_dim_t wei_N_blk) {
    size_t seed = 0;
#define HASH(X) seed = hash_combine(seed, X)
    HASH(src_dt); HASH(wei_dt); HASH(format); HASH(isa);
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
    PRINT(src_dt); PRINT(wei_dt); PRINT(format); PRINT(isa);
    PRINT(is_with_comp); PRINT(is_transposed_B); PRINT(wei_N_blk);
    return ss.str();
}
#undef PRINT
#endif

BrgemmCopyBKernel::BrgemmCopyBKernel() : jit_generator(jit_name()), ker_(nullptr) {}

BrgemmCopyBKernel::BrgemmCopyBKernel(const BrgemmCopyBKernelConfig& conf)
    : jit_generator(jit_name()), ker_(nullptr), conf(conf), kernel(create_brgemm_copy_b_kernel()) {
    OV_CPU_JIT_EMITTER_ASSERT(kernel, "Kernel is missed!");
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

void BrgemmCopyBKernel::generate() {
    preamble();

    mov(src_reg, ptr[abi_param1 + GET_OFF_BRGEMM_COPY_B_ARGS(src)]);
    mov(tr_src_reg, ptr[abi_param1 + GET_OFF_BRGEMM_COPY_B_ARGS(tr_src)]);
    if (conf.is_with_comp())
        mov(comp_reg, ptr[abi_param1 + GET_OFF_BRGEMM_COPY_B_ARGS(compensation_ptr)]);

    const size_t data_size = dnnl_data_type_size(conf.get_wei_dt());
    const size_t brgemmVNNIFactor = data_type_vnni_granularity(conf.get_wei_dt());
    size_t start_in = 0;
    size_t start_out = 0;
    size_t start_comp = 0;

    auto add_ptr_increments = [&](size_t current_N) {
        start_in += conf.is_transposed_B() ? conf.get_K() * current_N * data_size : current_N * data_size;
        start_out += current_N * brgemmVNNIFactor * data_size;
        start_comp += conf.is_with_comp() ? current_N * sizeof(int32_t) : 0;
    };

    // OneDNN requires tail handling before main iterations
    if (conf.get_wei_N_tail() != 0) {
        emit_brgemm_copy_b_kernel_call(conf.get_wei_N_tail(), conf.get_K(), start_in, start_out, start_comp);
        add_ptr_increments(conf.get_wei_N_tail());
    }

    for (auto nb = conf.get_wei_N_tail(); nb < conf.get_N_blk(); nb += conf.get_wei_N_blk()) {
        emit_brgemm_copy_b_kernel_call(conf.get_wei_N_blk(), conf.get_K(), start_in, start_out, start_comp);
        add_ptr_increments(conf.get_wei_N_blk());
    }

    postamble();
}

std::shared_ptr<BrgemmCopyBKernel::dnnl_brgemm_copy_b_kernel> BrgemmCopyBKernel::create_brgemm_copy_b_kernel() const {
    matmul::brgemm_matmul_conf_t brgCopyKernelConf;
    brgCopyKernelConf.src_dt = conf.get_src_dt();
    brgCopyKernelConf.wei_dt = conf.get_wei_dt();
    brgCopyKernelConf.orig_wei_dt = brgCopyKernelConf.wei_dt;
    brgCopyKernelConf.wei_n_blk = static_cast<int>(conf.get_wei_N_blk());
    brgCopyKernelConf.wei_tag = conf.get_format();
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

    std::unique_ptr<matmul::jit_brgemm_matmul_copy_b_t> kernel = nullptr;
    OV_CPU_JIT_EMITTER_ASSERT(matmul::create_brgemm_matmul_copy_b(kernel, &brgCopyKernelConf) == dnnl_success,
                              "cannot create kernel due to invalid params");
    return kernel;
}

void BrgemmCopyBKernel::emit_brgemm_copy_b_kernel_call(size_t N, size_t K, size_t offset_in, size_t offset_out, size_t offset_comp) {
    JitSafeInternalCall safe_internal_caller(this, conf.get_isa());

    const auto add_offset = [&](Xbyak::Reg64 reg, size_t bytes_offset) {
        if (bytes_offset) add(reg, bytes_offset);
    };

    // save function address in gpr to pass in call instruction
    const auto &kernel_overload = static_cast<void (*)(matmul::jit_brgemm_matmul_copy_b_t*,
                                                       const void*,
                                                       const void*,
                                                       const void*,
                                                       size_t,
                                                       size_t)>(execute);
    mov(rbp, reinterpret_cast<uintptr_t>(kernel_overload));
    mov(abi_param1, reinterpret_cast<uintptr_t>(kernel.get()));
    add_offset(src_reg, offset_in); // abi_param2
    add_offset(tr_src_reg, offset_out); // abi_param3
    if (conf.is_with_comp()) {
        add_offset(comp_reg, offset_comp); // abi_param4
    } else {
        mov(abi_param4, reinterpret_cast<uintptr_t>(nullptr));
    }

#ifdef _WIN32
    // Note: ABI requires that the remaining parameters (except the first for) are pushed to the stack in right-to-left order
    //  Shadow space will be allocated inside internal_call_rsp_align()
    push(K);
    push(N);
#else
    mov(abi_param5, N);
    mov(abi_param6, K);
#endif

    safe_internal_caller.call(rbp);

#ifdef _WIN32
    static constexpr int gpr_size = 8;
    add(rsp, gpr_size * 2);
#endif
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
        compiled_kernel = std::unique_ptr<BrgemmCopyBKernel>(new BrgemmCopyBKernel(config));
        OV_CPU_JIT_EMITTER_ASSERT(compiled_kernel, "compiled kernel is nullptr");
        compiled_kernel->create_kernel();
    }

    return compiled_kernel;
}

void BrgemmCopyBKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                              const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                              BrgemmCopyBKernelConfig& config) const {
    const auto planar_shape = ov::snippets::utils::get_planar_vdims(expr->get_input_port(0));
    const auto N = *planar_shape.rbegin();
    const auto K = *++planar_shape.rbegin();

    const auto& in_subtensor = ov::snippets::utils::get_projected_subtensor(expr->get_input_port(0));
    const auto N_blk = *in_subtensor.rbegin();
    const auto K_blk = *++in_subtensor.rbegin();
    OV_CPU_JIT_EMITTER_ASSERT(N_blk <= N && K_blk <= K, "BrgemmCopyB has incompatible subtensor dimensions");

    const auto& brg_weight_etype = expr->get_node()->get_input_element_type(0);
    const auto LDB = brgemm_utils::repacking::compute_out_leading_dim(N_blk, brg_weight_etype);
    const auto copy_B_wei_stride = ov::snippets::utils::get_dim_stride(expr->get_input_port(0), config.is_transposed_B() ? 0 : 1) * brg_weight_etype.size();

    config.update(N, N_blk, K, K_blk, copy_B_wei_stride, LDB);
}

void BrgemmCopyBKernelExecutor::execute(const BrgemmCopyBKernelExecutor* executor, BrgemmCopyBKernel::call_args* args) {
    auto kernel = executor->get_kernel();
    OV_CPU_JIT_EMITTER_ASSERT(kernel, "has nullptr kernel");
    OV_CPU_JIT_EMITTER_ASSERT(args, "has nullptr call args");
    (*kernel)(args);
}

}   // namespace intel_cpu
}   // namespace ov
