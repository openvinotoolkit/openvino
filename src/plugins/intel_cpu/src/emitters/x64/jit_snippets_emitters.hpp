// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/rt_info.hpp>
#include <ie_ngraph_utils.hpp>

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/expression.hpp"

#include "jit_emitter.hpp"
#include "jit_load_store_emitters.hpp"

#include "transformations/snippets/x64/op/store_convert.hpp"
// Matmul support:
#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/matmul/brgemm_matmul_copy_utils.hpp>
#include <cpu/x64/matmul/brgemm_matmul_utils.hpp>
#include <cpu/x64/amx_tile_configure.hpp>

namespace ov {
namespace intel_cpu {

#define SNIPPETS_MAX_SNIPPETS_DIMS 12
#define SNIPPETS_MAX_HARNESS_DIMS 5
#define SNIPPETS_MAX_TILE_RANK 2
#define SNIPPETS_DYNAMIC_MASTER_SHAPE_RANK 6
#define GET_OFF(field) offsetof(jit_snippets_call_args, field)
struct jit_snippets_call_args {
    const void *src_ptrs[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
    void *dst_ptrs[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
    void *buffer_scratchpad_ptr = nullptr;
};

struct jit_snippets_compile_args {
    size_t parallel_executor_ndims = 1;
};
///
/// \brief jit_container_emitter designed to wrap Emitters that contain other Emitters (for example, KernelEmitter)
///  This is needed to provide common interface for register mapping
/// (abstract to physical) and nested code access.
///
class jit_container_emitter: public jit_emitter {
public:
    jit_container_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                          dnnl::impl::cpu::x64::cpu_isa_t isa,
                          const ov::snippets::lowered::ExpressionPtr& expr);
    // mapping info contains abstract_to_physical map + regs_pool
    using mapping_info = std::pair<std::map<size_t, size_t>, std::vector<size_t>&>;
protected:
    // maps gpr and vec abstract registers to physical ones. Physical reg indexes are taken from the provided pools
    // (the first 2 args). All the used gpr and vec registers are also stored in the provided sets (the second 2 args).
    void map_abstract_registers(mapping_info& gpr_map_pool,  mapping_info& vec_map_pool,
                                snippets::lowered::LinearIR::container& expressions) const;
    snippets::lowered::LinearIR body;
};
///
/// \brief    Kernel is the only entry point to Codogen Jit compilation. Kernel perform abstract-to-physical register
/// mapping and creates a pools of available gpr and vec registers. Kernel usually contains (at least one)
/// LoopBeginEmitter and LoopEndEmitter pair. In general the enclosed emitters should be organized in the following way:
/// KernelEmitter {                 /* entry point, maps registers, creates pools of available registers */
///     1.S LoopBeginEmitter        /* Scalar Loop over the outer dimension [START] */
///         2.S LoopBeginEmitter    /* inner vector loop [START] */
///             ...                 /* All the necessary Load/Strore/elementwise emitters */
///         2.E LoopEndEmitter      /* inner vector loop [END] */
///         3.S LoopBeginEmitter    /* inner scalar loop for tail processing [START]*/
///             ...                 /* All the necessary Load/Strore/elementwise emitters */
///         3.E LoopEndEmitter      /* inner scalar loop for tail processing [END]*/
///     1.E LoopEndEmitter          /* Scalar Loop over the outer dimension [END] */
/// }
/// Note that Kernel doesn't accept any input arguments.
///

class KernelEmitter : public jit_container_emitter {
public:
    KernelEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                  dnnl::impl::cpu::x64::cpu_isa_t isa,
                  const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 0;}
    void emit_code(const std::vector<size_t> &in,
                   const std::vector<size_t> &out) const;

private:
    using jit_emitter::emit_code;
    void validate_arguments(const std::vector<size_t> &in,
                            const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;
    void init_data_pointers(const Xbyak::Reg64&, const Xbyak::Reg64&, const std::vector<Xbyak::Reg64>&) const;

    jit_snippets_compile_args jcp;
    std::vector<size_t> gp_regs_pool;
    std::vector<size_t> master_shape;
    size_t num_inputs;
    size_t num_outputs;
    size_t num_unique_buffers;
    // Vector of indices (lenght = input tensor rank) per every input and output that describes in which order
    // corresponding tensor dimensions are accessed (default: consecutive dense, e.g. 0,1,2,3 for 4D tensor).
    // Needed to calc i/o offsets.
    std::vector<std::vector<size_t>> io_data_layouts;
    std::vector<std::vector<size_t>> io_shapes = {};
    std::vector<size_t> io_data_sizes {};

    // gpr's used to store data pointers, track them to apply offsets in Kernel
    std::vector<size_t> data_ptr_regs_idx;
    std::vector<size_t> vec_regs_pool;

    const size_t reg_indexes_idx;
    const size_t reg_const_params_idx;
};

class LoopBeginEmitter : public jit_emitter {
public:
    LoopBeginEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                     dnnl::impl::cpu::x64::cpu_isa_t isa,
                     const ov::snippets::lowered::ExpressionPtr& expr);
    void emit_code(const std::vector<size_t> &in,
                   const std::vector<size_t> &out) const;
    // todo: it is purely virtual in the base class, but do we need it?
    size_t get_inputs_num() const override {return 0;}

private:
    using jit_emitter::emit_code;
    void validate_arguments(const std::vector<size_t> &in,
                            const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    std::shared_ptr<snippets::op::LoopBegin> loop_begin;
    bool evaluate_once = false;
    size_t work_amount = 0; // need to store work_amount explicitly, since two loops can work on the same dim (e.g. vector + scalar)
};

class LoopEndEmitter : public jit_emitter {
public:
    LoopEndEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                   dnnl::impl::cpu::x64::cpu_isa_t isa,
                   const ov::snippets::lowered::ExpressionPtr& expr);
    void emit_code(const std::vector<size_t> &in,
                   const std::vector<size_t> &out) const;
    // todo: it is purely virtual in the base class, but do we need it?
    size_t get_inputs_num() const override {return 0;}

private:
    using jit_emitter::emit_code;
    void validate_arguments(const std::vector<size_t> &in,
                            const std::vector<size_t> &out) const override;

    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    std::shared_ptr<snippets::op::LoopBegin> loop_begin;
    std::shared_ptr<snippets::op::LoopEnd> loop_end;

    size_t num_inputs = 0;
    size_t num_outputs = 0;
    // keep data_size int64_t to avoid conversion to size_t (and overflow) when multiplied by negative increments or offsets
    std::vector<int64_t> io_data_size {};
    int64_t wa_increment = 0;
    int64_t work_amount = 0;
    bool evaluate_once = false;
    std::vector<int64_t> ptr_increments;
    std::vector<int64_t> finalization_offsets;
};

class NopEmitter : public jit_emitter {
public:
    NopEmitter(dnnl::impl::cpu::x64::jit_generator* h,
               dnnl::impl::cpu::x64::cpu_isa_t isa,
               const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override {
    }
};

class ParameterEmitter : public NopEmitter {
public:
    ParameterEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                     dnnl::impl::cpu::x64::cpu_isa_t isa,
                     const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override { return 0; }
};

class ResultEmitter : public NopEmitter {
public:
    ResultEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                  dnnl::impl::cpu::x64::cpu_isa_t isa,
                  const ov::snippets::lowered::ExpressionPtr& expr);
    size_t get_inputs_num() const override {return 1;}
};

class BroadcastMoveEmitter : public jit_emitter {
public:
    BroadcastMoveEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                         dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;

private:
    size_t byte_size = 0lu;
};

class ScalarEmitter : public jit_emitter {
public:
    ScalarEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                  dnnl::impl::cpu::x64::cpu_isa_t isa,
                  const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 0;}

protected:
    size_t aux_gprs_count() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;

private:
    int32_t value;
};

///
/// Memory emitters:
///
/// *Note*: post increment is embedded into Load/Store operation which means that
/// it's illigal to load/store to the same address multiple times
/// Typical application can be if Load and BroadcastLoad are performed from the same pointer.
/// If Load goes before BroadcastLoad topologicaly the resilt will be incorrect
/// For scalar loads we can use different loops. Tiling indeed can be arbitrary and post increment should be somehow coded into ISA.
/// Blocked parameter to tell if input is actually blocked. Broadcast means broadcast by W in other cases no need to substitute load.
class MemoryEmitter : public jit_emitter  {
public:
    MemoryEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                  dnnl::impl::cpu::x64::cpu_isa_t isa,
                  const ov::snippets::lowered::ExpressionPtr& expr);

protected:
    InferenceEngine::Precision src_prc;
    InferenceEngine::Precision dst_prc;

    size_t count = 0;
    size_t byte_offset = 0;
};

class StoreEmitter : public MemoryEmitter  {
public:
    StoreEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                 dnnl::impl::cpu::x64::cpu_isa_t isa,
                 const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
    void emit_data() const override;

private:
    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;
};

class LoadEmitter : public MemoryEmitter {
public:
    LoadEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                dnnl::impl::cpu::x64::cpu_isa_t isa,
                const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
    void emit_data() const override;

private:
    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
};

class BroadcastLoadEmitter : public MemoryEmitter {
public:
    BroadcastLoadEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                         dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
};

class LoadConvertEmitter : public MemoryEmitter {
public:
    LoadConvertEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                       dnnl::impl::cpu::x64::cpu_isa_t isa,
                       const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
    void emit_data() const override;

private:
    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
};

class StoreConvertEmitter : public MemoryEmitter {
public:
    StoreConvertEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                        dnnl::impl::cpu::x64::cpu_isa_t isa,
                        const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
    void emit_data() const override;

private:
    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;
};

class BrgemmEmitter : public jit_emitter {
public:
    BrgemmEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                  dnnl::impl::cpu::x64::cpu_isa_t isa,
                  const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override { return m_with_scratch ? 3 : 2; }
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);
    size_t aux_gprs_count() const override;

    static size_t get_in_leading_dim(const VectorDims& shape, const std::vector<size_t>& layout);
    static size_t get_out_leading_dim(const VectorDims& shape, const std::vector<size_t>& layout);

private:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;
    struct brgemmCtx {
        brgemmCtx() : M(0), N(0), K(0),
                    LDA(0), LDB(0), LDC(0),
                    dt_in0(dnnl_f32), dt_in1(dnnl_f32),
                    is_with_amx(false), is_with_comp(false), beta(0) {}
        size_t M, N, K, LDA, LDB, LDC;
        dnnl_data_type_t dt_in0, dt_in1;
        char palette[64] = {};
        bool is_with_amx;
        bool is_with_comp;
        float beta;
    };
    static void initBrgemm(brgemmCtx& ctx, std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& brgKernel, bool use_amx);
    static size_t getBrgIdx(size_t kIdx, size_t nIdx);

    void emit_brgemm_kernel_call(const dnnl::impl::cpu::x64::brgemm_kernel_t* brg_kernel, const brgemmCtx& ctx,
                                 Xbyak::Reg64 addr_A, Xbyak::Reg64 addr_B, Xbyak::Reg64 scratch, Xbyak::Reg64 addr_C,
                                 size_t in0_kernel_offset = 0, size_t in1_kernel_offset = 0,
                                 size_t in2_kernel_offset = 0, size_t out0_kernel_offset = 0) const;
    static void kernel_execute(const dnnl::impl::cpu::x64::brgemm_kernel_t *brg_kernel, const void *A, const void *B, void *C, void *scratch, int with_comp);
    void emit_N_blocking_loops(size_t k_kernel_id,
                               const Xbyak::Reg64& input_0, const Xbyak::Reg64& input_1,
                               const Xbyak::Reg64& input_2, const Xbyak::Reg64& output_0,
                               const Xbyak::Reg64& work_amount_N) const;

    // Note: K dimension is covered by TWO blocked kernels (with beta = 0 and 1) + 1 for tail
    static constexpr size_t BRGEMM_K_KERNEL_NUM = 3;
    static constexpr size_t BRGEMM_N_KERNEL_NUM = 2;
    std::array<brgemmCtx, BRGEMM_K_KERNEL_NUM * BRGEMM_N_KERNEL_NUM> m_brgCtxs;
    std::array<std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>, BRGEMM_K_KERNEL_NUM * BRGEMM_N_KERNEL_NUM> m_brgKernels;

    size_t m_M;
    size_t m_K, m_K_blk, m_K_tail;
    size_t m_N, m_N_blk, m_N_tail;
    size_t m_brg0VnniFactor;
    bool m_N_blk_loop = false;
    bool m_K_blk_loop = false;

    bool m_with_scratch = false;
    bool m_with_comp = false;

    size_t m_load_offset_a = 0lu;
    size_t m_load_offset_b = 0lu;
    size_t m_load_offset_scratch = 0lu;
    size_t m_store_offset_c = 0lu;

    std::vector<size_t> io_data_size {};
};

class BrgemmCopyBEmitter : public jit_emitter {
public:
    BrgemmCopyBEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                       dnnl::impl::cpu::x64::cpu_isa_t isa,
                       const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 1;}
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr) {
        return {{element::i8}, {element::bf16}};
    }

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    void init_brgemm_copy(std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t>& kernel,
                          size_t N, size_t N_blk, size_t N_tail, size_t LDB, size_t K,
                          bool is_with_amx, dnnl_data_type_t dt_in0, dnnl_data_type_t dt_in1) const;
    void emit_kernel_call(const dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t* kernel,
                          Xbyak::Reg64 src, Xbyak::Reg64 dst, Xbyak::Reg64 comp, size_t N, size_t K,
                          size_t offset_in, size_t offset_out, size_t offset_comp) const;

    static void execute(dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t* kernel,
                        const void* src, const void* dst, const void* comp, size_t N, size_t K);

    std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t> m_kernel;

    ov::element::Type m_brgemm_prc_in0, m_brgemm_prc_in1;
    size_t m_N, m_N_blk, m_N_tail;
    size_t m_K, m_K_blk, m_K_tail;
    size_t m_LDB;
    size_t m_brgemmVNNIFactor;
    bool m_with_comp = false;

    size_t m_in_offset = 0lu;
    size_t m_out_offset = 0lu;
    size_t m_comp_offset = 0lu;
};

class HorizonEmitter : public jit_emitter {
public:
    HorizonEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                   dnnl::impl::cpu::x64::cpu_isa_t isa,
                   const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 1;}
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr) {
        return {{element::f32}};
    }

protected:
    size_t aux_vecs_count() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;

    template<typename Vmm>
    void perform_op(const Vmm &vmm1, const Vmm &vmm2, const Vmm &vmm3) const;

    enum class OpType { max, sum };
    OpType m_op_type = OpType::max;
};
class FillEmitter : public jit_emitter {
public:
    FillEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                dnnl::impl::cpu::x64::cpu_isa_t isa,
                const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 1;}

protected:
    size_t aux_gprs_count() const override;

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
    template <typename Vmm>
    void fill_full(const Vmm& vmm_dst) const;
    template <typename Vmm>
    void fill_tail(const Vmm& vmm_src, const Vmm& vmm_dst) const;

    bool is_full_reg() const { return offset == 0; }
    bool is_optimized() const { return is_full_reg() && fill_value == uint32_t(0x0); }

    size_t offset = 0;
    uint32_t fill_value = 0x0;
};

}   // namespace intel_cpu
}   // namespace ov
