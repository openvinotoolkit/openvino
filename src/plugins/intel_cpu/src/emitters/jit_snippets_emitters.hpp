// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/rt_info.hpp>
#include <ie_ngraph_utils.hpp>

#include "jit_emitter.hpp"
#include "jit_load_store_emitters.hpp"

#include "snippets_transformations/op/store_convert.hpp"
// Matmul support:
#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/matmul/brgemm_matmul_copy_utils.hpp>
#include <cpu/x64/matmul/brgemm_matmul_utils.hpp>
#include <cpu/x64/amx_tile_configure.hpp>

using namespace Xbyak;
using ngraph::snippets::AllocatedEmitter;

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
    std::vector<size_t> master_shape{};
    size_t tile_rank = 0;
};
///
/// \brief jit_container_emitter designed to wrap Emitters that contain other Emitters (for example, KernelEmitter)
///  This is needed to provide common interface for register mapping
/// (abstract to physical) and nested code access.
///
class jit_container_emitter: public jit_emitter {
public:
    jit_container_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                          const std::shared_ptr<ov::Node>& n);
    // mapping info contains abstract_to_physical map + regs_pool
    using mapping_info = std::pair<std::map<size_t, size_t>, std::vector<size_t>&>;
protected:
    // maps gpr and vec abstract registers to physical ones. Physical reg indexes are taken from the provided pools
    // (the first 2 args). All the used gpr and vec registers are also stored in the provided sets (the second 2 args).
    void map_abstract_registers(mapping_info& gpr_map_pool,  mapping_info& vec_map_pool,
                                std::vector<AllocatedEmitter>& allocated_emitters) const;
    std::vector<AllocatedEmitter> body;
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
    KernelEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                  const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override {return 0;}
    void emit_code(const std::vector<size_t> &in,
                   const std::vector<size_t> &out) const;

private:
    void validate_arguments(const std::vector<size_t> &in,
                            const std::vector<size_t> &out) const;
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;
    void init_data_pointers(size_t, size_t, bool, const Reg64&, const Reg64&, const std::vector<Reg64>&) const;

    jit_snippets_compile_args jcp;
    std::vector<size_t> gp_regs_pool;
    size_t num_inputs;
    size_t num_outputs;
    bool is_buffer_needed;
    // Vector of indices (lenght = input tensor rank) per every input and output that describes in which order
    // corresponding tensor dimensions are accessed (default: consecutive dense, e.g. 0,1,2,3 for 4D tensor).
    // Needed to calc i/o offsets.
    std::vector<std::vector<size_t>> data_layout;
    std::vector<std::vector<size_t>> io_shapes = {};
    std::vector<size_t> io_data_size {};

    // gpr's used to store data pointers, track them to apply offsets in Kernel
    std::vector<size_t> data_ptr_regs_idx;
    std::vector<size_t> vec_regs_pool;
    const size_t reg_indexes_idx = abi_param1.getIdx();
    const size_t reg_const_params_idx = abi_param2.getIdx();
};

class LoopBeginEmitter : public jit_emitter {
public:
    LoopBeginEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);
    void emit_code(const std::vector<size_t> &in,
                   const std::vector<size_t> &out) const;
    // todo: it is purely virtual in the base class, but do we need it?
    size_t get_inputs_num() const override {return 0;}

private:
    void validate_arguments(const std::vector<size_t> &in,
                            const std::vector<size_t> &out) const;
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    std::shared_ptr<ngraph::snippets::op::LoopBegin> loop_begin;
    size_t num_inputs = 0;
    bool evaluate_once = false;
    size_t work_amount = 0; // need to store work_amount explicitly, since two loops can work on the same dim (e.g. vector + scalar)
};

class LoopEndEmitter : public jit_emitter {
public:
    LoopEndEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);
    void emit_code(const std::vector<size_t> &in,
                   const std::vector<size_t> &out) const;
    // todo: it is purely virtual in the base class, but do we need it?
    size_t get_inputs_num() const override {return 0;}

private:
    void validate_arguments(const std::vector<size_t> &in,
                            const std::vector<size_t> &out) const;

    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    std::shared_ptr<ngraph::snippets::op::LoopBegin> loop_begin;
    std::shared_ptr<ngraph::snippets::op::LoopEnd> loop_end;

    size_t num_inputs = 0;
    size_t num_outputs = 0;
    // keep data_size int64_t to avoid conversion to size_t (and overflow) when multiplied by negative increments or offsets
    std::vector<int64_t> io_data_size {};
    size_t wa_increment = 0;
    size_t work_amount = 0;
    bool evaluate_once = false;
    std::vector<int64_t> ptr_increments;
    std::vector<int64_t> finalization_offsets;
};


class NopEmitter : public jit_emitter {
public:
    NopEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n)
    : jit_emitter(h, isa, n) {
    }

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override {
    }
};

class BroadcastMoveEmitter : public jit_emitter {
public:
    BroadcastMoveEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

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
    ScalarEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

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
    MemoryEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

protected:
    Precision src_prc;
    Precision dst_prc;

    size_t count = 0;
    size_t byte_offset = 0;
};

class StoreEmitter : public MemoryEmitter  {
public:
    StoreEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

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
    LoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

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
    BroadcastLoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
};

class LoadConvertEmitter : public MemoryEmitter {
public:
    LoadConvertEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

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
    StoreConvertEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

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
    BrgemmEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override {return 2;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
    std::vector<size_t> io_data_size {};
    struct brgemmCtx {
        size_t M, N, K, LDA, LDB, LDC;
        dnnl_data_type_t dt_in0, dt_in1;
        char palette[64];
        bool is_with_amx;
        bool is_with_comp;
        float beta;
    };
    void initBrgemm(brgemmCtx& ctx, std::unique_ptr<brgemm_kernel_t>& brgKernel, bool use_amx) const;
    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void callBrgemm(brgemmCtx& ctx, std::unique_ptr<brgemm_kernel_t>& brgKernel, const void* pin0, const void* pin1, void* pout, void* wsp) const;
    size_t getBrgIdx(size_t mIdx, size_t kIdx, size_t nIdx) const;
    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_brgemm_kernel_call(const brgemm_kernel_t *brg_kernel, int bs,
                                 Reg64 addr_A, Reg64 addr_B,
                                 const brgemm_batch_element_t *batch, Reg64 addr_C, void *scratch,
                                 const size_t in0_kernel_offset, const size_t in1_kernel_offset, const size_t out0_kernel_offset) const;
    static void kernel_execute(const brgemm_kernel_t *brg_kernel, const void *A, const void *B, void *C);
    static constexpr size_t BRGEMM_KERNELS_NUM = 8;
    static constexpr size_t matmulOptimalM = 32;
    brgemmCtx brgCtxs0[BRGEMM_KERNELS_NUM];
    std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgKernels0[BRGEMM_KERNELS_NUM];

    size_t M, M_blk, M_tail;
    size_t K, K_blk, K_tail;
    size_t N, N_blk, N_tail;
    size_t brg0VnniFactor;

    size_t load_offset_a = 0lu;
    size_t load_offset_b = 0lu;
    size_t store_offset_c = 0lu;
};

class HorizonMaxEmitter : public jit_emitter {
public:
    HorizonMaxEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override {return 1;}

protected:
    size_t aux_gprs_count() const override {return 1;}
    size_t aux_vecs_count() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
};

class HorizonSumEmitter : public jit_emitter {
public:
    HorizonSumEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override {return 1;}

protected:
    size_t aux_gprs_count() const override {return 1;}
    size_t aux_vecs_count() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
};

class VectorBufferEmitter : public jit_emitter {
public:
    VectorBufferEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
};

class FillEmitter : public jit_emitter {
public:
    FillEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override {return 1;}

protected:
    size_t aux_gprs_count() const override;

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;

    void register_table_entries() override;

    size_t offset = 0;
    uint32_t fill_value = 0x0;
};

}   // namespace intel_cpu
}   // namespace ov
