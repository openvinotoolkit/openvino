// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>

#include "jit_emitter.hpp"

using namespace Xbyak;

namespace ov {
namespace intel_cpu {

using EmitterCode = std::pair<std::shared_ptr<ngraph::snippets::Emitter>, ngraph::snippets::RegInfo>;

#define SNIPPETS_MAX_SNIPPETS_DIMS 7
#define SNIPPETS_MAX_HARNESS_DIMS 5
#define SNIPPETS_MAX_TILE_RANK 2
#define GET_OFF(field) offsetof(jit_snippets_call_args, field)
struct jit_snippets_call_args {
    const void *src_ptrs[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
    void *dst_ptrs[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
};

struct jit_snippets_compile_args {
    int64_t scheduler_dims[SNIPPETS_MAX_TILE_RANK] = {};
    int64_t scheduler_offsets[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
    int64_t data_offsets[SNIPPETS_MAX_SNIPPETS_DIMS * SNIPPETS_MAX_HARNESS_DIMS] = {};
    std::vector<size_t> output_dims = {};
};
///
/// \brief    Kernel is the only entry point to Codogen Jit compilation. Kernel calculates appropriate data offsets,
/// and invokes enclosed outer Tiles. Only 2d Tiles are currently supported, so the emitters should
/// be organized in the following way:
/// KernelEmitter {          /* entry point */
///     TileEmitter {        /* outer tile */
///         TileEmitter {    /* inner vector tile */
///             ...          /* All the necessary Load/Strore/elementwise emitters */
///         }
///         TileEmitter {    /* inner scalar tile for tail processing */
///             ...          /* All the necessary Load/Strore/elementwise emitters */
///         }
///     }
/// }
/// Note that Kernel params are passed directly to the emit_code(). The vector of inputs should contain 2 arguments, the
/// output vector should be empty. Input parameters
///
/// \param      in[0]       The number of the node inputs
/// \param      in[1]      The number of the node outputs
///
// Todo: Scheduler dims and offsets are currently calculated in Subgraph node and passed to the KernelEmitter.
//  However, it seems more natural to calculate all the offsets right in the Kernel op, because the calculation is
//  not device-specific. It is based only on input/output dims (which we already know) and harness num dims
//  (which we should pass from the plugin). It seems also better to wrap the enclosed emitters in tiles in the Kernel op
//  and avoid creating empty tiles.
class KernelEmitter : public jit_emitter {
public:
    KernelEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                  const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override {return 0;}
    void emit_code(const std::vector<size_t> &in,
                   const std::vector<size_t> &out,
                   const std::vector<size_t> &pool,
                   const std::vector<size_t> &gpr) const override;
private:
    void validate_arguments(const std::vector<size_t> &in,
                            const std::vector<size_t> &out,
                            const std::vector<size_t> &pool,
                            const std::vector<size_t> &gpr) const override;
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out,
                   const std::vector<size_t>& pool,
                   const std::vector<size_t>& gpr,
                   const ov::intel_cpu::emitter_context *emit_context) const override;

    std::vector<EmitterCode> body;
};

class TileSchedulerEmitter : public jit_emitter {
public:
    TileSchedulerEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override {return 0;}
    void emit_code(const std::vector<size_t> &in,
                   const std::vector<size_t> &out,
                   const std::vector<size_t> &pool,
                   const std::vector<size_t> &gpr) const override;

private:
    void validate_arguments(const std::vector<size_t> &in,
                            const std::vector<size_t> &out,
                            const std::vector<size_t> &pool,
                            const std::vector<size_t> &gpr) const override;
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out,
                   const std::vector<size_t>& pool,
                   const std::vector<size_t>& gpr,
                   const ov::intel_cpu::emitter_context *emit_context) const override;

    jit_snippets_compile_args jcp;
    EmitterCode vector_tile;
    std::vector<EmitterCode> vector_tile_body;
    EmitterCode scalar_tile;
    std::vector<EmitterCode> scalar_tile_body;
};

///
/// \brief    Tile is designed to organize loop over the input and output data. It is essentially a for(...) loop:
/// it calculates the total number of iterations, performs operations specified by enclosed emitters, advances iteration counters
/// and breaks when necessary.
///
/// \param      in[0]    The number of input entities (or scheduler counts) processed during one iteration of the tile.
/// It is expected to be 1 for outer or scalar tiles and vlen for vector tiles.
/// \param      in[1]    Increment of the previous Tile in current dimension. Must be 0 if this is the first Tile.
/// So previous_inc is zero for outer and vector tiles (the are the first in dim) and vlen for scalar tiles (they usually go after vector Tiles).
/// \param      in[2]    sum number inputs and number of outputs of the node.
/// \param      in[3]    dimension of the tile. Note that only 2d Tile are currently supported, so dim is 0 for outer tiles, 1 for inner tiles.
///
// Todo: Inner and outer tiles have different semantics. For example, outer tile always has the increment == 1, and it can contain only
//  tile emitters (one outer or two inner). So it seems better to create different classes for inner and outer tiles.
// Todo: Currently data pointers incremented after each read/write in Load/Store emitters, so we have to decrement them here
//  if the same data needs to be read twice. Better to move all the pointer increments to TileEmitter and avoid the increments if necessary.
class TileEmitter : public jit_emitter {
public:
    TileEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

    std::vector<EmitterCode> get_body() {return body;}
    size_t get_inputs_num() const override {return 0;}
    void emit_code(const std::vector<size_t> &in,
                   const std::vector<size_t> &out,
                   const std::vector<size_t> &pool,
                   const std::vector<size_t> &gpr) const override;
private:
    void validate_arguments(const std::vector<size_t> &in,
                            const std::vector<size_t> &out,
                            const std::vector<size_t> &pool,
                            const std::vector<size_t> &gpr) const override;
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out,
                   const std::vector<size_t>& pool,
                   const std::vector<size_t>& gpr,
                   const ov::intel_cpu::emitter_context *emit_context) const override;

    std::vector<EmitterCode> body;
};

class NopEmitter : public jit_emitter {
public:
    NopEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n)
    : jit_emitter(h, isa, n) {
    }

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out,
                   const std::vector<size_t>& pool,
                   const std::vector<size_t>& gpr,
                   const ov::intel_cpu::emitter_context *emit_context) const override {
    }
};

class FakeBroadcastEmitter : public jit_emitter {
public:
    FakeBroadcastEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const ov::intel_cpu::emitter_context *emit_context) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;

private:
    bool use_broadcast;
};

class ScalarEmitter : public jit_emitter {
public:
    ScalarEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override {return 0;}

protected:
    size_t aux_gprs_count() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const ov::intel_cpu::emitter_context *emit_context) const override;

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
/// For scalar loads we can use different tiles. Tiling indeed can be arbitrary and post increment should be somehow coded into ISA.
/// Blocked parameter to tell if input is actually blocked. Broadcast means broadcast by W in other cases no need to substitute load.
class MemoryEmitter : public jit_emitter  {
public:
    MemoryEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);
    size_t get_inputs_num() const override {return 1;}

protected:
//    static size_t getEA(const std::shared_ptr<ov::Node>& n);
//    size_t ea;
};

class StoreEmitter : public MemoryEmitter  {
public:
    StoreEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);
    size_t get_inputs_num() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const ov::intel_cpu::emitter_context *emit_context) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
};

class ScalarStoreEmitter : public MemoryEmitter {
public:
    ScalarStoreEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const ov::intel_cpu::emitter_context *emit_context) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
};

class LoadEmitter : public MemoryEmitter {
public:
    LoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const ov::intel_cpu::emitter_context *emit_context) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;

private:
    bool shouldPostIncrement;
};

class BroadcastLoadEmitter : public MemoryEmitter {
public:
    BroadcastLoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);
    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const ov::intel_cpu::emitter_context *emit_context) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
};

class ScalarLoadEmitter : public MemoryEmitter {
public:
    ScalarLoadEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

    size_t get_inputs_num() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out,
              const std::vector<size_t>& pool,
              const std::vector<size_t>& gpr,
              const ov::intel_cpu::emitter_context *emit_context) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;

private:
    bool shouldPostIncrement;
};
}   // namespace intel_cpu
}   // namespace ov
