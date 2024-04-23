// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

#include "emitters/snippets/x64/cpu_generator.hpp"
#include "snippets/op/subgraph.hpp"

#include <array>

namespace ov {
namespace intel_cpu {
namespace node {

class Subgraph : public Node {
    class SubgraphExecutor;
    class SubgraphJitExecutor;
    class SubgraphJitStaticExecutor;
    class SubgraphJitShapeAgnosticExecutor;
    class SubgraphJitDynamicSpecializedExecutor;
public:
    Subgraph(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);
    ~Subgraph() override = default;

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void selectOptimalPrimitiveDescriptor() override;
    ov::element::Type getRuntimePrecision() const override;

    void createPrimitive() override;
    void prepareParams() override;

    bool canBeInPlace() const override;
    bool created() const override;

    // if generator is set, it would execute generated code otherwise it would fallback to nGraph reference
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;

    struct SubgraphAttrs {
        // Local copy of subgraph node for canonization & code generation
        std::shared_ptr<snippets::op::Subgraph> snippet;
        uint64_t bodyHash;
        std::vector<VectorDims> inMemOrders;
        std::vector<VectorDims> outMemOrders;
        std::vector<ov::element::Type> inMemPrecs;
        std::vector<ov::element::Type> outMemPrecs;
    };

private:
    void init_memory_ptrs();
    void init_attrs();
    void init_start_offsets();
    void init_snippets_blocked_shapes(snippets::op::Subgraph::BlockedShapeVector& in_blocked_shapes) const;
    void init_precisions(std::vector<ov::element::Type>& input_types, std::vector<ov::element::Type>& output_types) const;
    void lower();

    bool need_blocked_shape_infer() const;

    static uint64_t get_body_hash(const std::shared_ptr<snippets::op::Subgraph>& snippet);

    uint8_t get_broadcasting_mask(const std::vector<VectorDims>& input_shapes) const;

    using DataFlowPasses = std::vector<ov::snippets::pass::Manager::PositionedPassBase>;
    using ControlFlowPasses = std::vector<ov::snippets::lowered::pass::PassPipeline::PositionedPassLowered>;
    using ControlFlowConfig = std::shared_ptr<ov::snippets::lowered::pass::PassConfig>;

    DataFlowPasses get_data_flow_passes() const;
    std::pair<ControlFlowConfig, ControlFlowPasses> get_control_flow_passes() const;

    // Holds ISA version used is codeGeneration target
    dnnl::impl::cpu::x64::cpu_isa_t host_isa;
    std::shared_ptr<SubgraphAttrs> snippetAttrs;

    size_t input_num = 0;
    size_t output_num = 0;

    std::vector<MemoryPtr> srcMemPtrs = {};
    std::vector<MemoryPtr> dstMemPtrs = {};

    std::vector<ptrdiff_t> start_offset_in = {};
    std::vector<ptrdiff_t> start_offset_out = {};

    bool is_dynamic = false;
    // Input shapes that are used in PrepareParams to avoid frequent memory allocation
    std::vector<VectorDims> in_shapes;

    std::shared_ptr<SubgraphExecutor> execPtr = nullptr;
};

// Base class for all Executors
class Subgraph::SubgraphExecutor {
public:
    SubgraphExecutor() = default;
    virtual ~SubgraphExecutor() = default;

    virtual void exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) = 0;
};

// Base class for Jit Executors
class Subgraph::SubgraphJitExecutor : public Subgraph::SubgraphExecutor {
public:
    SubgraphJitExecutor(const std::shared_ptr<SubgraphAttrs>& snippet_attrs);
    virtual ~SubgraphJitExecutor() = default;

    void exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;

protected:
    // Evaluates generated snippet using parallel backend
    virtual void schedule_6d(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) = 0;
    virtual void schedule_nt(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) = 0;

    virtual void init_runtime_params(const std::shared_ptr<CPURuntimeConfig>& cpu_config);
    void generate(const std::shared_ptr<SubgraphAttrs>& snippet_attrs, const std::shared_ptr<CPURuntimeConfig>& cpu_config);

    std::shared_ptr<snippets::Schedule> schedule;
    // Holds index of output used as in execution domain
    // it should be compatible with a schedule's work size
    std::vector<size_t> parallel_exec_domain = {};
    size_t harness_work_amount = 0;

    // Buffer scratchpad
    std::vector<uint8_t> buffer_scratchpad = {};
    size_t buffer_scratchpad_size = 0;

    const size_t rank6D = 6;

#ifdef SNIPPETS_DEBUG_CAPS
    bool enabled_segfault_detector = false;
    inline void segfault_detector();
#endif
};

// Class for Subgraphs with static shapes
class Subgraph::SubgraphJitStaticExecutor : public Subgraph::SubgraphJitExecutor {
public:
    SubgraphJitStaticExecutor(const std::shared_ptr<SubgraphAttrs>& snippet_attrs,
                              const std::vector<ptrdiff_t>& start_offset_in,
                              const std::vector<ptrdiff_t>& start_offset_out);

protected:
    typedef void (*kernel)(const void*, const void*);

    void schedule_6d(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;
    void schedule_nt(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;

    inline void update_ptrs(jit_snippets_call_args& call_args, const std::vector<MemoryPtr>& srcMemPtrs, const std::vector<MemoryPtr>& dstMemPtrs);

    std::vector<ptrdiff_t> start_offset_in = {};
    std::vector<ptrdiff_t> start_offset_out = {};
};

// Class for dynamic Subgraph with shape-agnostic that just generate lowered code
class Subgraph::SubgraphJitShapeAgnosticExecutor : public Subgraph::SubgraphJitExecutor {
    friend class Subgraph::SubgraphJitDynamicSpecializedExecutor;
public:
    SubgraphJitShapeAgnosticExecutor(const std::shared_ptr<SubgraphAttrs>& snippet_attrs);

protected:
    void schedule_6d(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;
    void schedule_nt(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;
};

// Specialized dynamic executor based on shape agnostic kernel for the specific input shapes
class Subgraph::SubgraphJitDynamicSpecializedExecutor : public Subgraph::SubgraphJitExecutor {
public:
    SubgraphJitDynamicSpecializedExecutor(const std::shared_ptr<SubgraphAttrs>& snippet_attrs,
                                         const std::vector<ptrdiff_t>& start_offset_in,
                                         const std::vector<ptrdiff_t>& start_offset_out,
                                         const std::shared_ptr<SubgraphJitShapeAgnosticExecutor>& agnostic);

protected:
    typedef void (*dynamic_kernel)(const void *);

    inline void init_original_ptrs(const std::vector<MemoryPtr>& srcMemPtrs, const std::vector<MemoryPtr>& dstMemPtrs,
                                   std::vector<const uint8_t*>& src_ptrs, std::vector<uint8_t*>& dst_ptrs);
    inline void init_call_args(jit_snippets_call_args& call_args);
    inline void update_ptrs(jit_snippets_call_args& call_args, const std::vector<const uint8_t*>& src_ptrs,
                            const std::vector<uint8_t*>& dst_ptrs, const size_t* indexes) const;
    void init_runtime_params(const std::shared_ptr<CPURuntimeConfig>& cpu_config) override;
    // Evaluates generated snippet using parallel backend
    void schedule_6d(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;
    void schedule_nt(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;

    std::vector<std::vector<size_t>> data_offsets = {};
    std::vector<jit_snippets_call_args::loop_args_t> loop_args = {};

    std::vector<ptrdiff_t> start_offset_in = {};
    std::vector<ptrdiff_t> start_offset_out = {};
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
