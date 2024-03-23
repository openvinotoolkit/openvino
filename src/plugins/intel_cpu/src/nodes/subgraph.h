// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

#include "emitters/snippets/x64/jit_kernel_emitter.hpp"
#include "snippets/op/subgraph.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class Subgraph : public Node {
    class SnippetExecutor;
    class SnippetJitExecutor;
    class SnippetJitStaticExecutor;
    class SnippetJitShapeAgnosticExecutor;
    class SnippetJitDynamicSpecializedExecutor;

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

    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;

    IShapeInfer::Result shapeInfer() const override;

    struct SnippetAttrs {
        // Local copy of subgraph node for canonization & code generation
        std::shared_ptr<snippets::op::Subgraph> snippet = nullptr;
        uint64_t bodyHash = 0;
        std::vector<VectorDims> inMemOrders = {};
        std::vector<VectorDims> outMemOrders = {};
        std::vector<ov::element::Type> inMemPrecs = {};
        std::vector<ov::element::Type> outMemPrecs = {};
    };

private:
    static uint64_t get_body_hash(const std::shared_ptr<snippets::op::Subgraph>& snippet);

    void init_memory_ptrs();
    void init_attrs();
    void init_start_offsets();
    void init_snippets_blocked_shapes(snippets::op::Subgraph::BlockedShapeVector& in_blocked_shapes) const;
    void init_precisions(std::vector<ov::element::Type>& input_types, std::vector<ov::element::Type>& output_types) const;
    void init_blocked_broadcasting_mask(uint8_t& mask) const;
    void lower();
    IShapeInfer::Result blocked_shape_infer() const;

    std::vector<ov::snippets::pass::Manager::PositionedPassBase> get_data_flow_passes() const;
    std::vector<ov::snippets::lowered::pass::PassPipeline::PositionedPassLowered> get_control_flow_passes() const;

    uint8_t get_broadcasting_mask(const std::vector<VectorDims>& input_shapes) const;

    std::shared_ptr<SnippetAttrs> snippetAttrs;
    dnnl::impl::cpu::x64::cpu_isa_t host_isa;

    size_t input_num = 0;
    size_t output_num = 0;

    std::vector<MemoryPtr> srcMemPtrs = {};
    std::vector<MemoryPtr> dstMemPtrs = {};

    std::vector<ptrdiff_t> start_offset_in = {};
    std::vector<ptrdiff_t> start_offset_out = {};

    // Input shapes that are used in ShapeInfer, PrepareParams to avoid frequent memory allocation
    mutable std::vector<VectorDims> in_blocked_shapes;
    std::vector<bool> is_blocked_out_layout;

    bool is_dynamic = false;

    size_t tensor_rank = 0;
    static const size_t rank6D {6};

    mutable std::shared_ptr<SnippetExecutor> execPtr = nullptr;
};

class Subgraph::SnippetExecutor {
public:
    SnippetExecutor() = default;
    virtual ~SnippetExecutor() = default;

    virtual void exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) = 0;
};

// Base class for Jit Executors
class Subgraph::SnippetJitExecutor : public Subgraph::SnippetExecutor {
public:
    SnippetJitExecutor(const std::shared_ptr<SnippetAttrs>& snippet_attrs, size_t tensor_rank);
    virtual ~SnippetJitExecutor() = default;

    void exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;

protected:
    // Evaluates generated snippet using parallel backend
    virtual void schedule_6d(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) = 0;
    virtual void schedule_nt(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) = 0;

    void init_runtime_params(const std::shared_ptr<SnippetAttrs>& snippet_attrs);
    void generate(const std::shared_ptr<SnippetAttrs>& snippet_attrs);

    std::shared_ptr<snippets::Schedule> schedule;
    // Holds index of output used as in execution domain
    // it should be compatible with a schedule's work size
    std::vector<size_t> parallel_exec_domain = {};
    size_t harness_work_amount = 0;
    size_t tensor_rank = 0;

    // Buffer scratchpad
    std::vector<uint8_t> buffer_scratchpad = {};
    size_t buffer_scratchpad_size = 0;

#ifdef SNIPPETS_DEBUG_CAPS
    bool enabled_segfault_detector = false;
    inline void segfault_detector();
#endif
};

// Class for Subgraphs with static shapes
class Subgraph::SnippetJitStaticExecutor : public Subgraph::SnippetJitExecutor {
public:
    SnippetJitStaticExecutor(const std::shared_ptr<SnippetAttrs>& snippet_attrs,
                             const std::vector<ptrdiff_t>& start_offset_in,
                             const std::vector<ptrdiff_t>& start_offset_out,
                             size_t tensor_rank);

protected:
    typedef void (*kernel)(const void*, const void*);

    void schedule_6d(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;
    void schedule_nt(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;

    inline void update_ptrs(jit_snippets_call_args& call_args, const std::vector<MemoryPtr>& srcMemPtrs, const std::vector<MemoryPtr>& dstMemPtrs);

    std::vector<ptrdiff_t> start_offset_in = {};
    std::vector<ptrdiff_t> start_offset_out = {};
};

// Class for dynamic Subgraph with shape-agnostic that just generate lowered code
class Subgraph::SnippetJitShapeAgnosticExecutor : public Subgraph::SnippetJitExecutor {
    friend class Subgraph::SnippetJitDynamicSpecializedExecutor;
public:
    SnippetJitShapeAgnosticExecutor(const std::shared_ptr<SnippetAttrs>& snippet_attrs, size_t tensor_rank);

protected:
    void schedule_6d(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;
    void schedule_nt(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;
};

// Specialized dynamic executor based on shape agnostic kernel for the specific input shapes
class Subgraph::SnippetJitDynamicSpecializedExecutor : public Subgraph::SnippetJitExecutor {
public:
    SnippetJitDynamicSpecializedExecutor(const std::shared_ptr<SnippetAttrs>& snippet_attrs,
                                         const std::vector<ptrdiff_t>& start_offset_in,
                                         const std::vector<ptrdiff_t>& start_offset_out,
                                         const std::shared_ptr<SnippetJitShapeAgnosticExecutor>& agnostic,
                                         size_t tensor_rank);

protected:
    typedef void (*dynamic_kernel)(const void *);

    inline void init_original_ptrs(const std::vector<MemoryPtr>& srcMemPtrs, const std::vector<MemoryPtr>& dstMemPtrs,
                                   std::vector<const uint8_t*>& src_ptrs, std::vector<uint8_t*>& dst_ptrs);
    inline void init_call_args(jit_snippets_call_args& call_args);
    inline void update_ptrs(jit_snippets_call_args& call_args, const std::vector<const uint8_t*>& src_ptrs,
                            const std::vector<uint8_t*>& dst_ptrs, const size_t* indexes) const;
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
