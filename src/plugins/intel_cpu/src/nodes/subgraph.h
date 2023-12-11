// Copyright (C) 2020-2023 Intel Corporation
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

public:
    Subgraph(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);
    ~Subgraph() override = default;

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void selectOptimalPrimitiveDescriptor() override;
    ov::element::Type getRuntimePrecision() const override;

    void createPrimitive() override;
    void prepareParams() override;
    bool needPrepareParams() const override;

    bool canBeInPlace() const override;
    bool created() const override;

    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;

    struct SnippetAttrs {
        // Local copy of subgraph node for canonization & code generation
        std::shared_ptr<snippets::op::Subgraph> snippet;
        uint64_t bodyHash;
        std::vector<VectorDims> inMemOrders;
        std::vector<VectorDims> outMemOrders;
        std::vector<ov::element::Type> inMemPrecs;
        std::vector<ov::element::Type> outMemPrecs;
        // todo: used flag if we need extra shape infer, can be removed after [121670]
        bool has_non_planar_inputs;
        // broadcasting mask
        uint8_t broadcasting_mask = UINT8_MAX;
    };

private:
    static uint64_t get_body_hash(const std::shared_ptr<snippets::op::Subgraph>& snippet);

    void init_memory_ptrs();
    void init_attrs();
    void init_start_offsets();
    void init_snippets_blocked_shapes(snippets::op::Subgraph::BlockedShapeVector& in_blocked_shapes);
    void init_precisions(std::vector<ov::element::Type>& input_types, std::vector<ov::element::Type>& output_types);
    void lower();
    uint8_t get_blocked_broadcasting_mask();

    std::vector<ov::snippets::pass::Manager::PositionedPassBase> get_data_flow_passes() const;
    std::vector<ov::snippets::lowered::pass::PassPipeline::PositionedPassLowered> get_control_flow_passes() const;

    SnippetAttrs snippetAttrs;
    dnnl::impl::cpu::x64::cpu_isa_t host_isa;

    size_t input_num = 0;
    size_t output_num = 0;

    std::vector<MemoryPtr> srcMemPtrs = {};
    std::vector<MemoryPtr> dstMemPtrs = {};

    std::vector<ptrdiff_t> start_offset_in = {};
    std::vector<ptrdiff_t> start_offset_out = {};

    size_t tensor_rank = 0;
    bool is_dynamic = false;

    static const size_t rank6D {6};

    std::shared_ptr<SnippetExecutor> execPtr = nullptr;
};

class Subgraph::SnippetExecutor {
public:
    SnippetExecutor(SnippetAttrs attrs, bool is_dynamic, size_t tensor_rank,
                    const std::vector<ptrdiff_t>& start_offset_in, const std::vector<ptrdiff_t>& start_offset_out);
    virtual ~SnippetExecutor() = default;

    virtual void exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) = 0;
    virtual void prepare() = 0;

protected:
    SnippetAttrs snippet_attrs;
    bool is_dynamic = false;
    size_t tensor_rank = 0;

    std::vector<ptrdiff_t> start_offset_in = {};
    std::vector<ptrdiff_t> start_offset_out = {};
};

class Subgraph::SnippetJitExecutor : public Subgraph::SnippetExecutor {
public:
    SnippetJitExecutor(SnippetAttrs attrs, bool is_dynamic, size_t tensor_rank,
                       const std::vector<ptrdiff_t>& start_offset_in, const std::vector<ptrdiff_t>& start_offset_out);

    void exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;
    void prepare() override;

protected:
    typedef void (*kernel)(const void *, const void *);
    typedef void (*dynamic_kernel)(const void *);

    inline void update_ptrs(jit_snippets_call_args& call_args, const std::vector<MemoryPtr>& srcMemPtrs, const std::vector<MemoryPtr>& dstMemPtrs);
    inline void update_ptrs(jit_snippets_call_args& call_args, const std::vector<MemoryPtr>& srcMemPtrs, const std::vector<MemoryPtr>& dstMemPtrs,
                            const int64_t* indexes, const std::vector<std::vector<int64_t>>& data_offsets);
    // Evaluates generated snippet using parallel backend
    void schedule_6d(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs);
    void schedule_nt(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs);

    snippets::Schedule schedule;
    std::vector<std::vector<int64_t>> data_offsets = {};
    std::vector<jit_snippets_call_args::loop_args_t> loop_args = {};
    // Holds index of output used as in execution domain
    // it should be compatible with a schedule's work size
    std::vector<size_t> parallel_exec_domain = {};
    size_t harness_work_amount = 0;

    // Buffer scratchpad
    std::vector<uint8_t> buffer_scratchpad = {};
    size_t buffer_scratchpad_size = 0;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
