// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "snippets/op/subgraph.hpp"

#if defined(OPENVINO_ARCH_ARM64)
#include "cpu/aarch64/cpu_isa_traits.hpp"
#else
#include "cpu/x64/cpu_isa_traits.hpp"
#endif

#include <array>

namespace ov {
namespace intel_cpu {
namespace node {

class Subgraph : public Node {
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

    // Class for snippet compilation
    class SubgraphCodeGenerator;
    // Base class for executors
    class SubgraphExecutor;

protected:
    IShapeInfer::Result shapeInfer() const override;

private:
    void initMemoryPtrs();
    void initAttributes();
    void initStartOffsets();
    void initPluginBlockedShapes() const;
    void optimizeIR();

    snippets::op::Subgraph::BlockedShapeVector getSnippetsBlockedShapes() const;
    std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>> getIOPrecisions() const;

    static uint64_t getBodyHash(const std::shared_ptr<snippets::op::Subgraph>& snippet);
    uint8_t getBroadcastingMask(const std::vector<VectorDims>& input_shapes);

    using DataFlowPasses = std::vector<ov::snippets::pass::Manager::PositionedPassBase>;
    using ControlFlowPasses = std::vector<ov::snippets::lowered::pass::PassPipeline::PositionedPassLowered>;

    DataFlowPasses getDataFlowPasses() const;
    ControlFlowPasses getControlFlowPasses() const;

    // Holds ISA version used is codeGeneration target
#if defined(OPENVINO_ARCH_ARM64)
    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa;
#else
    dnnl::impl::cpu::x64::cpu_isa_t host_isa;
#endif

    std::shared_ptr<SubgraphAttrs> subgraph_attrs;

    size_t input_num = 0;
    size_t output_num = 0;

    std::vector<MemoryPtr> srcMemPtrs = {};
    std::vector<MemoryPtr> dstMemPtrs = {};

    std::vector<ptrdiff_t> start_offset_in = {};
    std::vector<ptrdiff_t> start_offset_out = {};

    bool is_dynamic = false;
    // Input shapes that are used in PrepareParams and ShapeInfer to avoid frequent memory allocation
    mutable std::vector<VectorDims> in_shapes;

    std::shared_ptr<SubgraphExecutor> execPtr = nullptr;
};

class Subgraph::SubgraphCodeGenerator {
public:
    SubgraphCodeGenerator(const std::shared_ptr<Subgraph::SubgraphAttrs>& snippet_attrs, const std::shared_ptr<CPURuntimeConfig>& config);

    const std::shared_ptr<snippets::Schedule>& get() const { return schedule; }

private:
    std::shared_ptr<snippets::Schedule> schedule;
};

class Subgraph::SubgraphExecutor {
public:
    SubgraphExecutor(const std::shared_ptr<Subgraph::SubgraphAttrs>& snippet_attrs,
                     const std::shared_ptr<Subgraph::SubgraphCodeGenerator>& snippet,
                     const std::vector<ptrdiff_t>& start_offset_in,
                     const std::vector<ptrdiff_t>& start_offset_out);
    virtual ~SubgraphExecutor() = default;

    virtual void exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) = 0;

protected:
    void parallel_for6d(const std::function<void(jit_snippets_call_args&)>& initializer,
                        const std::function<void(jit_snippets_call_args&, const size_t*)>& caller);
    void parallel_forNd(const std::function<void(jit_snippets_call_args&)>& initializer,
                        const std::function<void(jit_snippets_call_args&, const size_t*)>& caller);

    virtual void init_runtime_params(const std::shared_ptr<CPURuntimeConfig>& snippet_config);

    std::shared_ptr<snippets::Schedule> m_schedule;
    // Holds index of output used as in execution domain
    // it should be compatible with a schedule's work size
    std::vector<size_t> m_parallel_exec_domain = {};
    size_t m_harness_work_amount = 0;

    // Buffer scratchpad
    std::vector<uint8_t> m_buffer_scratchpad = {};
    size_t m_buffer_scratchpad_size = 0;
    int m_buffer_output_inplace = -1;

    const size_t rank6D = 6;

    // Count of threads for parallel_nt
    int m_nthreads = 0;

    std::vector<ptrdiff_t> m_start_offset_in = {};
    std::vector<ptrdiff_t> m_start_offset_out = {};

#ifdef SNIPPETS_DEBUG_CAPS
    bool enabled_segfault_detector = false;
    inline void segfault_detector();
#endif
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
