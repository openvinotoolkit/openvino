// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>

#include <onednn/dnnl.h>
#include <cpu/x64/jit_generator.hpp>
#include "emitters/jit_snippets_emitters.hpp"

#include <node.h>
#include "snippets/op/subgraph.hpp"

#include <array>

namespace ov {
namespace intel_cpu {
namespace node {

/// Snippet represents subgraph node in CPU plugin
/// potentially, snippet can be placed as a postop to any support operation while it doesn't support postops itself
/// precision: fp32
class Snippet : public Node {
public:
    Snippet(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);
    ~Snippet() override = default;

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void selectOptimalPrimitiveDescriptor() override;
    InferenceEngine::Precision getRuntimePrecision() const override;

    // to avoid collisions in throughput mode with copy of TypeRelaxed nodes
    // we should have common shared mutex between streams
    void setSharedMutex(const std::shared_ptr<std::mutex>& mutex);

    // Here we convert to canonical for & jit everything
    void createPrimitive() override;
    void prepareParams() override;
    std::vector<VectorDims> shapeInfer();
    bool needPrepareParams() const override;

    bool canBeInPlace() const override;
    bool created() const override;

    // if generator is set, it would execute generated code otherwise it would fallback to nGraph reference
    void execute(dnnl::stream strm) override;

private:
    static const size_t rank6D {6};

    typedef void (*kernel)(const void *, const void *);

    // Create a deep local copy of the input snippet to perform canonicalization & code generation
    // TODO: Probably better to implement a proper copy constructor
    // NOTE: Before call mutex should be initialized
    void copy_snippet();

    ov::PartialShape canonicalizeBody();
    // returns true if exec domain was modified
    bool optimizeExecDomain(std::vector<VectorDims>&, std::vector<VectorDims>&, VectorDims&, size_t&) const;

    void generate(const jit_snippets_compile_args*);
    inline void update_ptrs(jit_snippets_call_args&);
    // Evaluates generated snippet using parallel backend
    void schedule_6d();
    void schedule_nt();

    // Original subgraph node
    std::shared_ptr<ngraph::snippets::op::Subgraph> original_snippet;
    // Local copy of subgraph node for canonization & code generation
    std::shared_ptr<ngraph::snippets::op::Subgraph> snippet;

    // Holds generated snippet with information about how to schedule it
    ngraph::snippets::Schedule schedule;

    // Holds ISA version used is codeGeneration target
    dnnl::impl::cpu::x64::cpu_isa_t host_isa;
    size_t isa_num_lanes; // number of elements that fit in vector size

    // Holds index of output used as in execution domain
    // it should be compatible with a schedule's work size
    std::vector<size_t> exec_domain = {};

    /// scheduling info
    size_t tensorRank = 0;
    size_t tileRank = 1;
    size_t fullWorkAmount = 0;
    size_t harnessWorkAmount = 0;
    const size_t maxTileRank = 2;

    std::vector<MemoryPtr> srcMemPtrs = {};
    std::vector<MemoryPtr> dstMemPtrs = {};
    std::vector<size_t> dataSize = {};

    // this is needed for fast shape inference of blocking-invariant prepended shapes
    std::vector<bool> inputShapeIsBlocked = {}; // we need this info to shape-infer mixed layouts
    std::vector<bool> outputShapeIsBlocked = {}; // we need this info to shape-infer mixed layouts
    bool masterShapeIsBlocked = false;

    VectorDims masterShape = {};
    std::vector<VectorDims> normInputShapes = {};
    std::vector<VectorDims> normOutputShapes = {};

    std::vector<ptrdiff_t> start_offset_in = {};
    std::vector<ptrdiff_t> start_offset_out = {};

    // Buffer scratchpad
    std::vector<uint8_t> buffer_scratchpad = {};
    size_t buffer_scratchpad_size = 0;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
