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
    Snippet(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);
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
    std::vector<VectorDims> shapeInfer() const override;
    bool needPrepareParams() const override;

    bool canBeInPlace() const override;
    bool created() const override;

    // if generator is set, it would execute generated code otherwise it would fallback to nGraph reference
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;
//    std::vector<VectorDims> shapeInfer() const override;

private:
    static const size_t rank6D {6};

    typedef void (*kernel)(const void *, const void *);

    // Create a deep local copy of the input snippet to perform canonicalization & code generation
    // TODO: Probably better to implement a proper copy constructor
    // NOTE: Before call mutex should be initialized
    void copy_snippet();

    void define_schedule();
    static  ov::PartialShape prependWithOnes(const PartialShape& dims, size_t rank);
    void normalizeShapes();
    void optimizeExecDomain(std::vector<PartialShape>&, std::vector<PartialShape>&, PartialShape&, size_t&) const;
    void calcJITParams(std::vector<int64_t>& offsets, std::vector<int64_t>& sch_offsets, std::vector<bool>& broadcasting_mask,
                       std::vector<int64_t>& vector_tile_increments, std::vector<int64_t>& scalar_tile_increments) const;

    void generate(const jit_snippets_compile_args*);

    // Evaluates generated snippet using parallel backend
    void schedule_6d(const jit_snippets_call_args& const_args) const;
    void schedule_6d_dynamic(const jit_snippets_call_args& const_args) const;
    void schedule_nt(const jit_snippets_call_args& const_args) const;

    // Original subgraph node
    std::shared_ptr<ngraph::snippets::op::Subgraph> original_snippet;
    // Local copy of subgraph node for canonization & code generation
    std::shared_ptr<ngraph::snippets::op::Subgraph> snippet;
    NodeVector snippet_inputs;  // dummy inputs used to simplify reshape in dynamic scenario

    // Holds generated snippet with information about how to schedule it
    ngraph::snippets::Schedule schedule;

    // Holds ISA version used is codeGeneration target
    dnnl::impl::cpu::x64::cpu_isa_t host_isa;
    size_t isa_num_lanes; // number of elements that fit in vector size

    // Holds index of output used as in execution domain
    // it should be compatible with a schedule's work size
    std::vector<size_t> exec_domain = {};

    /// scheduling info
    size_t batchDimIdx = 0;
    size_t tensorRank = 0;
    size_t tileRank = 1;
    size_t fullWorkAmount = 0;
    size_t harnessWorkAmount = 0;
    const size_t maxTileRank = 2;

    std::vector<MemoryPtr> srcMemPtrs = {};
    std::vector<MemoryPtr> dstMemPtrs = {};
    std::vector<size_t> dataSize = {};

    std::vector<int64_t> data_offsets;
    std::vector<int64_t> scheduler_offsets;
    std::vector<bool> broadcasting_mask; // one bool for every input/output. If true then this input is broadcasted
    std::vector<size_t> scheduler_work_amounts;
    std::vector<size_t> static_master_shape_placeholder = {}; // placeholder to pass per-inference static master_shape for dynamic cases
    std::vector<int64_t> vector_tile_increments = {}; // increments for vector (and scalar) tiles used in dynamic tiles.
    std::vector<int64_t> scalar_tile_increments = {};

    // this is needed for fast shape inference of blocking-invariant prepended shapes
    std::vector<bool> inputShapeIsBlocked = {}; // we need this info to shape-infer mixed layouts
    std::vector<bool> outputShapeIsBlocked = {}; // we need this info to shape-infer mixed layouts
    bool masterShapeIsBlocked = false;
    //

    // need to remember the original ones to avoid reshaping body in dynamic case
    std::vector<PartialShape> originalNormOutputShapes = {};
    // master shape is mutable since we need to modify it inside const shapeInfer method
    mutable PartialShape masterShape = {};
    // body Input & output shapes anre optimized and not necessarily the same as inputShapes and outputShapes
    mutable std::vector<PartialShape> normInputShapes = {};
    mutable std::vector<PartialShape> normOutputShapes = {};

    std::vector<ptrdiff_t> start_offset_in = {};
    std::vector<ptrdiff_t> start_offset_out = {};

    std::vector<std::vector<size_t>> dims_out = {};
    std::vector<std::vector<size_t>> offsets_out = {};

    std::vector<int64_t> sch_dims = {};
    std::vector<int64_t> sch_offsets_in = {};
    std::vector<int64_t> sch_offsets_out = {};
    bool canUseOptimizedImpl = true;
    // memory buffer for physical broadcasting in dynamic case, use std::vector to facilitate memory management
    std::vector<float> scratchpad_memory_chunk = {};
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
