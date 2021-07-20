// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>

#include <mkldnn.hpp>
#include <cpu/x64/jit_generator.hpp>

#include "mkldnn_node.h"
#include "snippets/op/subgraph.hpp"

#include <array>

namespace MKLDNNPlugin {

/// MKLDNNSnippetNode represents subgraph node in MKLDNN plugin
/// potentially, snippet can be placed as a postop to any support operation while it doesn't support postops itself
/// precision: fp32
class MKLDNNSnippetNode : public MKLDNNNode {
public:
    MKLDNNSnippetNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNSnippetNode() override = default;

    // It should be initSupportedDescriptors after all
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void selectOptimalPrimitiveDescriptor() override;

    // Here we convert to canonical for & jit everything
    void createPrimitive() override;

    bool created() const override;

    // if generator is set, it would execute generated code otherwise it would fallback to nGraph reference
    void execute(mkldnn::stream strm) override;

private:
    static const size_t rank6D {6};

    typedef void (*kernel)(const void *, const void *, const void *);

    // Evaluates generated snippet in a single thread
    bool evaluate(const ngraph::HostTensorVector& outputs, const ngraph::HostTensorVector& inputs) const;

    // Interpret snippet with nGraph reference
    void interpret() const;

    void define_shedule();

    void generate();

    // Evaluates generated snippet using parallel backend
    void shedule_nt(const std::vector<uint8_t *>& outputs, const std::vector<const uint8_t *>& inputs) const;
    void shedule_6d(const std::vector<uint8_t *>& outputs, const std::vector<const uint8_t *>& inputs) const;

    // Local copy of subgraph node for canonization & code generation
    std::shared_ptr<ngraph::snippets::op::Subgraph> snippet;
    // Original subgraph node for fallback and regression testing
    // store it here since MKLDNN eraces CNNLayers at some point
    std::shared_ptr<ngraph::snippets::op::Subgraph> snippet_ref;

    // Holds generated snippet with information about how to shedule it
    ngraph::snippets::Schedule schedule;

    // Holds ISA version used is codeGeneration target
    dnnl::impl::cpu::x64::cpu_isa_t host_isa;

    // Holds index of output used as in execution domain
    // it should be compatible with a schedule's work size
    size_t max_rank_out_desc_idx = 0;

    /// scheduling info
    bool isDynBatchEnabled = false;
    size_t batchDimIdx = 0;
    size_t tensorRank = 0;
    size_t tileRank = 1;
    size_t fullWorkAmount = 0;
    size_t schedulerWorkAmount = 0;
    const size_t maxTileRank = 2;

    std::vector<std::vector<int64_t>> dims_in = {};
    std::vector<std::vector<int64_t>> offsets_in = {};
    std::vector<ptrdiff_t> start_offset_in = {};

    std::vector<std::vector<int64_t>> dims_out = {};
    std::vector<std::vector<int64_t>> offsets_out = {};
    std::vector<ptrdiff_t>  start_offset_out = {};

    std::vector<int64_t> sch_dims = {};
    std::vector<int64_t> sch_offsets_in = {};
    std::vector<int64_t> sch_offsets_out = {};

    // Minimalistic structure which encapsulates kernel arguments
    struct CallArgs {
        void push(int64_t arg) {
            args[nargs].i = arg;
            nargs++;
        }
        void push(const uint8_t* arg) {
            args[nargs].ptr = arg;
            nargs++;
        }
        const void* raw() const {
            return &args[0];
        }
    private:
        union param {
            const uint8_t* ptr;
            int64_t i;
        };

        std::array<param, 8> args = {};
        size_t nargs = {0};
    };
};

}  // namespace MKLDNNPlugin
