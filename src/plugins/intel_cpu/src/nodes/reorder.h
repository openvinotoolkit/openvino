// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include "nodes/executors/transpose.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class Reorder : public Node {
public:
    Reorder(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);
    Reorder(const MemoryDesc& input,
            const MemoryDesc& output,
            const std::string& name,
            const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;
    const std::vector<impl_desc_type>& getDefaultImplPriority() override;

    bool neverExecute() const override;
    bool isExecutable() const override;

    void createPrimitive() override;

    void prepareParams() override;

    void executeDynamicImpl(const dnnl::stream& strm) override;

    void setSrcPermutation(const std::vector<int>& src_perm) {
        this->src_permutation = src_perm;
    }

    void setOptimized(bool isOptimized) {
        this->isOptimized = isOptimized;
    }

    bool getOptimized() const {
        return isOptimized;
    }

    bool canBeInPlace() const override {
        return false;
    }

    const MemoryDesc& getInput() {
        return *input;
    }
    const MemoryDesc& getOutput() {
        return *output;
    }

    static std::string getReorderArgs(const MemoryDesc& parentDesc, const MemoryDesc& childDesc);

    static void reorderData(const IMemory& input, const IMemory& output, const MultiCachePtr& cache = nullptr);

private:
    dnnl::reorder::primitive prim;
    std::shared_ptr<MemoryDesc> input;
    std::shared_ptr<MemoryDesc> output;

    std::vector<int> src_permutation;

    bool isOptimized = false;

    bool isNspc2NcspCase = false;
    bool isNcsp2NspcCase = false;
    bool canUseNspc2Ncsp = false;
    bool canUseNcsp2Nspc = false;

    void optimizedNspc2Ncsp();
    void optimizedNcsp2Nspc();
    void createReorderPrimitive(const DnnlMemoryDescPtr& srcDesc, const DnnlMemoryDescPtr& dstDesc);

    void prepareReorderAsTranspose(const MemoryDescPtr& parentDesc, const MemoryDescPtr& childDesc);
    TransposeExecutorPtr transposeExecutor;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
