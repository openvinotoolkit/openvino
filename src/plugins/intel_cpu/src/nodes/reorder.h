// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include "nodes/executors/transpose.hpp"
#include <utils/general_utils.h>

namespace ov {
namespace intel_cpu {
namespace node {

class Reorder : public Node {
public:
    Reorder(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);
    Reorder(const std::string& name, const GraphContext::CPtr context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    const std::vector<impl_desc_type>& getDefaultImplPriority() override;

    bool isExecutable() const override;

    void createPrimitive() override;

    void prepareParams() override;

    void executeDynamicImpl(dnnl::stream strm) override;

    void setDescs(const MemoryDesc& input, const MemoryDesc& output) {
        this->input = input.clone();
        inputShapes.clear();
        inputShapes.push_back(this->input->getShape());

        this->output = output.clone();
        outputShapes.clear();
        outputShapes.push_back(this->output->getShape());
    }

    void setSrcPermutation(const std::vector<int> & src_perm) {
        this->src_permutation = src_perm;
    }

    void setOptimized(bool isOptimized) {
        this->isOptimized = isOptimized;
    }

    bool canBeInPlace() const override {
        return false;
    }

    const MemoryDesc& getInput() { return *input; }
    const MemoryDesc& getOutput() { return *output; }

    static std::string getReorderArgs(const MemoryDesc &parentDesc, const MemoryDesc &childDesc);

    static void reorderData(const IMemory &input, const IMemory &output, MultiCachePtr cache = nullptr);

private:
    dnnl::reorder::primitive prim;
    std::shared_ptr<MemoryDesc> input;
    std::shared_ptr<MemoryDesc> output;

    std::vector<int> src_permutation;

    MemoryPtr dst_blocked;
    MemoryPtr src_blocked;

    bool isOptimized = false;

    bool isNspc2NcspCase = false;
    bool isNcsp2NspcCase = false;
    bool canUseNspc2Ncsp = false;
    bool canUseNcsp2Nspc = false;

    void optimizedNspc2Ncsp();
    void optimizedNcsp2Nspc();
    void createReorderPrimitive(const dnnl::memory::desc &srcDesc, void* srcPtr, const dnnl::memory::desc &dstDesc, void* dstPtr);
#if defined(OV_CPU_ARM_ENABLE_FP16)
    void prepareReorderAsTranspose(MemoryDescPtr parentDesc, MemoryDescPtr childDesc);
    TransposeExecutorPtr transposeExecutor;
#endif
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
