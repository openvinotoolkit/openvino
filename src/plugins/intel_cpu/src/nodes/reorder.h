// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <utils/general_utils.h>

namespace ov {
namespace intel_cpu {
namespace node {

class Reorder : public Node {
public:
    Reorder(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);
    Reorder(const std::string& name, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    const std::vector<impl_desc_type>& getPrimitivesPriority() override;

    bool isExecutable() const override;

    void createPrimitive() override;

    std::vector<VectorDims> shapeInfer() const override;

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

    void setOptimized(bool isOptimized) {
        this->isOptimized = isOptimized;
    }
    bool getOptimized() {
        return this->isOptimized;
    }

    void setDynamicBatchLim(int lim) override;

    bool canBeInPlace() const override {
        return false;
    }

    const MemoryDesc& getInput() { return *input; }
    const MemoryDesc& getOutput() { return *output; }

    static std::string getReorderArgs(const MemoryDesc &parentDesc, const MemoryDesc &childDesc);

    static void reorderData(const Memory &input, const Memory &output);

private:
    std::shared_ptr<MemoryDesc> input;
    std::shared_ptr<MemoryDesc> output;

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
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
