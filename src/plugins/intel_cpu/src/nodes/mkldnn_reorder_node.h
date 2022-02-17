// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>
#include <utils/general_utils.h>

namespace MKLDNNPlugin {

class MKLDNNReorderNode : public MKLDNNNode {
public:
    MKLDNNReorderNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    MKLDNNReorderNode(const std::string& name, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    const std::vector<impl_desc_type>& getPrimitivesPriority() override;

    bool isExecutable() const override;

    void createPrimitive() override;

    std::vector<VectorDims> shapeInfer() const override;

    void prepareParams() override;

    void executeDynamicImpl(mkldnn::stream strm) override;

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

    void setDynamicBatchLim(int lim) override;

    bool canBeInPlace() const override {
        return false;
    }

    const MemoryDesc& getInput() { return *input; }
    const MemoryDesc& getOutput() { return *output; }

    static std::string getReorderArgs(const MemoryDesc &parentDesc, const MemoryDesc &childDesc);

    static void reorderData(const MKLDNNMemory &input, const MKLDNNMemory &output);

private:
    std::shared_ptr<MemoryDesc> input;
    std::shared_ptr<MemoryDesc> output;

    MKLDNNMemoryPtr dst_blocked;
    MKLDNNMemoryPtr src_blocked;

    bool isOptimized = false;

    bool isNspc2NcspCase = false;
    bool isNcsp2NspcCase = false;
    bool canUseNspc2Ncsp = false;
    bool canUseNcsp2Nspc = false;

    void optimizedNspc2Ncsp();
    void optimizedNcsp2Nspc();
    void createReorderPrimitive(const mkldnn::memory::desc &srcDesc, void* srcPtr, const mkldnn::memory::desc &dstDesc, void* dstPtr);
};

}  // namespace MKLDNNPlugin
