// Copyright (C) 2018-2021 Intel Corporation
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
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    const std::vector<impl_desc_type>& getPrimitivesPriority() override;

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

private:
    std::unique_ptr<MemoryDesc> input;
    std::unique_ptr<MemoryDesc> output;

    MKLDNNMemoryPtr dst_blocked;
    MKLDNNMemoryPtr src_blocked;

    bool isOptimized = false;
    bool canUseOptimizedNspc2Ncsp = false;
    bool canUseOptimizedNcsp2Nspc = false;

    void optimizedNspc2Ncsp();
    void optimizedNcsp2Nspc();
    void createReorderPrimitive(const mkldnn::memory::desc &srcDesc, void* srcPtr, const mkldnn::memory::desc &dstDesc, void* dstPtr);
};

}  // namespace MKLDNNPlugin
