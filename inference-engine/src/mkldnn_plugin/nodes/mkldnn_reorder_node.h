// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

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

    void setDescs(const InferenceEngine::TensorDesc& input, const InferenceEngine::TensorDesc& output) {
        this->input = input;
        this->output = output;
    }

    void setOptimized(bool isOptimized) {
        this->isOptimized = isOptimized;
    }

    void setDynamicBatchLim(int lim) override;

    bool canBeInPlace() const override {
        return false;
    }

    const InferenceEngine::TensorDesc& getInput() { return input; }
    const InferenceEngine::TensorDesc& getOutput() { return output; }

    /**
     * @brief A pointer to a scales blob
     */
    InferenceEngine::Blob::Ptr _scales;

private:
    InferenceEngine::TensorDesc input;
    InferenceEngine::TensorDesc output;

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
