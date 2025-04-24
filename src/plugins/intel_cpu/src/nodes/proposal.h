// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"
#include "proposal_imp.hpp"

using proposal_conf = ov::Extensions::Cpu::proposal_conf;

namespace ov {
namespace intel_cpu {
namespace node {

class Proposal : public Node {
public:
    Proposal(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    bool needPrepareParams() const override {
        return false;
    };
    void executeDynamicImpl(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    const size_t PROBABILITIES_IN_IDX = 0lu;
    const size_t ANCHORS_IN_IDX = 1lu;
    const size_t IMG_INFO_IN_IDX = 2lu;
    const size_t ROI_OUT_IDX = 0lu;
    const size_t PROBABILITIES_OUT_IDX = 1lu;

    proposal_conf conf;
    std::vector<float> anchors;
    std::vector<int> roi_indices;
    bool store_prob;  // store blob with proposal probabilities
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
