// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <vector>
#include <utility>
#include <map>

namespace MKLDNNPlugin {

class MKLDNNPermuteNode : public MKLDNNNode {
public:
    MKLDNNPermuteNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng);
    ~MKLDNNPermuteNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

private:
    static Register<MKLDNNPermuteNode> reg;
    InferenceEngine::SizeVector order;

    typedef std::function<void(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr)> permuteImpl;
    typedef std::function<bool(MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr)> isApplicable;
    struct PermuteImpl {
        PermuteImpl(permuteImpl f0, isApplicable f1): execute(std::move(f0)), isValidParams(std::move(f1)) {}

        permuteImpl execute;
        isApplicable isValidParams;
    };

    static std::map<InferenceEngine::SizeVector, PermuteImpl> OptimizedCases;
};

}  // namespace MKLDNNPlugin

