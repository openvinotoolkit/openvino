// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>
#include <ie_blob.h>

namespace MKLDNNPlugin {

class MKLDNNOneHotNode : public MKLDNNNode {
public:
    MKLDNNOneHotNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    typedef InferenceEngine::PrecisionTrait<InferenceEngine::Precision::I32>::value_type in_type;

    struct OneHotContext {
        MKLDNNOneHotNode* nodePtr;
        size_t prefix_size;
        size_t suffix_size;
    };

    template<typename dst_t>
    struct OneHotExecute {
        void operator()(OneHotContext & ctx) {
            ctx.nodePtr->one_hot<dst_t>(ctx.prefix_size, ctx.suffix_size);
        }
    };

    uint32_t depth;
    int32_t axis = -1;
    InferenceEngine::SizeVector src_dims;
    InferenceEngine::SizeVector dst_dims;

    InferenceEngine::Precision output_precision;

    std::string errorPrefix;

    static const size_t INDICES_ID = 0;
    static const size_t DEPTH_ID = 1;
    static const size_t ON_VALUE_ID = 2;
    static const size_t OFF_VALUEAXES_ID = 3;

    template<typename out_type>
    void one_hot(size_t prefix_size, size_t suffix_size);
};

}  // namespace MKLDNNPlugin
