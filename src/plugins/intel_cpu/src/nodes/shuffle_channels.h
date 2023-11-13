// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include "common/permute_kernel.h"

namespace ov {
namespace intel_cpu {
namespace node {

class ShuffleChannels : public Node {
public:
    ShuffleChannels(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);
    ~ShuffleChannels() override = default;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    void prepareParams() override;
    struct ShuffleChannelsAttributes {
        LayoutType layoutType;
        int dataRank = 0;
        int axis = 0;
        int spatialRank = 0;
        size_t group = 0lu;
        size_t dataSize = 1lu;
        VectorDims srcDims;
        VectorDims srcBlockedDims;
        size_t hash() const;
        bool operator==(const ShuffleChannelsAttributes& rhs) const;
    };

protected:
    void executeDynamicImpl(dnnl::stream strm) override;

private:
    ShuffleChannelsAttributes attrs;

    struct ShuffleChannelsExecutor final {
        ShuffleChannelsExecutor(const ShuffleChannelsAttributes& attrs);
        void exec(const uint8_t* srcData, uint8_t* dstData, const int MB);
        ~ShuffleChannelsExecutor() = default;

    private:
        std::unique_ptr<PermuteKernel> permuteKernel = nullptr;
    };
    using executorPtr = std::shared_ptr<ShuffleChannelsExecutor>;
    executorPtr execPtr = nullptr;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
