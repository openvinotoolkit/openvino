// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "common/permute_kernel.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class ShuffleChannels : public Node {
public:
    ShuffleChannels(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);
    ~ShuffleChannels() override = default;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;

    void prepareParams() override;
    struct ShuffleChannelsAttributes {
        LayoutType layoutType = LayoutType::nspc;
        int dataRank = 0;
        int axis = 0;
        int spatialRank = 0;
        size_t group = 0LU;
        size_t dataSize = 1LU;
        VectorDims srcDims;
        VectorDims srcBlockedDims;
        [[nodiscard]] size_t hash() const;
        bool operator==(const ShuffleChannelsAttributes& rhs) const;
    };

protected:
    void executeDynamicImpl(const dnnl::stream& strm) override;

private:
    ShuffleChannelsAttributes attrs;

    struct ShuffleChannelsExecutor final {
        explicit ShuffleChannelsExecutor(const ShuffleChannelsAttributes& attrs);
        void exec(const uint8_t* srcData, uint8_t* dstData, int MB);
        ~ShuffleChannelsExecutor() = default;

    private:
        std::unique_ptr<PermuteKernel> permuteKernel = nullptr;
    };
    using executorPtr = std::shared_ptr<ShuffleChannelsExecutor>;
    executorPtr execPtr = nullptr;
};

}  // namespace ov::intel_cpu::node
