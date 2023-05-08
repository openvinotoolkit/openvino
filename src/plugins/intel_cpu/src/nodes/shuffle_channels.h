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
#include "executors/shuffle_channels_list.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class ShuffleChannels : public Node {
public:
    ShuffleChannels(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);
    ~ShuffleChannels() override = default;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    void prepareParams() override;
    void addSupportedPrimDescFactory(const std::vector<PortConfigurator>& inPortConfigs,
                                     const std::vector<PortConfigurator>& outPortConfigs,
                                     impl_desc_type implType,
                                     bool dynBatchSupport = false);

protected:
    void executeDynamicImpl(dnnl::stream strm) override;

private:
    ShuffleChannelsAttributes attrs;

    ShuffleChannelsExecutorPtr execPtr = nullptr;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
