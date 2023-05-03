// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include "common/permute_kernel.h"
#include "executors/space_to_depth_list.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class SpaceToDepth : public Node {
public:
    SpaceToDepth(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    void prepareParams() override;

protected:
    void executeDynamicImpl(dnnl::stream strm) override;

private:
    SpaceToDepthAttrs attrs;
    SpaceToDepthExecutorPtr execPtr = nullptr;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
