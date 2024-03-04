// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include "kernels/x64/gather_uni_kernel.hpp"

#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class GatherCompression : public Node {
public:
    GatherCompression(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    void executeDynamicImpl(dnnl::stream strm) override;
    bool needPrepareParams() const override;
    void prepareParams() override;

private:
    void execReference();
    void execReferenceU8();
    void execReferenceU4();

    bool isAxisInputConst = false;
    bool reverseIndexing = false;

    static constexpr size_t GATHER_DATA = 0;
    static constexpr size_t GATHER_ZP = 1;
    static constexpr size_t GATHER_SCALE = 2;
    static constexpr size_t GATHER_INDICES = 3;
    static constexpr size_t GATHER_AXIS = 4;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
