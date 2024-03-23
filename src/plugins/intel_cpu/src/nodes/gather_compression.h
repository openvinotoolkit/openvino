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

private:
    void execReference();

    template <typename IN_TYPE, typename OUT_TYPE>
    void execReference8bit();

    template <typename OUT_TYPE>
    void execReferenceU4();
    template <typename OUT_TYPE>
    void execReferenceI4();

    bool reverseIndexing = true;

    static constexpr size_t GATHER_DATA = 0;
    static constexpr size_t GATHER_INDICES = 1;
    static constexpr size_t GATHER_AXIS = 2;
    static constexpr size_t GATHER_SCALE = 3;
    static constexpr size_t GATHER_ZP = 4;

    int m_batchDims = 0;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov