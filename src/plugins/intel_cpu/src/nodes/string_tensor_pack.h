// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class StringTensorPack : public Node {
public:
    StringTensorPack(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    bool needPrepareParams() const override;
    void executeDynamicImpl(dnnl::stream strm) override;

private:
    template <class OV_INDEX_TYPE>
    void executeImpl();

    template<typename T_idx>
    struct StringTensorPackExecute;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
