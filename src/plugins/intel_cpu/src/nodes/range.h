// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class Range : public Node {
public:
    Range(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;
    bool needPrepareParams() const override {
        return false;
    };
    bool needShapeInfer() const override {
        return false;
    };
    void executeDynamicImpl(const dnnl::stream& strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    enum StatusCode : int {
        OK = 0,
        PARAMETER_MISMATCH = -1,
    };

private:
    template <typename data_t>
    StatusCode rangeKernel();
    template <typename data_t>
    size_t getWorkAmount(data_t* startPtr = nullptr, data_t* stopPtr = nullptr, data_t* stepPtr = nullptr) const;

    static const size_t RANGE_START = 0;
    static const size_t RANGE_LIMIT = 1;
    static const size_t RANGE_DELTA = 2;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
