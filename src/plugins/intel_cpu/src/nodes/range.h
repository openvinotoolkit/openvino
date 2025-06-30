// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class Range : public Node {
public:
    Range(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;
    [[nodiscard]] bool needPrepareParams() const override {
        return false;
    };
    [[nodiscard]] bool needShapeInfer() const override {
        return false;
    };
    void executeDynamicImpl(const dnnl::stream& strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    enum StatusCode : int8_t {
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

}  // namespace ov::intel_cpu::node
