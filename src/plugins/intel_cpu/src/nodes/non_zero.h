// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "dnnl_extension_utils.h"
#include "graph_context.h"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"

namespace ov::intel_cpu::node {

class NonZero : public Node {
public:
    NonZero(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;
    bool needShapeInfer() const override {
        return false;
    };
    bool needPrepareParams() const override {
        return false;
    };
    void executeDynamicImpl(const dnnl::stream& strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    bool neverExecute() const override {
        return false;
    }
    bool isExecutable() const override {
        return true;
    }

private:
    int threadsCount = 1;
    template <typename inputType>
    void executeSpecified();
    template <typename T>
    struct NonZeroExecute;
    template <typename T>
    std::vector<size_t> getNonZeroElementsCount(const T* src, const Shape& inShape);
};

}  // namespace ov::intel_cpu::node
