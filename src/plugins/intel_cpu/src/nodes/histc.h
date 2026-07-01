// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

class Histc : public Node {
public:
    Histc(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    void prepareParams() override;

    [[nodiscard]] bool needShapeInfer() const override {
        return false;
    }

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    static constexpr size_t INPUT_DATA_PORT = 0;
    static constexpr size_t OUTPUT_PORT = 0;

    int64_t m_bins = 100;
    double m_min_val = 0.0;
    double m_max_val = 0.0;

    ov::element::Type m_data_precision;
    ov::element::Type m_output_precision;
};

}  // namespace ov::intel_cpu::node
