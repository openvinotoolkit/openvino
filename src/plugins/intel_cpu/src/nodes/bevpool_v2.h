// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "openvino/op/bevpool_v2.hpp"

namespace ov::intel_cpu::node {

class BevPoolV2 : public Node {
public:
    BevPoolV2(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;
    [[nodiscard]] bool needPrepareParams() const override {
        return false;
    }
    [[nodiscard]] bool canBeInPlace() const override {
        return false;
    }

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                     std::string& errorMessage) noexcept;

private:
    template <typename T, typename IdxT>
    void executeImpl();

    uint32_t m_input_channels = 0;
    uint32_t m_output_channels = 0;
    uint32_t m_image_width = 0;
    uint32_t m_image_height = 0;
    uint32_t m_feature_width = 0;
    uint32_t m_feature_height = 0;
    ov::op::v15::Bound m_d_bound{};

    int64_t m_feature_area = 0;
    int64_t m_depth_bins = 0;
    int64_t m_depth_span = 0;
    int64_t m_out_plane = 0;

    static constexpr size_t CF_IDX = 0;
    static constexpr size_t DW_IDX = 1;
    static constexpr size_t IDX_IDX = 2;
    static constexpr size_t ITV_IDX = 3;
};

}  // namespace ov::intel_cpu::node
