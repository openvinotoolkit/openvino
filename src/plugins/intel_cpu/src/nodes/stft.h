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
#include "rdft.h"

namespace ov::intel_cpu::node {

class STFT : public Node {
public:
    STFT(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    [[nodiscard]] bool created() const override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    [[nodiscard]] bool needPrepareParams() const override;
    void createPrimitive() override;

    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    [[nodiscard]] bool canBeInPlace() const override {
        return false;
    }

protected:
    [[nodiscard]] bool needShapeInfer() const override;

private:
    /// STFT params
    bool m_transpose_frames = false;

    // RDFT executor
    std::shared_ptr<RDFTExecutor> rdft_executor = nullptr;
    bool m_is_frame_size_const = false;
    bool m_is_frame_step_const = false;

    // Input indices
    static constexpr size_t DATA_IDX = 0LU;
    static constexpr size_t WINDOW_IDX = 1LU;
    static constexpr size_t FRAME_SIZE_IDX = 2LU;
    static constexpr size_t FRAME_STEP_IDX = 3LU;
};

}  // namespace ov::intel_cpu::node
