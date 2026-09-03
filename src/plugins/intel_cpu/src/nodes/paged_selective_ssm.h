// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/paged_selective_ssm_config.hpp"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu {

enum class PagedSelectiveSSMInputPort : uint8_t {
    A,
    TimeStep,
    InputProjection,
    Input,
    OutputProjection,
    State,
    SubsequenceBegins,
    BlockIndices,
    BlockIndicesBegins,
    NumProcessedTokens,
    CacheInterval,
    Count,
};

constexpr size_t input_port_index(PagedSelectiveSSMInputPort port) noexcept {
    return static_cast<size_t>(port);
}

inline constexpr size_t paged_ssm_input_count = input_port_index(PagedSelectiveSSMInputPort::Count);

enum class PagedSelectiveSSMOutputPort : uint8_t {
    Output,
    Count,
};

constexpr size_t output_port_index(PagedSelectiveSSMOutputPort port) noexcept {
    return static_cast<size_t>(port);
}

inline constexpr size_t paged_ssm_output_count = output_port_index(PagedSelectiveSSMOutputPort::Count);

inline constexpr std::array paged_ssm_computation_ports{
    PagedSelectiveSSMInputPort::A,
    PagedSelectiveSSMInputPort::TimeStep,
    PagedSelectiveSSMInputPort::InputProjection,
    PagedSelectiveSSMInputPort::Input,
    PagedSelectiveSSMInputPort::OutputProjection,
};

inline constexpr std::array paged_ssm_metadata_ports{
    PagedSelectiveSSMInputPort::SubsequenceBegins,
    PagedSelectiveSSMInputPort::BlockIndices,
    PagedSelectiveSSMInputPort::BlockIndicesBegins,
    PagedSelectiveSSMInputPort::NumProcessedTokens,
    PagedSelectiveSSMInputPort::CacheInterval,
};

namespace node {

class PagedSelectiveSSM : public Node {
public:
    PagedSelectiveSSM(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::PagedSelectiveSSM;
    }
    bool isExecutable() const override {
        return !isInputTensorAtPortEmpty(input_port_index(PagedSelectiveSSMInputPort::Input));
    }

    void createPrimitive() override;
    void prepareParams() override;
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& error_message) noexcept;

private:
    void bindMemoryArguments();
    PagedSelectiveSSMAttrs m_attrs;
    ExecutorFactoryPtr<PagedSelectiveSSMAttrs> m_factory;
    ExecutorPtr m_executor;
    MemoryArgs m_memory;
};

}  // namespace node
}  // namespace ov::intel_cpu
