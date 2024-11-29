// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include "cpu_shape.h"
#include "memory_desc/blocked_memory_desc.h"
#include "node.h"
#include "graph_context.h"
#include "edge.h"
#include "node_config.h"
#include "openvino/core/partial_shape.hpp"

namespace ov {
namespace intel_cpu {
namespace cpu_unit_test {

static std::shared_ptr<CpuBlockedMemoryDesc> createSharedDesc(const ov::PartialShape& shape,
                                                              const ov::element::Type& precision,
                                                              const LayoutType layout = LayoutType::ncsp) {
    const auto& layoutCreator = BlockedDescCreator::getCommonCreators().at(layout);
    return layoutCreator->createSharedDesc(precision, {ov::intel_cpu::Shape(shape)});
}

struct NoOpExecutor {
    void operator()(){
        // do nothing
    }
};

class GenericNode : public Node {
public:
    GenericNode(const ov::PartialShape& shape,
                const ov::element::Type_t& prc,
                const std::string& name,
                const std::string& type,
                const GraphContext::CPtr context,
                std::vector<PortConfig> inplace_input,
                std::vector<PortConfig> inplace_output,
                bool is_executable)
        : Node(type, {ov::intel_cpu::Shape(shape)}, {ov::intel_cpu::Shape(shape)}, {prc}, {prc}, name, context),
          inputPortsConfig(inplace_input),
          outputPortsConfig(inplace_output),
          m_is_executable(is_executable) {}

    // single input single output node of the same shape and precision to both input and output.
    GenericNode(const ov::PartialShape& shape,
                const ov::element::Type_t& prc,
                const std::string& name,
                const std::string& type,
                const GraphContext::CPtr context,
                LayoutType layout = LayoutType::ncsp,
                int in_place_direction = Edge::LOOK::LOOK_UP,
                bool is_executable = false)
        : GenericNode(shape,
                      prc,
                      name,
                      type,
                      context,
                      std::vector<PortConfig>{
                          {
                              createSharedDesc(shape, prc, layout),
                              BlockedMemoryDesc::FULL_MASK,
                              in_place_direction == static_cast<int>(Edge::LOOK_DOWN) ||
                              in_place_direction == static_cast<int>(Edge::LOOK_BOTH)
                              ? 0
                              : -1,
                              false
                          }
                      },
                      std::vector<PortConfig>{
                          {
                              createSharedDesc(shape, prc, layout),
                              BlockedMemoryDesc::FULL_MASK,
                              in_place_direction == static_cast<int>(Edge::LOOK_UP) ||
                              in_place_direction == static_cast<int>(Edge::LOOK_BOTH)
                              ? 0
                              : -1,
                              false
                          }
                      },
                      is_executable) {}

    void getSupportedDescriptors() override {
        if (getParentEdges().size() != 1)
            OPENVINO_THROW("Incorrect number of input edges for layer " + getName());
        if (getChildEdges().empty())
            OPENVINO_THROW("Incorrect number of output edges for layer " + getName());
    }

    void initSupportedPrimitiveDescriptors() override {
        if (!supportedPrimitiveDescriptors.empty())
            return;

        NodeConfig nodeConfig;
        nodeConfig.inConfs.reserve(inputPortsConfig.size());
        nodeConfig.outConfs.reserve(outputPortsConfig.size());

        for (const auto& config : inputPortsConfig) {
            nodeConfig.inConfs.push_back(config);
        }

        for (const auto& config : outputPortsConfig) {
            nodeConfig.outConfs.push_back(config);
        }

        supportedPrimitiveDescriptors.emplace_back(nodeConfig, impl_desc_type::undef);
    };

    bool isExecutable() const override {
        return m_is_executable;
    }

    void execute(dnnl::stream strm) override {};

    bool created() const override {
        return true;
    }

    bool needPrepareParams() const override {
        return false;
    }

    bool needShapeInfer() const override {
        return false;
    }

private:
    using Node::Node;

private:
    std::vector<PortConfig> inputPortsConfig;
    std::vector<PortConfig> outputPortsConfig;
    bool m_is_executable = false;
};
}  // namespace cpu_unit_test
} // namespace intel_cpu
} // namespace ov
