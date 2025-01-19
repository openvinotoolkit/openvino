// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_shape.h"
#include "node.h"
#include "graph_context.h"
#include "edge.h"
#include "openvino/core/shape.hpp"

namespace ov {
namespace intel_cpu {
namespace cpu_unit_test {

class DummyNode : public Node {
public:
    // dummy node of the same shape and precision to both input and output.
    DummyNode(const ov::PartialShape& shape,
              const ov::element::Type_t& prc,
              const std::string& name,
              const std::string& type,
              const GraphContext::CPtr context,
              LayoutType layout = LayoutType::ncsp,
              int in_place_direction = Edge::LOOK::LOOK_UP,
              bool is_executable = false) :
        Node(type,
             {ov::intel_cpu::Shape(shape)},
             {ov::intel_cpu::Shape(shape)},
             {prc},
             {prc},
             name,
             context),
        m_layout(layout),
        m_inplace(in_place_direction),
        m_is_executable(is_executable) {
    }

    void getSupportedDescriptors() override {
        if (getParentEdges().size() != 1)
            OPENVINO_THROW("Incorrect number of input edges for layer " + getName());
        if (getChildEdges().empty())
            OPENVINO_THROW("Incorrect number of output edges for layer " + getName());
    }

    void initSupportedPrimitiveDescriptors() override {
        if (!supportedPrimitiveDescriptors.empty())
            return;

        NodeConfig config;
        config.inConfs.resize(1);
        config.outConfs.resize(1);

        config.inConfs[0].inPlace(m_inplace == static_cast<int>(Edge::LOOK::LOOK_DOWN) ||
                                  m_inplace == static_cast<int>(Edge::LOOK::LOOK_BOTH)? 0 : -1);
        config.inConfs[0].constant(false);
        config.outConfs[0].inPlace(m_inplace == static_cast<int>(Edge::LOOK::LOOK_UP) ||
                                   m_inplace == static_cast<int>(Edge::LOOK::LOOK_BOTH) ? 0 : -1);
        config.outConfs[0].constant(false);

        auto layoutCreator = BlockedDescCreator::getCommonCreators().at(m_layout);
        auto& originInputPrecisions = getOriginalInputPrecisions();
        config.inConfs[0].setMemDesc(layoutCreator->createSharedDesc(originInputPrecisions[0], getInputShapeAtPort(0)));
        config.outConfs[0].setMemDesc(layoutCreator->createSharedDesc(originInputPrecisions[0], getOutputShapeAtPort(0)));

        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::undef);
    };

    bool isExecutable() const override {return m_is_executable;}
    void execute(const dnnl::stream& strm) override {};
    bool created() const override {return true;}

    bool needPrepareParams() const override {
        return false;
    }

    bool needShapeInfer() const override {
        return false;
    }

private:
    using Node::Node;

private:
    LayoutType m_layout = LayoutType::ncsp;
    int m_inplace = Edge::LOOK::LOOK_UP;
    bool m_is_executable = false;
};
} // namespace cpu_unit_test
} // namespace intel_cpu
} // namespace ov
