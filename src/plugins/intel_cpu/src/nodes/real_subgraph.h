// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"
#include "graph.h"
#include <atomic>
#include <future>
#include <memory>
#include <thread>

namespace ov {
namespace intel_cpu {
namespace node {

class SubGraph : public Node {
public:
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    SubGraph(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    bool created() const override;

    void getSupportedDescriptors() override;

    void selectOptimalPrimitiveDescriptor() override;

    void createPrimitive() override;
    void prepareParams() override;

    void execute(dnnl::stream) override;
    void executeDynamicImpl(dnnl::stream strm) override;
    void resolveInPlaceEdges(Edge::LOOK look) override;
    bool needShapeInfer() const override {
        return false;
    }

    bool needPrepareParams() const override {
        return false;
    }

    bool isExecutable() const override {
        return true;
    }

    void infer();

    void markAsRunning() override {
        m_isRunning = true;
    }

    void wait() override {
        while (m_isRunning.load()) {
            // std::this_thread::sleep_for(std::chrono::nanoseconds(50));
            std::this_thread::yield();
        }
    }

    const Graph& graph() const {
        return m_graph;
    }

    int getSubStream() const {
        return m_subStreamToUse;
    }

private:
    std::shared_ptr<const ov::Model> m_body;
    Graph m_graph;
    std::atomic<bool> m_isRunning{false};
    // int m_subStreamToUse = -1;
    std::shared_ptr<Executor> executor;
};

} // namespace node
} // namespace intel_cpu
} // namespace ov
