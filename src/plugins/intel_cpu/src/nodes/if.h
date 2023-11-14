// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <graph.h>

#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class If : public Node {
public:
    If(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void initSupportedPrimitiveDescriptors() override;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(dnnl::stream strm) override;
    bool isExecutable() const override { return true; }

    void resolveInPlaceEdges(Edge::LOOK look) override;

protected:
    void executeDynamicImpl(dnnl::stream strm) override;
    bool needPrepareParams() const override { return false; };
    bool needShapeInfer() const override { return false; }

private:
    void prepareBeforeMappers(const bool isThen, const dnnl::engine& eng);
    void prepareAfterMappers(const bool isThen, const dnnl::engine& eng);

    std::deque<MemoryPtr> getToMemories(const Node* node, const size_t port) const;

    struct PortMap {
        int from; /**< Index of external/internal out data */
        int to;   /**< Index of external/internal in data */
    };

    class PortMapHelper {
    public:
        PortMapHelper(const MemoryPtr& from, const std::deque<MemoryPtr>& to, const dnnl::engine& eng, int src_port, int dst_port);
        ~PortMapHelper() = default;
        void execute(dnnl::stream& strm);

        void update(ProxyMemoryMngrPtr dstMemMngr);

        int src_port;
        int dst_port;

    private:
        void redefineTo();

        MemoryPtr srcMemPtr;
        std::deque<MemoryPtr> dstMemPtrs;
    };

    ExtensionManager::Ptr ext_mng;
    Graph subGraphThen;
    Graph subGraphElse;
    std::vector<std::deque<MemoryPtr>> inputMemThen, inputMemElse;
    std::deque<MemoryPtr> outputMemThen, outputMemElse;

    std::vector<std::shared_ptr<PortMapHelper>>
        beforeThenMappers,
        beforeElseMappers,
        afterThenMappers,
        afterElseMappers;

    std::vector<PortMap>
        thenInputPortMap,
        thenOutputPortMap,
        elseInputPortMap,
        elseOutputPortMap;

    const std::shared_ptr<ov::Node> ovOp;

    std::unordered_map<int, ProxyMemoryMngrPtr> outputMemMngrs;
    std::unordered_map<int, ProxyMemoryMngrPtr> inputThenMemMngrs, inputElseMemMngrs;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
