// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <mkldnn_graph.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class PortMapHelper {
public:
    virtual ~PortMapHelper() = default;
    virtual void execute(int n_iter, mkldnn::stream strm) = 0;
protected:
    std::vector<mkldnn::reorder> reorders;
    std::vector<mkldnn::memory> mem_holder;
    int iter_count;
};

class MKLDNNTensorIteratorNode : public MKLDNNNode {
public:
    MKLDNNTensorIteratorNode(InferenceEngine::CNNLayerPtr layer, const mkldnn::engine& eng, int socket);
    ~MKLDNNTensorIteratorNode() override = default;

    void initSupportedPrimitiveDescriptors() override;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;

    void setExtManager(const MKLDNNExtensionManager::Ptr& extMgr) { ext_mng = extMgr; }
private:
    static Register<MKLDNNTensorIteratorNode> reg;

    int n_iter = 0;

    MKLDNNExtensionManager::Ptr ext_mng;
    MKLDNNGraph sub_graph;
    std::vector<MKLDNNMemoryPtr> input_mem, output_mem;

    std::vector<std::shared_ptr<PortMapHelper>> in_port_mappers, out_port_mappers;
};

}  // namespace MKLDNNPlugin
