// Copyright (C) 2018-2020 Intel Corporation
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

/**
 * Functor interface to perform some action with pointed tensors (captured in constructor)
 * Generally it's read, write or move data from specified tensors.
 * Action may depends on iteration index.
 */
class PortMapHelper {
public:
    virtual ~PortMapHelper() = default;
    virtual void execute(mkldnn::stream strm, int n_iter = -1) = 0;
protected:
    std::vector<mkldnn::reorder> reorders;
    std::vector<mkldnn::memory> mem_holder;
};


/**
 * Functor interface to perform check of data tensor (captured in constructor)
 * Information extracted as int. Meaning of returned value is specific for
 * particular type of checker.
 */
class PortChecker {
public:
    virtual ~PortChecker() = default;
    virtual int getStatus() = 0;
protected:
    std::vector<mkldnn::memory> mem_holder;
};


class MKLDNNTensorIteratorNode : public MKLDNNNode {
public:
    MKLDNNTensorIteratorNode(InferenceEngine::CNNLayerPtr layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNTensorIteratorNode() override = default;

    void initSupportedPrimitiveDescriptors() override;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;

    void setExtManager(const MKLDNNExtensionManager::Ptr& extMgr) { ext_mng = extMgr; }

private:
    int n_iter = 0;

    MKLDNNExtensionManager::Ptr ext_mng;
    MKLDNNGraph sub_graph;
    std::vector<MKLDNNMemoryPtr> input_mem, output_mem;

    std::vector<std::shared_ptr<PortMapHelper>>
        first_mappers,   /// < Applied once before loop
        last_mappers,    /// < Applied once after loop
        before_mappers,  /// < Applied before each iteration
        after_mappers;   /// < Applied after each iteration

    std::shared_ptr<PortChecker>
        trip_count_check,      /// < Perform check of trip count value. value >= -1
        initial_cond_check,   /// < Perform check of initial continue condition value. value [0, 1]
        continue_cond_check;  /// < Perform check of continue condition value of body. value [0, 1]
};

}  // namespace MKLDNNPlugin
