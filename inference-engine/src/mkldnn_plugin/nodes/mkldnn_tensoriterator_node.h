// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_node.h>
#include <mkldnn_graph.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

struct PortMap {
    // Data map rule
    int from; /**< Index of external data from ins/outs fields of node */
    int to;   /**< Index of internal data in iterator body */

    // Iteration rule
    int axis;      /**< Axis to iterate throught */
    int stride;    /**< Stride to iterate throught */
    int start;     /**< Start index of iteration range */
    int end;       /**< Last index of iteration range  */
    int part_size; /**< Part size which will be transfered to body subnetwork */
};

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
    mkldnn::reorder reorder;
    mkldnn::memory mem_holder_src;
    mkldnn::memory mem_holder_dst;
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
    mkldnn::memory mem_holder;
};


class MKLDNNTensorIteratorNode : public MKLDNNNode {
public:
    MKLDNNTensorIteratorNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
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

    std::vector<PortMap> inputPortMap;  //!< Input ports map
    std::vector<PortMap> outputPortMap;  //!< Output ports map
    std::vector<PortMap> backEdges;  //!< Back edges map

    std::vector<int> loopBodyCurrentIterationIdx;
    int loopBodyConditionOutputIdx = -1;
    int loopTripCountIdx = -1;
    int loopExecutionConditionIdx = -1;

    InferenceEngine::LayerConfig config;

    const std::shared_ptr<ngraph::Node> ngraphOp;
};

}  // namespace MKLDNNPlugin
