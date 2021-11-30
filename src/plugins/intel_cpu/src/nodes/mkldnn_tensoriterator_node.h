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


/**
 * Class for storing intermediate output buffer state for dynamism when we don't know
 * final output shape but we should concatenate output after each iteration
 */
class DynamicBuffer {
public:
    DynamicBuffer(const MKLDNNMemoryPtr &from, const MKLDNNMemoryPtr &to, const PortMap &map_rule);
    ~DynamicBuffer() = default;

    void execute(mkldnn::stream strm, const mkldnn::engine& eng, const int iter);
    void transfer(mkldnn::stream strm, const mkldnn::engine& eng, MKLDNNNode* node);

private:
    void init(mkldnn::stream strm, const mkldnn::engine& eng);
    void copy(mkldnn::stream strm, mkldnn::memory& src, mkldnn::memory& dst);
    void overwrite(mkldnn::stream strm, const mkldnn::engine& eng);

    /* methods for resize and refill buffer */
    std::shared_ptr<mkldnn::memory> create_buffer(const mkldnn::engine& eng);
    void move_buffer(mkldnn::stream strm, const mkldnn::engine& eng, std::shared_ptr<mkldnn::memory> new_buffer);
    void move_data(mkldnn::stream strm, const mkldnn::engine& eng);

    size_t elem_size = 0lu;
    ptrdiff_t chunk_stride_in_byte = 0;
    ptrdiff_t chunk_offset_in_byte = 0;

    MKLDNNMemoryPtr from;
    MKLDNNMemoryPtr to;
    PortMap map_rule;

    std::shared_ptr<mkldnn::memory> mem_holder_buffer;
};

class MKLDNNTensorIteratorNode : public MKLDNNNode {
public:
    MKLDNNTensorIteratorNode(const std::shared_ptr<ov::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void initSupportedPrimitiveDescriptors() override;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;

    void setExtManager(const MKLDNNExtensionManager::Ptr& extMgr) { ext_mng = extMgr; }

    //  needShapeInfer() should return false
    //  because we cannot know which output dimensions will be without inference
    bool needShapeInfer() const override { return false; };

    bool needPrepareParams() const override;
    void prepareParams() override;
    void executeDynamicImpl(mkldnn::stream strm) override;

private:
    void prepareInputPorts(const mkldnn::engine& eng);
    void prepareOutputPorts(const mkldnn::engine& eng);
    void prepareBackEdges(const mkldnn::engine& eng);
    void prepareDynamicBackEdges(const mkldnn::engine& eng);
    void prepareDynamicBuffers();
    void prepareLoopBodyCurrentIteration(const mkldnn::engine& eng);
    void prepareContinueCond();
    void prepareInitialCond();
    void prepareTripCount();

    int getNumIteration(const std::vector<PortMap>& inputPortMap, const std::vector<PortMap>& outputPortMap);

    /* Dynamic support */
    void reshapeSubgraphInput();
    void reshapeAndFillOutput(mkldnn::stream strm, const mkldnn::engine& eng);

    int n_iter = 0;

    MKLDNNExtensionManager::Ptr ext_mng;
    MKLDNNGraph sub_graph;
    std::vector<MKLDNNMemoryPtr> input_mem, output_mem;

    std::vector<std::shared_ptr<PortMapHelper>>
        first_mappers,   /// < Applied once before loop
        last_mappers,    /// < Applied once after loop
        before_mappers,  /// < Applied before each iteration
        after_mappers,   /// < Applied after each iteration
        back_mappers;    /// < Applied before each iteration for dynamic shapes

    std::shared_ptr<PortChecker>
        trip_count_check,      /// < Perform check of trip count value. value >= -1
        initial_cond_check,    /// < Perform check of initial continue condition value. value [0, 1]
        continue_cond_check;   /// < Perform check of continue condition value of body. value [0, 1]

    std::vector<std::shared_ptr<DynamicBuffer>> buffers;

    std::vector<PortMap> inputPortMap;  //!< Input ports map
    std::vector<PortMap> outputPortMap;  //!< Output ports map
    std::vector<PortMap> backEdges;  //!< Back edges map

    std::vector<int> loopBodyCurrentIterationIdx;
    int loopBodyConditionOutputIdx = -1;
    int loopTripCountIdx = -1;
    int loopExecutionConditionIdx = -1;

    int lastTripCount = -1;
    bool lastCond = false;

    const std::shared_ptr<ngraph::Node> ngraphOp;
};

}  // namespace MKLDNNPlugin
