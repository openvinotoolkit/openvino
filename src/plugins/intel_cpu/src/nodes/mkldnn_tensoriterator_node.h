// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_node.h>
#include <mkldnn_graph.h>
#include <string>
#include <memory>
#include <vector>
#include <common/memory_desc_wrapper.hpp>

namespace MKLDNNPlugin {

struct PortMap {
    // Data map rule
    int m_from; /**< Index of external data from ins/outs fields of node */
    int m_to;   /**< Index of internal data in iterator body */

    // Iteration rule
    int m_axis;      /**< Axis to iterate throught */
    int m_stride;    /**< Stride to iterate throught */
    int m_start;     /**< Start index of iteration range */
    int m_end;       /**< Last index of iteration range  */
    int m_part_size; /**< Part size which will be transfered to body subnetwork */
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
    mkldnn::reorder m_reorder;
    mkldnn::memory m_mem_holder_src;
    mkldnn::memory m_mem_holder_dst;
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
    mkldnn::memory m_mem_holder;
};


/**
 * Class for storing intermediate output buffer state for dynamism when we don't know
 * final output shape but we should concatenate output after each iteration
 */
class DynamicBuffer {
public:
    DynamicBuffer(const MKLDNNMemoryPtr &from, const MKLDNNMemoryPtr &to, const PortMap &map_rule);
    ~DynamicBuffer() = default;

    void execute(const mkldnn::engine& eng, const int iter);
    void transfer(MKLDNNNode* node);

private:
    void init(const mkldnn::engine& eng);

    /* methods for resize and refill buffer */
    std::shared_ptr<mkldnn::memory> create_buffer(const mkldnn::engine& eng);
    void move_buffer(std::shared_ptr<mkldnn::memory> new_buffer);
    void move_data();

    static inline void copy(const uint8_t* src, uint8_t* dst, const size_t src_stride, const size_t dst_stride, const size_t count, const size_t len);
    static uint8_t* get_ptr(mkldnn::memory& prim);

    size_t m_elem_size = 0lu;
    ptrdiff_t m_chunk_offset_in_byte = 0;
    ptrdiff_t m_buffer_offset_in_byte = 0;

    MKLDNNMemoryPtr m_from;
    MKLDNNMemoryPtr m_to;
    PortMap m_map_rule;

    size_t m_len = 1;
    size_t m_count = 1;

    std::shared_ptr<mkldnn::memory> m_mem_holder_buffer;
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

    void setExtManager(const MKLDNNExtensionManager::Ptr& extMgr) { m_ext_mng = extMgr; }

protected:
    //  needShapeInfer() should return false
    //  because we cannot resolve the output dimensions before the inference is completed
    bool needShapeInfer() const override { return false; };

    bool needPrepareParams() const override;
    void prepareParams() override;
    void executeDynamicImpl(mkldnn::stream strm) override;

private:
    void prepareInputPorts();
    void prepareOutputPorts();
    void prepareBackEdges();
    void prepareDynamicBackEdges();
    void prepareDynamicBuffers();
    void prepareLoopBodyCurrentIteration();
    void prepareContinueCond();
    void prepareInitialCond();
    void prepareTripCount();

    /* Dynamic support */
    void reshapeSubgraphInput();
    void reshapeAndFillOutput(mkldnn::stream strm);

    int m_n_iter = 0;

    MKLDNNExtensionManager::Ptr m_ext_mng;
    MKLDNNGraph m_sub_graph;
    std::vector<MKLDNNMemoryPtr> m_input_mem, m_output_mem;

    std::vector<std::shared_ptr<PortMapHelper>>
        m_first_mappers,   /// < Applied once before loop
        m_last_mappers,    /// < Applied once after loop
        m_before_mappers,  /// < Applied before each iteration
        m_after_mappers,   /// < Applied after each iteration
        m_back_mappers;    /// < Applied before each iteration for dynamic shapes

    std::shared_ptr<PortChecker>
        m_trip_count_check,      /// < Perform check of trip count value. value >= -1
        m_initial_cond_check,    /// < Perform check of initial continue condition value. value [0, 1]
        m_continue_cond_check;   /// < Perform check of continue condition value of body. value [0, 1]

    std::vector<std::shared_ptr<DynamicBuffer>> m_buffers;

    std::vector<PortMap> m_inputPortMap;  //!< Input ports map
    std::vector<PortMap> m_outputPortMap;  //!< Output ports map
    std::vector<PortMap> m_backEdges;  //!< Back edges map

    std::vector<int> m_loopBodyCurrentIterationIdx;
    int m_loopBodyConditionOutputIdx = -1;
    int m_loopTripCountIdx = -1;
    int m_loopExecutionConditionIdx = -1;

    int m_lastUsedTripCount = -1;
    bool m_lastUsedCond = false;

    const std::shared_ptr<ngraph::Node> m_ngraphOp;
};

}  // namespace MKLDNNPlugin
