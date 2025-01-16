// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"
#include "common/dnnl_executor.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"


namespace ov {
namespace intel_cpu {
namespace node {

class RNN : public Node {
public:
    RNN(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    static bool isCell(const std::shared_ptr<const ov::Node>& op);
    static bool testNativeOrder(const std::shared_ptr<const ov::Node>& op);
    void getSupportedDescriptors() override;
    std::shared_ptr<MemoryDesc> getSrcMemDesc(const dnnl::primitive_desc& prim_desc, size_t idx) const override;
    std::shared_ptr<MemoryDesc> getDstMemDesc(const dnnl::primitive_desc& prim_desc, size_t idx) const override;
    bool created() const override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;
    std::shared_ptr<dnnl::primitive_attr> initPrimitiveAttr() override;

    void execute(dnnl::stream strm) override;

    inline bool hasNativeOrder() const {
        return nativeOrder;
    }

    void cleanup() override;

    enum InOutKind {
        Layer       = 0,
        HiddenState = 1,
        CellState   = 2,
        Attention   = 2
    };

protected:
    void prepareParams() override;
    void executeDynamicImpl(dnnl::stream strm) override;

private:
    void configurePortDataTypes();
    void initCell();
    void initSequence();
    void fillCellDesc();
    void fillSequenceDesc();
    void fillDescs();
    bool verifyWeightsPrecision(const ov::element::Type& layerPrec,
                                const ov::element::Type& weightsPrec);

    template <ov::element::Type_t ET>
    void fillWeights();
    template <ov::element::Type_t ET>
    void fillBiases();

    void copyWeightsData();

    void prepareMemory(const DnnlMemoryDescPtr& intDesc, size_t indx) override;
    class RnnDnnlExecutor : public DnnlExecutor {
        public:
            RnnDnnlExecutor(const dnnl::primitive_desc& pd);

            DnnlMemoryDescPtr getWeightIterDesc() const {
                return wghts_iter_md;
            }

            DnnlMemoryDescPtr getBiasDesc() const {
                return bias_md;
            }

        private:
            DnnlMemoryDescPtr wghts_iter_md;
            DnnlMemoryDescPtr bias_md;
    };

    using executorPtr = std::shared_ptr<RnnDnnlExecutor>;
    executorPtr execPtr = nullptr;

    /** Specify mode Cell or Seq. true - Cell, false - Seq */
    bool is_cell = false;

    bool is_augru = false;

    /** Native order if [batch, seq, data], other case is [seq, batch, data] */
    bool nativeOrder = true;

    /** Direction of iteration through sequence dimension */
    dnnl::rnn_direction direction = dnnl::rnn_direction::unidirectional_left2right;

    /** RNN Cell type (type/activation_alg/clip)*/
    dnnl::algorithm cell_type = dnnl::algorithm::undef;

    /** activation type for vanilla RNN cell */
    dnnl::algorithm cell_act = dnnl::algorithm::undef;

    struct Interval {
        Interval() = default;

        Interval(Dim min, Dim max) {
            minVal = min;
            maxVal = max;
        }

        bool isStatic() {
            return minVal == maxVal;
        }

        Dim minVal = 0;
        Dim maxVal = 0;
    };
    // Internal attributes
    Interval N;     /**< Batch value */
    Interval N_SEQ; /**< Batch value of the 'sequence_length' input */
    Interval T;     /**< Sequence value */
    size_t DC = 0;  /**< Input data channel size */
    size_t SC = 0;  /**< State channel size value */
    size_t G = 0;   /**< Gate size. LSTM - 4, GRU - 3, RNN - 1 */
    size_t Gb = 0;  /**< Gate size for biases. Gb = GRU_lbr ? G+1 : G */
    size_t S = 2;   /**< Num of state. LSTM - 2, GRU & RNN - 1 */
    const size_t L = 1;   /**< What is it??. Constant for onednn impl */
    const size_t D = 1;   /**< Num of direction. 1 or 2 */

    std::vector<DnnlBlockedMemoryDescPtr> inDataDescs;
    std::vector<DnnlBlockedMemoryDescPtr> outDataDescs;
    std::vector<dnnl::memory::desc> wDescs;

    std::vector<dnnl::memory::data_type> inDataTypes;
    std::vector<dnnl::memory::data_type> outDataTypes;

    const size_t xIdx = 0; // ov -> input X;              dnnl -> src_layer
    const size_t hIdx = 1; // ov -> initial_hidden_state; dnnl -> src_iter_h
    const size_t cIdx = 2; // ov -> initial_cell_state;   dnnl -> src_iter_c
    size_t sIdx = 0;       // ov -> sequence_length;      dnnl -> additional input dimension 't'
                           //                             oneDNN does not support unique t (seq_len) per batch
    size_t wIdx = 0;       // ov -> W;                    dnnl -> weights_layer
    size_t rIdx = 0;       // ov -> R;                    dnnl -> weights_iter
    size_t bIdx = 0;       // ov -> B;                    dnnl -> bias
    size_t aIdx = 0;       // ov -> A:                    dnnl -> attention

    size_t yIdx = 0;       // ov -> Y;                    dnnl -> dst_layer
    size_t hoIdx = 0;      // ov -> Ho;                   dnnl -> dst_iter_h
    size_t coIdx = 0;      // ov -> Co;                   dnnl -> dst_iter_c

    static const std::map<dnnl::memory::data_type, dnnl::memory::data_type> weightsByinputDataType;

    static constexpr size_t optimalBatchSize = 16lu;
    static constexpr size_t batchDimDummyValue = 64lu;

    float inputScale    = 0.f;
    float inputShift    = 0.f;
    std::vector<float> weightsScales;

    const uint64_t* m_gate_map;
    // Need to reorder from the initial memory descs due to limited Reorders set.
    MemoryPtr m_initial_weights[3] = { nullptr, nullptr, nullptr };
    // Need to keep cache objects. Otherwise, they will be erased from the global cache.
    std::unordered_set<MemoryPtr> m_weights_pull;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
