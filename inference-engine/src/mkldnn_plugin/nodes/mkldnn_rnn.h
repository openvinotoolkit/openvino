// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>
#include "memory_desc/dnnl_blocked_memory_desc.h"

namespace MKLDNNPlugin {

class MKLDNNRNN : public MKLDNNNode {
public:
    MKLDNNRNN(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    std::shared_ptr<MemoryDesc> getSrcMemDesc(mkldnn::primitive_desc_iterator& primitive_desc_it, size_t idx) override;
    std::shared_ptr<MemoryDesc> getDstMemDesc(mkldnn::primitive_desc_iterator& primitive_desc_it, size_t idx) override;
    bool created() const override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;

    void execute(mkldnn::stream strm) override;

    inline bool hasNativeOrder() const {
        return nativeOrder;
    }

private:
    void initCell(const std::shared_ptr<ngraph::Node>& op);
    void initSeq(const std::shared_ptr<ngraph::Node>& op);
    void fillCellDesc();
    void fillSeqDesc();
    bool verifyWeightsPrecision(const InferenceEngine::Precision& layerPrec,
                                const InferenceEngine::Precision& weightsPrec);

    template <typename Prec>
    void fillWeights(const int* gate_map, const size_t wIdx, const size_t rIdx);
    template <InferenceEngine::Precision::ePrecision Prec>
    void fillBiases(const int* gate_map);

    void copyWeightsData();

private:
    InferenceEngine::Precision runtimePrecision;
    /** Specify mode Cell or Seq. true - Cell, false - Seq */
    bool is_cell = false;

    /** Native order if [batch, seq, data], other case is [seq, batch, data] */
    bool nativeOrder = true;

    /** Direction of iteration through sequence dimension */
    mkldnn::rnn_direction direction = mkldnn::rnn_direction::unidirectional;

    /** RNN Cell type (type/activation_alg/clip)*/
    mkldnn::algorithm cell_type = mkldnn::algorithm::vanilla_lstm;

    /** activation type for vanilla RNN cell */
    mkldnn::algorithm cell_act = mkldnn::algorithm::eltwise_tanh;

    /** Weights data and state memory format: ldigo or any */
    mkldnn::memory::format_tag w_format = mkldnn::memory::format_tag::any;

    // Internal attributes
    size_t N = 0;   /**< Batch value */
    size_t T = 0;   /**< Sequence value */
    size_t DC = 0;  /**< Input data channel size */
    size_t SC = 0;  /**< State channel size value */
    size_t G = 0;   /**< Gate size. LSTM - 4, GRU - 3, RNN - 1 */
    size_t Gb = 0;  /**< Gate size for biases. Gb = GRU_lbr ? G+1 : G */
    size_t S = 2;   /**< Num of state. LSTM - 2, GRU & RNN - 1 */
    const size_t L = 1;   /**< What is it??. Constant for mkldnn impl */
    const size_t D = 1;   /**< Num of direction. 1 or 2 */

    std::vector<DnnlBlockedMemoryDesc> in_data_d;
    std::vector<DnnlBlockedMemoryDesc> out_data_d;

    enum RNNInOutKind {
        Layer       = 0,
        HiddenState = 1,
        CellState   = 2
    };

    std::vector<size_t > in_data_dims;
    std::vector<size_t > out_data_dims;

    size_t wIdx = 0;
    size_t rIdx = 0;
    size_t bIdx = 0;

    static const std::map<InferenceEngine::Precision, InferenceEngine::Precision> weightsByLayerPrec;
};

}  // namespace MKLDNNPlugin
