// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNRNN : public MKLDNNNode {
public:
    MKLDNNRNN(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                          const std::vector<InferenceEngine::TensorDesc>& outputDesc) override;

    void execute(mkldnn::stream strm) override;

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

    // Internal attributes
    ptrdiff_t N = 0;   /**< Batch value */
    ptrdiff_t T = 0;   /**< Sequence value */
    ptrdiff_t DC = 0;  /**< Input data channel size */
    ptrdiff_t SC = 0;  /**< State channel size value */
    ptrdiff_t G = 0;   /**< Gate size. LSTM - 4, GRU - 3, RNN - 1 */
    ptrdiff_t Gb = 0;  /**< Gate size for biases. Gb = GRU_lbr ? G+1 : G */
    ptrdiff_t S = 2;   /**< Num of state. LSTM - 2, GRU & RNN - 1 */
    const ptrdiff_t L = 1;   /**< What is it??. Constant for mkldnn impl */
    const ptrdiff_t D = 1;   /**< Num of direction. 1 or 2 */

    std::vector<MKLDNNMemoryDesc> in_data_d;
    std::vector<MKLDNNMemoryDesc> out_data_d;

    enum RNNInOutKind {
        Layer       = 0,
        HiddenState = 1,
        CellState   = 2
    };

    MKLDNNMemoryDesc w_data_d;
    MKLDNNMemoryDesc w_state_d;
    MKLDNNMemoryDesc w_bias_d;

    std::vector<size_t > in_data_dims;
    std::vector<size_t > out_data_dims;

    size_t wIdx = 0;
    size_t rIdx = 0;
    size_t bIdx = 0;

    static const std::map<InferenceEngine::Precision, InferenceEngine::Precision> weightsByLayerPrec;
};

}  // namespace MKLDNNPlugin
