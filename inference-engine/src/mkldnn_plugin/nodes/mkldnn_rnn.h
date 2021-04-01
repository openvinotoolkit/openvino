// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNRNN : public MKLDNNNode {
public:
    MKLDNNRNN(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNRNN() override = default;

    void getSupportedDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                          const std::vector<InferenceEngine::TensorDesc>& outputDesc) override;

    void execute(mkldnn::stream strm) override;

private:
    void fillCellDesc();
    void fillSeqDesc();

private:
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

    MKLDNNMemoryDesc in_data_d;
    MKLDNNMemoryDesc out_data_d;

    std::vector<MKLDNNMemoryDesc> in_states_d;
    std::vector<MKLDNNMemoryDesc> out_states_d;

    MKLDNNMemoryDesc w_data_d;
    MKLDNNMemoryDesc w_state_d;
    MKLDNNMemoryDesc w_bias_d;

    // List of in/out reorders if required
    std::vector<mkldnn::reorder> exec_before;
    std::vector<mkldnn::reorder> exec_after;
};

}  // namespace MKLDNNPlugin

