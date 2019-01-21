// Copyright (C) 2018 Intel Corporation
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
    MKLDNNRNN(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng);
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
    static Register<MKLDNNRNN> reg;

    /** Specify mode Cell or Seq. true - Cell, false - Seq */
    bool is_cell = false;

    /** Native order if [batch, seq, data], other case is [seq, batch, data] */
    bool nativeOrder = true;

    /** Direction of iteration through sequence dimension */
    mkldnn::rnn_direction direction = mkldnn::unidirectional;

    // Internal attributes
    int N = 0;   /**< Batch value */
    int T = 0;   /**< Sequence value */
    int DC = 0;  /**< Input data channel size */
    int SC = 0;  /**< State channel size value */
    const int G = 4;   /**< Gate size. 4 for LSTM */
    const int L = 1;   /**< What is it??. Constant for mkldnn impl */
    const int D = 1;   /**< Num of direction. 1 or 2 */
    const int S = 2;   /**< Num of state. 2 for LSTM (hidden and sell state). */

    MKLDNNMemoryDesc in_data_d;
    MKLDNNMemoryDesc out_data_d;

    MKLDNNMemoryDesc in_state_d;
    MKLDNNMemoryDesc out_state_d;

    MKLDNNMemoryDesc w_data_d;
    MKLDNNMemoryDesc w_state_d;
    MKLDNNMemoryDesc w_bias_d;

    // List of in/out reorders if required
    std::vector<mkldnn::reorder> exec_before;
    std::vector<mkldnn::reorder> exec_after;
};

}  // namespace MKLDNNPlugin

