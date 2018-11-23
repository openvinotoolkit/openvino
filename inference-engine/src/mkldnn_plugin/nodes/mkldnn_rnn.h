// Copyright (C) 2018 Intel Corporation
//
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
    static Register<MKLDNNRNN> reg;

    InferenceEngine::CellType cellr_type = InferenceEngine::CellType::LSTM;
    /** Native order if [batch, seq, data], other case is [seq, batch, data] */
    bool nativeOrder = true;
    bool swap_state = false;

    int batch = 0;
    int seq = 0;
    int data_len = 0;
    int state_len = 0;
    const size_t num_gates = 4;

    MKLDNNMemoryDesc in_data_d;
    MKLDNNMemoryDesc out_data_d;

    MKLDNNMemoryDesc in_state_d;
    MKLDNNMemoryDesc out_state_d;

    MKLDNNMemoryDesc w_data_d;
    MKLDNNMemoryDesc w_state_d;
    MKLDNNMemoryDesc w_bias_d;

    std::vector<mkldnn::reorder> exec_before;
    std::vector<mkldnn::reorder> exec_after;
};

}  // namespace MKLDNNPlugin

