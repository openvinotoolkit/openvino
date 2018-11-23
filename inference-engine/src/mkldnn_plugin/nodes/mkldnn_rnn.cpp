// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_rnn.h"
#include "mkldnn_extension_utils.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>

#include <string>
#include <utility>

using namespace mkldnn;
using namespace InferenceEngine;

namespace MKLDNNPlugin {

MKLDNNRNN::MKLDNNRNN(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng) : MKLDNNNode(layer, eng) {}

bool MKLDNNRNN::created() const {
    return getType() == RNN;
}

void MKLDNNRNN::getSupportedDescriptors() {
    if (!descs.empty()) return;
    auto rnnLayer = std::dynamic_pointer_cast<RNNLayer>(getCnnLayer());

    if (!rnnLayer)
        THROW_IE_EXCEPTION << "Wrong RNN layer representation. Cannot cast to RNNLayer.";

    if (rnnLayer->cellType == LSTM)
        cellr_type = LSTM;
    else
        THROW_IE_EXCEPTION << "RNN layer supports only LSTM like cell";

    swap_state = rnnLayer->params["swap_state"] == "YES";

    if (rnnLayer->_axis == 0)
        nativeOrder = true;
    else if (rnnLayer->_axis == 1)
        nativeOrder = false;
    else
        THROW_IE_EXCEPTION << "RNN layer supports only sequence axis == 1";

    auto &ins = rnnLayer->insData;
    auto &outs = rnnLayer->outData;

    if (ins.size() != 3 && ins.size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input ports for layer " << getName();
    if (outs.size() != 3 && outs.size() !=1)
        THROW_IE_EXCEPTION << "Incorrect number of output ports for layer " << getName();

    auto in_data_dims = getParentEdgeAt(0)->getDims();
    auto out_data_dims = getChildEdgeAt(0)->getDims();

    if (in_data_dims.ndims() != 3 || out_data_dims.ndims() != 3)
        THROW_IE_EXCEPTION << "Incorrect shape of input/output ports for layer " << getName();

    if (!nativeOrder) {
        std::swap(in_data_dims[0], in_data_dims[1]);
        std::swap(out_data_dims[0], out_data_dims[1]);
    }

    // IE specific order
    seq       = in_data_dims[0];
    batch     = in_data_dims[1];
    data_len  = in_data_dims[2];
    state_len = out_data_dims[2];

    const int N = batch;
    const int T = seq;
    const int G = num_gates;
    const int DC = data_len;
    const int SC = state_len;
    const int L = 1;  // What is a L ??
    const int D = 1;
    const int S = 2;

    if (out_data_dims != MKLDNNDims {T, N, SC})
        THROW_IE_EXCEPTION << "Incorrect shape of input/output ports for layer " << getName();

    MKLDNNDims state_dims {batch, state_len};

    if (ins.size() == 3) {
        auto state_dims1 = getParentEdgeAt(1)->getDims();
        auto stats_dims2 = getParentEdgeAt(2)->getDims();

        if (state_dims1 != state_dims || stats_dims2 != state_dims)
            THROW_IE_EXCEPTION << "Incorrect shape of state ports for layer " << getName();

        in_state_d = {{L, D, S, N, SC}, memory::f32, memory::ldsnc};
    }

    if (outs.size() == 3) {
        auto state_dims1 = getChildEdgeAt(1)->getDims();
        auto stats_dims2 = getChildEdgeAt(2)->getDims();

        if (state_dims1 != state_dims || stats_dims2 != state_dims)
            THROW_IE_EXCEPTION << "Incorrect shape of state ports for layer " << getName();

        out_state_d = {{L, D, S, N, SC}, memory::f32, memory::ldsnc};
    }

    auto blobs = rnnLayer->blobs;
    Blob::Ptr weights, bias;
    if (blobs.find("weights") != blobs.end()) weights = blobs["weights"];
    if (blobs.find("biases") != blobs.end()) bias = blobs["biases"];

    if (!weights)
        THROW_IE_EXCEPTION << "RNN Layer. Weights do not present.";

    if (weights->size() != G*SC*(SC+DC))
        THROW_IE_EXCEPTION << "RNN Layer. Weights size is not correct. Expected size:" << G*SC*(SC+DC);

    w_data_d  = {{L, D, DC, G, SC}, memory::f32, memory::ldigo};
    w_state_d = {{L, D, SC, G, SC}, memory::f32, memory::ldigo};

    if (bias && bias->size() != G*SC)
        THROW_IE_EXCEPTION << "RNN Layer. Biases size is not correct. Expected size:" << G*SC;

    if (bias)
        w_bias_d = {{L, D, G, SC}, memory::f32, memory::ldgo};

    // Try to create descriptor and corresponding configuration
    in_data_d = {in_data_dims, memory::f32, memory::tnc};
    out_data_d = {out_data_dims, memory::f32, memory::tnc};

    std::vector<TensorDesc> in_candidate;
    if (nativeOrder)
        in_candidate.push_back(in_data_d);
    else
        in_candidate.push_back(MKLDNNMemoryDesc{{N, T, DC}, memory::f32, memory::ntc});

    if (ins.size() == 3) {
        in_candidate.emplace_back(MKLDNNMemoryDesc {state_dims, memory::f32, memory::nc});
        in_candidate.emplace_back(MKLDNNMemoryDesc {state_dims, memory::f32, memory::nc});
    }

    std::vector<TensorDesc> out_candidate;
    if (nativeOrder)
        out_candidate.push_back(out_data_d);
    else
        out_candidate.push_back(MKLDNNMemoryDesc{{N, T, SC}, memory::f32, memory::ntc});

    if (outs.size() == 3) {
        out_candidate.emplace_back(MKLDNNMemoryDesc {state_dims, memory::f32, memory::nc});
        out_candidate.emplace_back(MKLDNNMemoryDesc {state_dims, memory::f32, memory::nc});
    }

    createDescriptor(in_candidate, out_candidate);
}

void MKLDNNRNN::createDescriptor(const std::vector<TensorDesc> &inputDesc,
                                 const std::vector<TensorDesc> &outputDesc) {
    MKLDNNDescriptor desc(std::shared_ptr<rnn_forward::desc>(
            new rnn_forward::desc(forward_scoring,
                    {algorithm::vanilla_lstm, algorithm::eltwise_tanh },
                    unidirectional,
                    /* In Data       */ in_data_d,
                    /* In State      */ in_state_d,
                    /* Weights data  */ w_data_d,
                    /* Weights state */ w_state_d,
                    /* Bias          */ w_bias_d,
                    /* Out Data      */ out_data_d,
                    /* Out State     */ out_state_d)));
    descs.push_back(desc);

    // Fill supported config
    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = false;
    for (size_t i = 0; i < inputDesc.size(); i++) {
        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;
        dataConfig.desc = inputDesc[i];
        config.inConfs.push_back(dataConfig);
    }

    for (size_t i = 0; i < outputDesc.size(); i++) {
        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;
        dataConfig.desc = outputDesc[i];
        config.outConfs.push_back(dataConfig);
    }

    supportedPrimitiveDescriptors.push_back({config, ref_any});
}

void MKLDNNRNN::createPrimitive() {
    if (prim) return;

    std::shared_ptr<rnn_forward::desc> d = descs[0];
    rnn_forward::primitive_desc pd(*d, getEngine());

    auto src_data_mem = std::make_shared<MKLDNNMemory>(getEngine());
    src_data_mem->Create(in_data_d, getParentEdgeAt(0)->getMemoryPtr()->GetData());
    internalBlobMemory.push_back(src_data_mem);

    auto dst_data_mem = std::make_shared<MKLDNNMemory>(getEngine());
    dst_data_mem->Create(out_data_d, getChildEdgeAt(0)->getMemoryPtr()->GetData());
    internalBlobMemory.push_back(dst_data_mem);

    // create weight blobs (data and state part)
    auto w_data_mem = std::make_shared<MKLDNNMemory>(getEngine());
    w_data_mem->Create(w_data_d);
    internalBlobMemory.push_back(w_data_mem);

    auto w_state_mem = std::make_shared<MKLDNNMemory>(getEngine());
    w_state_mem->Create(w_state_d);
    internalBlobMemory.push_back(w_state_mem);

    auto w_bias_mem = std::make_shared<MKLDNNMemory>(getEngine());
    w_bias_mem->Create(w_bias_d);
    internalBlobMemory.push_back(w_bias_mem);

    {
        /* Copy Weight data
         *
         * IE format:
         *   W - [gates, out_state_size, in_data_size + in_state_size]
         *   B - [gates, out_state_size]
         *
         * MKLDNN format:
         *   W - [1, 1, in_date_size,  gates, out_state_size]
         *   R - [1, 1, in_state_size, gates, out_state_size]
         *   B - [gates, out_state_size]
         *
         *   Gate order
         *   Caffe - IFOC, ONNX   - IOFC
         *   IE    - FICO, mkldnn - FIOC
         *
         */
        // FICO -> FIOC
        const int gate_map[] = {0, 1, 3, 2};

        auto ie_w_ptr = getCnnLayer()->blobs["weights"]->buffer().as<const float*>();
        auto w_ptr = static_cast<float*>(w_data_mem->GetData());
        auto r_ptr = static_cast<float*>(w_state_mem->GetData());
        const int step = state_len * num_gates;

        for (int g = 0; g < num_gates; g++) {
            for (int out_i = 0; out_i < state_len; out_i++) {
                float *l_w_ptr = w_ptr + gate_map[g]*state_len + out_i;
                float *l_r_ptr = r_ptr + gate_map[g]*state_len + out_i;
                for (int in_i = 0; in_i < data_len; in_i++) {
                    *l_w_ptr = *ie_w_ptr;
                    ie_w_ptr++;
                    l_w_ptr += step;
                }

                for (int in_i = 0; in_i < state_len; in_i++) {
                    *l_r_ptr = *ie_w_ptr;
                    ie_w_ptr++;
                    l_r_ptr += step;
                }
            }
        }

        if (w_bias_d) {
            auto ie_b_ptr = getCnnLayer()->blobs["biases"]->buffer().as<const float*>();
            auto b_ptr = static_cast<float*>(w_bias_mem->GetData());
            for (int g = 0; g < num_gates; g++) {
                float *l_b_ptr = b_ptr + gate_map[g]*state_len;
                for (int out_i = 0; out_i < state_len; out_i++) {
                    *l_b_ptr = *ie_b_ptr;
                    ie_b_ptr++;
                    l_b_ptr++;
                }
            }
        }
    }

    auto src_state_mem = std::make_shared<MKLDNNMemory>(getEngine());
    src_state_mem->Create(in_state_d);
    internalBlobMemory.push_back(src_state_mem);
    if (in_state_d) {
        /* create copy/concat primitive */
        auto src_stat_1 = getParentEdgeAt(1)->getMemory().GetPrimitive();
        auto src_stat_2 = getParentEdgeAt(2)->getMemory().GetPrimitive();

        auto low_half_state_mem = std::make_shared<MKLDNNMemory>(getEngine());
        low_half_state_mem->Create(
                src_stat_1.get_primitive_desc().desc(),
                src_state_mem->GetPrimitive().get_data_handle());
        internalBlobMemory.push_back(low_half_state_mem);

        auto high_half_state_mem = std::make_shared<MKLDNNMemory>(getEngine());
        high_half_state_mem->Create(
                src_stat_2.get_primitive_desc().desc(),
                static_cast<uint8_t*>(src_state_mem->GetPrimitive().get_data_handle()) +
                src_stat_1.get_primitive_desc().get_size());
        internalBlobMemory.push_back(high_half_state_mem);

        if (!swap_state) {
            exec_before.emplace_back(src_stat_1, low_half_state_mem->GetPrimitive());
            exec_before.emplace_back(src_stat_2, high_half_state_mem->GetPrimitive());
        } else {
            exec_before.emplace_back(src_stat_2, low_half_state_mem->GetPrimitive());
            exec_before.emplace_back(src_stat_1, high_half_state_mem->GetPrimitive());
        }
    }

    auto dst_state_mem = std::make_shared<MKLDNNMemory>(getEngine());
    dst_state_mem->Create(out_state_d);
    internalBlobMemory.push_back(dst_state_mem);
    if (out_state_d) {
        /* create copy/split primitive */
        auto dst_stat_1 = getChildEdgeAt(1)->getMemory().GetPrimitive();
        auto dst_stat_2 = getChildEdgeAt(2)->getMemory().GetPrimitive();

        auto low_half_state_mem = std::make_shared<MKLDNNMemory>(getEngine());
        low_half_state_mem->Create(
                dst_stat_1.get_primitive_desc().desc(),
                src_state_mem->GetPrimitive().get_data_handle());
        internalBlobMemory.push_back(low_half_state_mem);

        auto high_half_state_mem = std::make_shared<MKLDNNMemory>(getEngine());
        high_half_state_mem->Create(
                dst_stat_2.get_primitive_desc().desc(),
                static_cast<uint8_t*>(src_state_mem->GetPrimitive().get_data_handle()) +
                        dst_stat_1.get_primitive_desc().get_size());
        internalBlobMemory.push_back(high_half_state_mem);

        exec_after.emplace_back(low_half_state_mem->GetPrimitive(),  dst_stat_1);
        exec_after.emplace_back(high_half_state_mem->GetPrimitive(), dst_stat_2);
    }

    auto workspace_mem = std::make_shared<MKLDNNMemory>(getEngine());
    workspace_mem->Create({}, memory::f32, memory::format_undef, nullptr);  // stub, not in use
    internalBlobMemory.push_back(workspace_mem);

    auto p = new rnn_forward(pd,
            /* In Data       */ src_data_mem ->GetPrimitive(),
            /* In State      */ src_state_mem->GetPrimitive(),
            /* Weights data  */ w_data_mem   ->GetPrimitive(),
            /* Weights state */ w_state_mem  ->GetPrimitive(),
            /* Bias          */ w_bias_mem   ->GetPrimitive(),
            /* Out Data      */ dst_data_mem ->GetPrimitive(),
            /* Out State     */ dst_state_mem->GetPrimitive(),
            /* Workspace     */ workspace_mem->GetPrimitive());

    prim.reset(p);
}

void MKLDNNRNN::execute(mkldnn::stream strm) {
    if (!exec_before.empty())
        strm.submit({exec_before.begin(), exec_before.end()});

    if (prim)
        strm.submit({*prim});

    if (!exec_after.empty())
        strm.submit({exec_after.begin(), exec_after.end()});
}

}  // namespace MKLDNNPlugin
