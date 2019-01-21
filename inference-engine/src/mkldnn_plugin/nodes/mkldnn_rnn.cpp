// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_rnn.h"
#include "mkldnn_extension_utils.h"
#include "desc_iterator.hpp"
#include <ie_layers_prv.h>

#include <string>
#include <utility>

using namespace mkldnn;
using namespace InferenceEngine;

namespace MKLDNNPlugin {

template <typename T, typename P>
inline bool one_of(T val, P item) { return val == item; }
template <typename T, typename P, typename... Args>
inline bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

rnn_direction ie2mkl(RNNLayer::Direction &direction) {
    return direction == RNNLayer::RNN_FWD ? unidirectional_left2right
         : direction == RNNLayer::RNN_BWD ? unidirectional_right2left
         : direction == RNNLayer::RNN_BDR ? bidirectional_concat
                                          : unidirectional;
}

MKLDNNRNN::MKLDNNRNN(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng) : MKLDNNNode(layer, eng) {
    is_cell = layer->type == "LSTMCell";
}

bool MKLDNNRNN::created() const {
    return getType() == (is_cell ? LSTMCell : RNN);
}

void MKLDNNRNN::getSupportedDescriptors() {
    if (is_cell)
        fillCellDesc();
    else
        fillSeqDesc();
}

void MKLDNNRNN::fillCellDesc() {
    if (!descs.empty()) return;
    auto cellLayer = std::dynamic_pointer_cast<InferenceEngine::LSTMCell>(getCnnLayer());

    if (!cellLayer)
        THROW_IE_EXCEPTION << "Wrong RNN layer representation. Cannot cast to RNNLayer.";

    auto &ins = cellLayer->insData;
    auto &outs = cellLayer->outData;

    if (ins.size() != 3)
        THROW_IE_EXCEPTION << "Incorrect number of input ports for layer " << getName();
    if (outs.size() != 2)
        THROW_IE_EXCEPTION << "Incorrect number of output ports for layer " << getName();

    auto in_data_dims = getParentEdgeAt(0)->getDims();
    auto in_h_state_dims = getParentEdgeAt(1)->getDims();
    auto in_c_state_dims = getParentEdgeAt(2)->getDims();

    auto out_h_state_dims = getChildEdgeAt(0)->getDims();
    auto out_c_state_dims = getChildEdgeAt(1)->getDims();

    if (in_data_dims.ndims() != 2
        || in_h_state_dims.ndims() != 2
        || in_c_state_dims.ndims() != 2
        || out_h_state_dims.ndims() != 2
        || out_c_state_dims.ndims() != 2)
        THROW_IE_EXCEPTION << "Incorrect shape of input/output ports for layer " << getName();

    T = 1;
    N  = in_data_dims[0];
    DC = in_data_dims[1];
    SC = in_h_state_dims[1];

    // Expected shapes
    MKLDNNDims D_shape {N, DC}, S_shape {N, SC};

    if (in_data_dims != D_shape
        || in_h_state_dims != S_shape
        || in_c_state_dims != S_shape
        || out_h_state_dims != S_shape
        || out_c_state_dims != S_shape)
        THROW_IE_EXCEPTION << "Incorrect shape of input/output ports for layer " << getName();

    auto blobs = cellLayer->blobs;
    Blob::Ptr weights, bias;
    if (blobs.find("weights") != blobs.end()) weights = blobs["weights"];
    if (blobs.find("biases") != blobs.end()) bias = blobs["biases"];

    if (!weights)
        THROW_IE_EXCEPTION << "RNN Layer. Weights do not present.";

    if (weights->size() != G*SC*(SC+DC))
        THROW_IE_EXCEPTION << "RNN Layer. Weights size is not correct. Expected size:" << G*SC*(SC+DC);

    if (bias && bias->size() != G*SC)
        THROW_IE_EXCEPTION << "RNN Layer. Biases size is not correct. Expected size:" << G*SC;

    // Shapes and Attributes are correct. Can start internal stuff initialization.

    in_state_d  = {{L, D, S, N, SC}, memory::f32, memory::ldsnc};
    out_state_d = {{L, D, S, N, SC}, memory::f32, memory::ldsnc};

    in_data_d  = {{T, N, DC}, memory::f32, memory::tnc};;
    out_data_d = {{T, N, SC}, memory::f32, memory::tnc};;

    w_data_d   = {{L, D, DC, G, SC}, memory::f32, memory::ldigo};
    w_state_d  = {{L, D, SC, G, SC}, memory::f32, memory::ldigo};

    if (bias)
        w_bias_d = {{L, D, G, SC}, memory::f32, memory::ldgo};

    std::vector<TensorDesc> in_candidate;
    in_candidate.emplace_back(MKLDNNMemoryDesc {D_shape, memory::f32, memory::nc});
    in_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::f32, memory::nc});
    in_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::f32, memory::nc});

    std::vector<TensorDesc> out_candidate;
    out_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::f32, memory::nc});
    out_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::f32, memory::nc});

    createDescriptor(in_candidate, out_candidate);
}

void MKLDNNRNN::fillSeqDesc() {
    if (!descs.empty()) return;
    auto rnnLayer = std::dynamic_pointer_cast<RNNLayer>(getCnnLayer());

    if (!rnnLayer)
        THROW_IE_EXCEPTION << "Wrong RNN layer representation. Cannot cast to RNNLayer.";

    if (!one_of(rnnLayer->cellType, "LSTM"))
        THROW_IE_EXCEPTION << "RNN layer supports only LSTM like cell";

    if (!one_of(rnnLayer->axis, 0, 1))
        THROW_IE_EXCEPTION << "RNN layer supports only sequence axis 0 or 1";
    nativeOrder = rnnLayer->axis == 0;

    if (!one_of(rnnLayer->direction, RNNLayer::RNN_FWD, RNNLayer::RNN_BWD))
        THROW_IE_EXCEPTION << "RNN layer supports only unidirectional RNN layer";
    direction = ie2mkl(rnnLayer->direction);

    auto &ins = rnnLayer->insData;
    auto &outs = rnnLayer->outData;

    if (!one_of(ins.size(), 3, 1))
        THROW_IE_EXCEPTION << "Incorrect number of input ports for layer " << getName();
    if (!one_of(outs.size(), 3, 1))
        THROW_IE_EXCEPTION << "Incorrect number of output ports for layer " << getName();

    auto in_data_dims = getParentEdgeAt(0)->getDims();
    auto out_data_dims = getChildEdgeAt(0)->getDims();

    if (in_data_dims.ndims() != 3 || out_data_dims.ndims() != 3)
        THROW_IE_EXCEPTION << "Incorrect shape of input/output ports for layer " << getName();

    if (!nativeOrder) {
        std::swap(in_data_dims[0], in_data_dims[1]);
        std::swap(out_data_dims[0], out_data_dims[1]);
    }

    T = in_data_dims[0];
    N = in_data_dims[1];
    DC = in_data_dims[2];
    SC = out_data_dims[2];

    MKLDNNDims ID_shape {T, N, DC}, OD_shape {T, N, SC}, S_shape {N, SC};

    if (out_data_dims != OD_shape)
        THROW_IE_EXCEPTION << "Incorrect shape of input/output ports for layer " << getName();

    if (ins.size() == 3) {
        auto state_dims1 = getParentEdgeAt(1)->getDims();
        auto stats_dims2 = getParentEdgeAt(2)->getDims();

        if (state_dims1 != S_shape || stats_dims2 != S_shape)
            THROW_IE_EXCEPTION << "Incorrect shape of state ports for layer " << getName();

        in_state_d = {{L, D, S, N, SC}, memory::f32, memory::ldsnc};
    }

    if (outs.size() == 3) {
        auto state_dims1 = getChildEdgeAt(1)->getDims();
        auto stats_dims2 = getChildEdgeAt(2)->getDims();

        if (state_dims1 != S_shape || stats_dims2 != S_shape)
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
        in_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::f32, memory::nc});
        in_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::f32, memory::nc});
    }

    std::vector<TensorDesc> out_candidate;
    if (nativeOrder)
        out_candidate.push_back(out_data_d);
    else
        out_candidate.push_back(MKLDNNMemoryDesc{{N, T, SC}, memory::f32, memory::ntc});

    if (outs.size() == 3) {
        out_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::f32, memory::nc});
        out_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::f32, memory::nc});
    }

    createDescriptor(in_candidate, out_candidate);
}

void MKLDNNRNN::createDescriptor(const std::vector<TensorDesc> &inputDesc,
                                 const std::vector<TensorDesc> &outputDesc) {
    MKLDNNDescriptor desc(std::shared_ptr<rnn_forward::desc>(
            new rnn_forward::desc(forward_scoring,
                    {algorithm::vanilla_lstm, algorithm::eltwise_tanh },
                    direction,
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

    auto src_data_mem = getParentEdgeAt(0)->getMemoryPtr();
    auto dst_data_mem = getChildEdgeAt(0)->getMemoryPtr();

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
         *   IE    - FICO, mkldnn - IFCO
         */
        // FICO -> IFCO
        const int gate_map[] = {1, 0, 2, 3};

        auto ie_w_ptr = getCnnLayer()->blobs["weights"]->buffer().as<const float*>();
        auto w_ptr = static_cast<float*>(w_data_mem->GetData());
        auto r_ptr = static_cast<float*>(w_state_mem->GetData());
        const int step = SC * G;

        for (int g = 0; g < G; g++) {
            for (int out_i = 0; out_i < SC; out_i++) {
                float *l_w_ptr = w_ptr + gate_map[g]*SC + out_i;
                float *l_r_ptr = r_ptr + gate_map[g]*SC+ out_i;
                for (int in_i = 0; in_i < DC; in_i++) {
                    *l_w_ptr = *ie_w_ptr;
                    ie_w_ptr++;
                    l_w_ptr += step;
                }

                for (int in_i = 0; in_i < SC; in_i++) {
                    *l_r_ptr = *ie_w_ptr;
                    ie_w_ptr++;
                    l_r_ptr += step;
                }
            }
        }

        if (w_bias_d) {
            auto ie_b_ptr = getCnnLayer()->blobs["biases"]->buffer().as<const float*>();
            auto b_ptr = static_cast<float*>(w_bias_mem->GetData());
            for (int g = 0; g < G; g++) {
                float *l_b_ptr = b_ptr + gate_map[g]*SC;
                for (int out_i = 0; out_i < SC; out_i++) {
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

        exec_before.emplace_back(src_stat_1, low_half_state_mem->GetPrimitive());
        exec_before.emplace_back(src_stat_2, high_half_state_mem->GetPrimitive());
    }

    auto dst_state_mem = std::make_shared<MKLDNNMemory>(getEngine());
    dst_state_mem->Create(out_state_d);
    internalBlobMemory.push_back(dst_state_mem);
    if (out_state_d) {
        int idx_H = is_cell ? 0 : 1;
        int idx_C = is_cell ? 1 : 2;
        /* create copy/split primitive */
        auto dst_stat_1 = getChildEdgeAt(idx_H)->getMemory().GetPrimitive();
        auto dst_stat_2 = getChildEdgeAt(idx_C)->getMemory().GetPrimitive();

        auto low_half_state_mem = std::make_shared<MKLDNNMemory>(getEngine());
        low_half_state_mem->Create(
                dst_stat_1.get_primitive_desc().desc(),
                dst_state_mem->GetPrimitive().get_data_handle());
        internalBlobMemory.push_back(low_half_state_mem);

        auto high_half_state_mem = std::make_shared<MKLDNNMemory>(getEngine());
        high_half_state_mem->Create(
                dst_stat_2.get_primitive_desc().desc(),
                static_cast<uint8_t*>(dst_state_mem->GetPrimitive().get_data_handle()) +
                        dst_stat_1.get_primitive_desc().get_size());
        internalBlobMemory.push_back(high_half_state_mem);


        if (!is_cell) exec_after.emplace_back(low_half_state_mem->GetPrimitive(),  dst_stat_1);
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
