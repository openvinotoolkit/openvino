// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "net_pass.h"
#include "ie_layers_prv.h"
#include "graph_tools.hpp"

#include <string>
#include <utility>
#include <memory>
#include <unordered_set>

template <typename T, typename P>
inline bool one_of(T val, P item) { return val == item; }
template <typename T, typename P, typename... Args>
inline bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

namespace InferenceEngine {
namespace NetPass {

inline bool is_full_ranged(const TensorIterator::PortMap& rule, const DataPtr &data) {
    if (!data)
        THROW_IE_EXCEPTION << "Internal error. data == nullptr";

    if (rule.axis == -1 || !one_of(rule.stride, 1, -1))
        return false;

    auto &shape = data->getDims();
    int size = shape[rule.axis];

    int begin = rule.start >= 0 ? rule.start : size + rule.start + 1;
    int end = rule.end >= 0 ? rule.end : size + rule.end + 1;

    return (rule.stride == 1)
        ? begin == 0 && end == size
        : begin == size && end == 0;
}

bool convertToLSTMSequence(CNNLayerPtr cur) {
    if (cur->type != "TensorIterator") return false;
    auto ti = std::dynamic_pointer_cast<TensorIterator>(cur);

    IE_ASSERT(ti) << "Cannot cast object with type TensorIterator to TensorIterator object";

    // Topological order
    std::vector<CNNLayerPtr> all_body_layers;
    CNNNetForestDFS(ti->body.inputs, [&](CNNLayerPtr  current){
        all_body_layers.push_back(current);
    }, false);
    std::reverse(all_body_layers.begin(), all_body_layers.end());

    // Check if body is:  squeeze -> lstm_cell -> unsqueeze
    if (all_body_layers.size() != 3
        || all_body_layers[0]->type != "Reshape"
        || all_body_layers[1]->type != "LSTMCell"
        || all_body_layers[2]->type != "Reshape")
        return false;

    auto &rsp1 = all_body_layers[0];
    auto &lstm = all_body_layers[1];
    auto &rsp2 = all_body_layers[2];

    IE_ASSERT(lstm->insData.size() == 3);  // {data, hidden, cell}
    IE_ASSERT(lstm->outData.size() == 2);  // {hidden, cell}

    if (lstm->insData[0].lock()->creatorLayer.lock() != rsp1 ||
        lstm->outData[0]->inputTo.begin()->second != rsp2)
        return false;

    // Check port mapping
    auto _indx_in = [&] (const std::vector<DataPtr> &scope,  const DataPtr &data) {
        int indx = std::find(scope.begin(), scope.end(), data) - scope.begin();
        return indx == scope.size() ? -1 : indx;
    };

    int in_hs_idx = _indx_in(ti->body.inputs, lstm->insData[1].lock());
    int in_cs_idx = _indx_in(ti->body.inputs, lstm->insData[2].lock());
    int in_dt_idx = _indx_in(ti->body.inputs, rsp1->insData[0].lock());

    int out_hs_idx = _indx_in(ti->body.outputs, lstm->outData[0]);
    int out_cs_idx = _indx_in(ti->body.outputs, lstm->outData[1]);
    int out_dt_idx = _indx_in(ti->body.outputs, rsp2->outData[0]);

    // indexes should be [0,1,2] : sum == 3
    if (in_hs_idx + in_cs_idx + in_dt_idx != 3 || out_hs_idx + out_cs_idx + out_dt_idx != 3)
        return false;

    std::map<int, TensorIterator::PortMap> i2map, o2map, be2map;
    for (auto &m : ti->input_port_map) i2map[m.to] = m;
    for (auto &m : ti->output_port_map) o2map[m.to] = m;
    for (auto &m : ti->back_edges) be2map[m.to] = m;

    if (!one_of(i2map.size(), 3, 1) ||
        !one_of(o2map.size(), 3, 1) ||
        !one_of(be2map.size(), 2))
        return false;


    auto in_iter_rule = i2map[in_dt_idx];
    auto in_iter_data = ti->insData[in_iter_rule.from].lock();

    auto out_iter_rule = o2map[out_dt_idx];
    auto out_iter_data = ti->outData[out_iter_rule.from];

    // TI iterates only for full range of tensor
    if (!is_full_ranged(in_iter_rule, in_iter_data) ||
        !is_full_ranged(out_iter_rule, out_iter_data))
        return false;

    // supported only same axis and strides for in/out data tensors
    if (in_iter_rule.axis != out_iter_rule.axis ||
        in_iter_rule.stride != out_iter_rule.stride)
        return false;

    // supported only firs and second dim for LSTM-Sequence
    if (!one_of(in_iter_rule.axis, 0, 1))
        return false;

    bool no_init_state = i2map.size() == 1;
    bool no_last_state = o2map.size() == 1;

    if (!no_init_state && ( i2map[in_hs_idx].axis != -1 || i2map[in_cs_idx].axis != -1 ))
        return false;
    if (!no_last_state && ( o2map[out_hs_idx].axis != -1 || o2map[out_cs_idx].axis != -1 ))
        return false;

    auto i_order = no_init_state
            ? std::vector<int>{i2map[in_dt_idx].from}
            : std::vector<int>{i2map[in_dt_idx].from,
                               i2map[in_hs_idx].from,
                               i2map[in_cs_idx].from};
    auto o_order = no_last_state
            ? std::vector<int>{o2map[out_dt_idx].from}
            : std::vector<int>{o2map[out_dt_idx].from,
                               o2map[out_hs_idx].from,
                               o2map[out_cs_idx].from};

    // need swap an i/o ports if it is not in natural order
    std::string name = lstm->name + "_sequence";
    auto rnn  = std::make_shared<RNNLayer>(LayerParams{ name, "RNN",  Precision::FP32 });
    rnn->cellType = "LSTM";
    rnn->axis = in_iter_rule.axis;
    rnn->direction = in_iter_rule.stride == 1
            ? RNNLayer::RNN_FWD
            : RNNLayer::RNN_BWD;

    rnn->_weights = dynamic_cast<WeightableLayer*>(lstm.get())->_weights;
    rnn->blobs["weights"] = lstm->blobs["weights"];
    rnn->_biases = dynamic_cast<WeightableLayer*>(lstm.get())->_biases;
    rnn->blobs["biases"] = lstm->blobs["biases"];

    for (int i : i_order) {
        rnn->insData.push_back(ti->insData[i]);
        rnn->insData.back().lock()->inputTo[ti->name] = rnn;
    }
    for (int i : o_order) {
        rnn->outData.push_back(ti->outData[i]);
        rnn->outData.back()->creatorLayer = rnn;
    }

    return true;
}

bool CombineLSTMSeq(const ICNNNetwork &net) {
    // Apply action for all nodes
    CNNNetForestDFS(CNNNetGetAllInputLayers(net), &convertToLSTMSequence, true);
    return true;
}

bool UnrollTI(const ICNNNetwork &net) {
    return false;
}

}  // namespace NetPass
}  // namespace InferenceEngine

