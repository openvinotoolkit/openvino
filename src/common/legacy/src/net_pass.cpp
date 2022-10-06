// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/net_pass.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <map>
#include <vector>

#include "blob_factory.hpp"
#include "legacy/details/ie_cnn_network_tools.h"
#include <legacy/cnn_network_impl.hpp>
#include "cnn_network_ngraph_impl.hpp"
#include "legacy/graph_tools.hpp"
#include "legacy/ie_layers_internal.hpp"
#include "ie_memcpy.h"
#include "precision_utils.h"

#include "ie_legacy_itt.hpp"

namespace InferenceEngine {
namespace NetPass {

template <typename T, typename P>
inline bool one_of(T val, P item) {
    return val == item;
}

template <typename T, typename P, typename... Args>
inline bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

/************************************************************/
/****  TI Utils  ********************************************/
/************************************************************/

static std::vector<DataPtr> getAllInputs(const std::vector<DataPtr>& heads) {
    CNNLayerSet inputLayers;
    std::unordered_set<CNNLayer*> allLayers;

    // define any layer connected to provided Data object (consumer or creator)
    auto findConnectedLayer = [] (const DataPtr& data) -> CNNLayerPtr {
        auto consumerLayers = getInputTo(data);
        if (!consumerLayers.empty())
            return consumerLayers.begin()->second;

        auto creator = getCreatorLayer(data).lock();
        if (creator != nullptr)
            return creator;

        return nullptr;
    };

    // Define all start layers
    for (const auto& data : heads) {
        auto entryLayer = findConnectedLayer(data);

        if (entryLayer == nullptr) continue;

        details::UnorderedDFS(
            allLayers, entryLayer,
            [&inputLayers](const CNNLayerPtr &layer) {
                if (layer->insData.empty()) {
                    inputLayers.insert(layer);
                }
            },
            false);
    }

    std::vector<DataPtr> res = heads;
    // Add fake input data to point on not achievable
    // layers from head (like const placeholders)
    for (auto& starter : inputLayers) {
        DataPtr holder(new Data(starter->name + ":input_holder", starter->precision));
        getInputTo(holder)[starter->name] = starter;
        res.push_back(holder);
    }

    return res;
}

std::vector<CNNLayerPtr> TIBodySortTopologically(const TensorIterator::Body& body) {
    std::vector<CNNLayerPtr> all_layers;

    // In case of graph with several connected component
    // total entry point is a union of [inputs]U[outputs]
    // All internal nodes are achievable starting from this.
    auto total_entry_point = body.inputs;
    total_entry_point.insert(total_entry_point.end(),
                             body.outputs.begin(), body.outputs.end());

    auto all_input_layers = getAllInputs(total_entry_point);
    CNNNetForestDFS(
        all_input_layers,
        [&all_layers](const CNNLayerPtr &current) {
            all_layers.push_back(current);
        },
        false);
    std::reverse(all_layers.begin(), all_layers.end());
    return all_layers;
}

TensorIterator::Body CopyTIBody(const TensorIterator::Body& body, std::string suffix) {
    struct NoneStruct {};
    auto cp = [&](CNNLayerPtr lp) {
        return injectData<NoneStruct>(lp);
    };

    const auto all_orig = TIBodySortTopologically(body);

    std::unordered_map<CNNLayer*, CNNLayerPtr> old2new_l;
    for (const auto& orig : all_orig) {
        old2new_l[orig.get()] = cp(orig);
    }

    std::unordered_map<Data*, DataPtr> old2new_d;
    for (auto& in : body.inputs) {
        auto new_data = std::make_shared<Data>(*in.get());
        for (auto& to : getInputTo(new_data)) to.second = old2new_l[to.second.get()];

        old2new_d[in.get()] = new_data;
    }

    for (const auto& old : all_orig) {
        auto& new_one = old2new_l[old.get()];
        // remap output data
        for (size_t i = 0; i < old->outData.size(); i++) {
            auto old_data = old->outData[i];
            auto new_data = new_one->outData[i];
            getCreatorLayer(new_data) = CNNLayerWeakPtr(new_one);
            old2new_d[old_data.get()] = new_data;

            for (auto& to : getInputTo(new_data)) to.second = old2new_l[to.second.get()];
        }
        // remap input data
        for (size_t i = 0; i < old->insData.size(); i++) {
            auto old_data = old->insData[i].lock();
            auto new_data = old2new_d.at(old_data.get());
            new_one->insData[i] = new_data;
        }
    }

    // Add suffix
    if (!suffix.empty()) {
        for (auto& kvp : old2new_l) {
            auto layer = kvp.second;
            auto old_name = layer->name;
            layer->name += suffix;
            for (auto& ins : layer->insData) {
                getInputTo(ins.lock()).erase(old_name);
                getInputTo(ins.lock())[layer->name] = layer;
            }
        }
        for (auto& kvp : old2new_d) kvp.second->setName(kvp.second->getName() + suffix);
    }

    TensorIterator::Body res;
    for (auto& in : body.inputs) {
        auto found = old2new_d.find(in.get());
        IE_ASSERT(found != old2new_d.end());
        res.inputs.emplace_back(found->second);
    }

    for (auto& out : body.outputs) {
        auto found = old2new_d.find(out.get());
        IE_ASSERT(found != old2new_d.end());
        res.outputs.emplace_back(found->second);
    }

    // Fake holder.
    // The graph itself is a shared_ptr set where parent holds child.
    // Res.inputs vector hold head of graph and all nodes should be
    // achievable for oriented search started from that. But place
    // const holder has no input and cannot be achieved. So we need
    // to hold then in other way.
    //
    // Let's add one more Data object which has no representation in
    // original network. It will hold all unreachable const placeholder
    // nodes.
    //
    std::unordered_set<CNNLayerPtr> already_on_hold;
    for (auto &in : res.inputs) {
        // fake holder Data should have UNSPECIFIED precision
        if (in->getPrecision() == Precision::UNSPECIFIED) {
            for (const auto &kvp : getInputTo(in)) {
                already_on_hold.emplace(kvp.second);
            }
        }
    }
    std::vector<CNNLayerPtr> to_hold;
    for (auto& kvp : old2new_l) {
        auto layer = kvp.second;
        // layer has no parent Data object and is not on hold
        if (layer->insData.empty() && !already_on_hold.count(layer))
            to_hold.emplace_back(layer);
    }
    if (!to_hold.empty()) {
        // detect existing holder or create new one
        if (res.inputs.back()->getPrecision() != Precision::UNSPECIFIED ||
            res.inputs.back()->getDims().size() != 0) {
            res.inputs.emplace_back(new Data("const_holder", Precision::UNSPECIFIED));
        }

        auto holder = res.inputs.back();
        for (auto layer : to_hold) {
            getInputTo(holder)[layer->name] = layer;
        }
    }
    return res;
}

/************************************************************/
/****  TI rule helpers  *************************************/
/************************************************************/

inline bool is_full_ranged(const TensorIterator::PortMap& rule, const DataPtr& data) {
    if (!data) IE_THROW() << "Internal error. data == nullptr";

    if (rule.axis == -1 || !one_of(rule.stride, 1, -1)) return false;

    auto& shape = data->getDims();
    int size = static_cast<int>(shape[rule.axis]);

    int begin = rule.start >= 0 ? rule.start : size + rule.start + 1;
    int end = rule.end >= 0 ? rule.end : size + rule.end + 1;

    return (rule.stride == 1) ? begin == 0 && end == size : begin == size && end == 0;
}

using RuleSet = std::vector<TensorIterator::PortMap>;
using RuleClassSet = std::tuple<RuleSet, RuleSet, RuleSet>;

/**
 * @brief Helper to split port mapping rules to three group
 *
 *   first_class  - which has iteration component
 *   second_class - which has no iteration and there are no backedge connection to the same port
 *   third_class  - which has no iteration and has corresponding backedge
 *
 * @param ti TensorIterator layer to analyze
 * @return tuple with three classes of port map rule
 */
static RuleClassSet classifyInputRules(const TensorIterator& ti) {
    RuleSet first_class_rules, second_class_rules, third_class_rules;

    std::set<int> ports_with_backedge;
    for (const auto& back_edge : ti.back_edges) ports_with_backedge.insert(back_edge.to);

    for (const auto& rule : ti.input_port_map) {
        if (rule.axis != -1)
            first_class_rules.push_back(rule);

        else if (!ports_with_backedge.count(rule.to))
            second_class_rules.push_back(rule);

        else
            third_class_rules.push_back(rule);
    }
    return RuleClassSet {first_class_rules, second_class_rules, third_class_rules};
}

static RuleClassSet classifyOutputRules(const TensorIterator& ti) {
    RuleSet first_class_rules, second_class_rules, third_class_rules;

    std::set<int> ports_with_backedge;
    for (const auto& back_edge : ti.back_edges) ports_with_backedge.insert(back_edge.from);

    for (const auto& rule : ti.output_port_map) {
        if (rule.axis != -1)
            first_class_rules.push_back(rule);

        else if (!ports_with_backedge.count(rule.to))
            second_class_rules.push_back(rule);

        else
            third_class_rules.push_back(rule);
    }
    return RuleClassSet {first_class_rules, second_class_rules, third_class_rules};
}

/**
 * Merge slave connections into master data
 *
 * @param master
 * @param slave
 */
static void CombineData(DataPtr& master, DataPtr& slave) {
    for (auto& kvp : getInputTo(slave)) {
        auto& slave_layer = kvp.second;
        for (auto& slv_ins_wptr : slave_layer->insData) {
            auto slv_ins = slv_ins_wptr.lock();
            // Replace slave ptr with master
            if (slv_ins == slave) slv_ins_wptr = master;
        }
        getInputTo(master)[slave_layer->name] = slave_layer;
    }
}

/**
 * Preserve output data name and update output data map of the network
 *
 * @param in_data name to update
 * @param out_data name to preserve
 * @param net output data map to update with in_data
 */
template <typename NET>
void SaveOutputDataName(InferenceEngine::DataPtr in_data, InferenceEngine::DataPtr out_data, NET &net) {
    // TODO: update outputs of the network if out_data was output
    if (getInputTo(out_data).empty()) {
        auto data_name = out_data->getName();
        in_data->setName(data_name);
    }
}

/**
 * void SaveOutputDataName(InferenceEngine::DataPtr in_data, InferenceEngine::DataPtr out_data, NET &net), where
 * NET = CNNNetwork
 */
static void SaveOutputDataName(InferenceEngine::DataPtr in_data, InferenceEngine::DataPtr out_data, CNNNetwork& net) {
    if (getInputTo(out_data).empty()) {
        InferenceEngine::OutputsDataMap outputs_data_map = net.getOutputsInfo();
        auto out_data_name = out_data->getName();
        in_data->setName(out_data_name);
        if (outputs_data_map.count(out_data_name)) {
            auto parent_layer_ptr = getCreatorLayer(in_data).lock();
            IE_ASSERT(parent_layer_ptr != nullptr);
            auto parent_layer_name = parent_layer_ptr->name;
            size_t in_data_out_index = 0;
            for (size_t ind = 0; ind < parent_layer_ptr->outData.size(); ++ind) {
                if (parent_layer_ptr->outData[ind] == in_data) {
                    in_data_out_index = ind;
                }
            }
            net.addOutput(parent_layer_name, in_data_out_index);
        }
    }
}


/**
 * Remove layer form graph
 * May be applied only for inplace layer. One input, one output,
 * with same tensor descriptors.
 *
 * @param layer to remove from graph
 */
template <typename NET>
void RemoveLayer(CNNLayerPtr& layer, NET &net) {
    IE_ASSERT(layer->insData.size() == 1);
    IE_ASSERT(layer->outData.size() == 1);

    auto in_data = layer->input();
    auto out_data = layer->outData[0];

    IE_ASSERT(in_data->getTensorDesc() == out_data->getTensorDesc());
    auto &input_to_map = getInputTo(in_data);
    auto self_found = std::find_if(input_to_map.begin(), input_to_map.end(),
            [&layer] (const std::pair<std::string, CNNLayerPtr> &kvp) {
        return kvp.second == layer;
    });
    IE_ASSERT(self_found != input_to_map.end());
    // detach layer from input data
    input_to_map.erase(self_found);

    // transfer output connections into parent data
    CombineData(in_data, out_data);

    // save name for output data and update network output
    SaveOutputDataName(in_data, out_data, net);
}

/************************************************************/
/****  Converter Passes  ************************************/
/************************************************************/

static std::string cell_name(RNNSequenceLayer::CellType type) {
    std::string res;
    switch (type) {
    case RNNSequenceLayer::LSTM:
        res = "LSTM";
        break;
    case RNNSequenceLayer::GRU:
    case RNNSequenceLayer::GRU_LBR:
        res = "GRU";
        break;
    case RNNSequenceLayer::RNN:
        res = "RNN";
        break;
    }
    return res;
}

template <typename N>
bool convertToRNNSeq(CNNLayerPtr cur, const N& net) {
    if (cur->type != "TensorIterator") return true;

    auto ti = std::dynamic_pointer_cast<TensorIterator>(cur);
    IE_ASSERT(ti) << "Cannot cast object with type TensorIterator to TensorIterator object";

    auto all_body_layers = TIBodySortTopologically(ti->body);

    // Check if body is:  squeeze -> lstm_cell -> unsqueeze
    if (all_body_layers.size() != 3 || all_body_layers[0]->type != "Reshape" ||
        !one_of(all_body_layers[1]->type, "GRUCell", "RNNCell", "LSTMCell") || all_body_layers[2]->type != "Reshape")
        return false;

    auto rsp1 = std::dynamic_pointer_cast<ReshapeLayer>(all_body_layers[0]);
    auto cell = std::dynamic_pointer_cast<RNNCellBase>(all_body_layers[1]);
    auto rsp2 = std::dynamic_pointer_cast<ReshapeLayer>(all_body_layers[2]);

    IE_ASSERT(rsp1);
    IE_ASSERT(cell);
    IE_ASSERT(rsp2);

    size_t NS = (cell->cellType == RNNSequenceLayer::LSTM) ? 2 : 1;  // number of states

    IE_ASSERT(cell->insData.size() == NS + 1);  // {data, state1, [state2]}
    IE_ASSERT(cell->outData.size() == NS);      // {state1, [state2]}

    auto outData0InputsTo = getInputTo(cell->outData[0]);
    if (getCreatorLayer(cell->insData[0].lock()).lock() != rsp1 ||
            outData0InputsTo.empty() ||
            outData0InputsTo.begin()->second != rsp2)
        return false;

    // Check port mapping
    auto _indx_in = [&](const std::vector<DataPtr>& scope, const DataPtr& data) {
        size_t indx = static_cast<size_t>(std::find(scope.begin(), scope.end(), data) - scope.begin());
        return indx == scope.size() ? -1 : indx;
    };

    int in_dt_idx = static_cast<int>(_indx_in(ti->body.inputs, rsp1->insData[0].lock()));
    int in_hs_idx = static_cast<int>(_indx_in(ti->body.inputs, cell->insData[1].lock()));
    int in_cs_idx = NS == 2 ? static_cast<int>(_indx_in(ti->body.inputs, cell->insData[2].lock())) : -1;

    int out_dt_idx = static_cast<int>(_indx_in(ti->body.outputs, rsp2->outData[0]));
    int out_hs_idx = static_cast<int>(_indx_in(ti->body.outputs, cell->outData[0]));
    int out_cs_idx = NS == 2 ? static_cast<int>(_indx_in(ti->body.outputs, cell->outData[1])) : -1;

    // indexes should be [0,1,2] : sum == 3 or [0,1,-1] : sum == 0
    int sum = (static_cast<int>(NS) - 1) * 3;
    if (in_hs_idx + in_cs_idx + in_dt_idx != sum || out_hs_idx + out_cs_idx + out_dt_idx != sum) return false;

    std::map<int, TensorIterator::PortMap> i2map, o2map, be2map;
    for (auto& m : ti->input_port_map) i2map[m.to] = m;
    for (auto& m : ti->output_port_map) o2map[m.to] = m;
    for (auto& m : ti->back_edges) be2map[m.to] = m;

    if (!one_of(i2map.size(), NS + 1, 1u) || !one_of(o2map.size(), NS + 1, 1u) || !one_of(be2map.size(), NS))
        return false;

    auto in_iter_rule = i2map[in_dt_idx];
    auto in_iter_data = ti->insData[in_iter_rule.from].lock();

    auto out_iter_rule = o2map[out_dt_idx];
    auto out_iter_data = ti->outData[out_iter_rule.from];

    // TI iterates only for full range of tensor
    if (!is_full_ranged(in_iter_rule, in_iter_data) || !is_full_ranged(out_iter_rule, out_iter_data)) return false;

    // supported only same axis and strides for in/out data tensors
    if (in_iter_rule.axis != out_iter_rule.axis || in_iter_rule.stride != out_iter_rule.stride) return false;

    // supported only firs and second dim for LSTM-Sequence
    if (!one_of(in_iter_rule.axis, 0, 1)) return false;

    bool no_init_state = i2map.size() == 1;
    bool no_last_state = o2map.size() == 1;

    if (!no_init_state && (i2map[in_hs_idx].axis != -1 || (NS == 2 && i2map[in_cs_idx].axis != -1))) return false;
    if (!no_last_state && (o2map[out_hs_idx].axis != -1 || (NS == 2 && o2map[out_cs_idx].axis != -1))) return false;

    std::vector<int> i_order {i2map[in_dt_idx].from};
    if (!no_init_state) i_order.push_back(i2map[in_hs_idx].from);
    if (!no_init_state && NS == 2) i_order.push_back(i2map[in_cs_idx].from);

    std::vector<int> o_order {o2map[out_dt_idx].from};
    if (!no_last_state) o_order.push_back(o2map[out_hs_idx].from);
    if (!no_last_state && NS == 2) o_order.push_back(o2map[out_cs_idx].from);

    // need swap an i/o ports if it is not in natural order
    std::string name = cell->name + "_sequence";
    std::string type = cell_name(cell->cellType) + "Sequence";

    auto rnn = std::make_shared<RNNSequenceLayer>(LayerParams {name, type, cell->precision});
    rnn->axis = in_iter_rule.axis;
    rnn->direction = in_iter_rule.stride == 1 ? RNNSequenceLayer::FWD : RNNSequenceLayer::BWD;

    // copy base RNN cell fields
    rnn->cellType = cell->cellType;
    rnn->_weights = cell->_weights;
    rnn->_biases = cell->_biases;
    rnn->blobs["weights"] = rnn->_weights;
    rnn->blobs["biases"] = rnn->_biases;
    rnn->blobs = cell->blobs;
    rnn->activations = cell->activations;
    rnn->activation_alpha = cell->activation_alpha;
    rnn->activation_beta = cell->activation_beta;
    rnn->hidden_size = cell->hidden_size;
    rnn->clip = cell->clip;

    for (int i : i_order) {
        auto in_data = ti->insData[i].lock();
        getInputTo(in_data).erase(ti->name);
        getInputTo(in_data)[rnn->name] = rnn;
        rnn->insData.push_back(in_data);
    }
    for (int i : o_order) {
        rnn->outData.push_back(ti->outData[i]);
        getCreatorLayer(rnn->outData.back()) = rnn;
    }

    return true;
}

static bool unrollTI(CNNLayerPtr cur, CNNNetwork& net) {
    IE_SUPPRESS_DEPRECATED_START
    auto & icnnnet = static_cast<ICNNNetwork&>(net);
    IE_SUPPRESS_DEPRECATED_END
    auto inet = dynamic_cast<details::CNNNetworkImpl*>(&icnnnet);
    IE_ASSERT(inet != nullptr);

    if (cur->type != "TensorIterator") return true;

    auto ti = std::dynamic_pointer_cast<TensorIterator>(cur);
    IE_ASSERT(ti) << "Cannot cast object with type TensorIterator to TensorIterator object";

    int num = getNumIteration(*ti);  // -1 means inconsistent TI
    if (num == -1) return false;     // TODO: better to throw exception

    const auto& body = ti->body;

    std::vector<TensorIterator::Body> body_list(num);
    for (int i = 0; i < num; i++) {
        // copy with additional suffix to each object name
        body_list[i] = CopyTIBody(body, ":" + std::to_string(i));

        auto holder = body_list[i].inputs.back();
        if (holder->getPrecision() == Precision::UNSPECIFIED) {
            for (auto kvp : getInputTo(holder)) {
                inet->addLayer(kvp.second);
            }
        }
    }

    RuleSet first_class, second_class, third_class;
    std::tie(first_class, second_class, third_class) = classifyInputRules(*ti);

    /** Clean links on TI */
    for (auto& ins : ti->insData) getInputTo(ins.lock()).erase(ti->name);
    for (auto& outs : ti->outData) getCreatorLayer(outs).reset();

    /** FIRST class comes */
    for (size_t i = 0; i < first_class.size(); i++) {
        auto& rule = first_class[i];
        auto in_data = ti->insData[rule.from].lock();

        std::string name = ti->name + ":in_split_" + std::to_string(i);
        auto split = std::make_shared<SplitLayer>(LayerParams {name, "Split", cur->precision});
        split->_axis = rule.axis;
        split->outData.resize(num);
        split->insData.emplace_back(in_data);
        getInputTo(in_data)[split->name] = split;

        for (int j = 0; j < num; j++) {
            auto body_idx = rule.stride == 1 ? j : num - 1 - j;
            auto& chunk = body_list[body_idx].inputs[rule.to];
            getCreatorLayer(chunk) = split;
            split->outData[j] = chunk;
        }
    }

    /** SECOND class come on */
    for (const auto& rule : second_class) {
        auto in_data = ti->insData[rule.from].lock();

        for (int j = 0; j < num; j++) {
            auto& chunk = body_list[j].inputs[rule.to];
            CombineData(in_data, chunk);
        }
    }

    /** BACK EDGES that's your time */
    for (const auto& rule : ti->back_edges) {
        for (int i = 1; i < num; i++) {
            auto& from_data = body_list[i - 1].outputs[rule.from];
            auto& to_data = body_list[i].inputs[rule.to];
            CombineData(from_data, to_data);
        }
    }

    /** THIRD class end up */
    for (const auto& rule : third_class) {
        // first iteration
        auto from_data = ti->insData[rule.from].lock();
        auto& to_data = body_list[0].inputs[rule.to];
        CombineData(from_data, to_data);
    }

    /** And the same actions for outputs connections */
    std::tie(first_class, second_class, third_class) = classifyOutputRules(*ti);

    /** FIRST class comes */
    for (size_t i = 0; i < first_class.size(); i++) {
        auto& rule = first_class[i];
        auto out_data = ti->outData[rule.from];

        if (num == 1) {
            auto to_data = body_list[0].outputs[rule.to];
            auto parent = getCreatorLayer(to_data).lock();
            std::replace(parent->outData.begin(), parent->outData.end(), to_data, out_data);
            getCreatorLayer(out_data) = parent;
            CombineData(out_data, to_data);
            continue;
        }

        std::string name = ti->name + ":out_concat_" + std::to_string(i);
        auto concat = std::make_shared<ConcatLayer>(LayerParams {name, "Concat", cur->precision});
        concat->_axis = rule.axis;
        concat->insData.resize(num);
        concat->outData.emplace_back(out_data);
        getCreatorLayer(out_data) = concat;

        for (int j = 0; j < num; j++) {
            auto body_idx = rule.stride == 1 ? j : num - 1 - j;
            auto& chunk = body_list[body_idx].outputs[rule.to];
            getInputTo(chunk)[concat->name] = concat;
            concat->insData[j] = chunk;
        }
    }

    /** SECOND class come on */
    for (const auto& rule : second_class) {
        auto out_data = ti->outData[rule.from];

        for (int j = 0; j < num; j++) {
            auto& chunk = body_list[j].outputs[rule.to];
            CombineData(chunk, out_data);
        }
    }

    /** THIRD class end up */
    for (const auto& rule : third_class) {
        // first iteration
        auto& from_data = ti->outData[rule.from];
        auto& to_data = body_list[num - 1].outputs[rule.to];

        auto parent = getCreatorLayer(to_data).lock();
        std::replace(parent->outData.begin(), parent->outData.end(), to_data, from_data);
        getCreatorLayer(from_data) = parent;

        CombineData(from_data, to_data);
    }
    return true;
}

/************************************************************/
/****  Builder helpers   ************************************/
/************************************************************/

static CNNLayerPtr _concat(std::string name, Precision prc, SizeVector dims, int num) {
    auto res = std::make_shared<ConcatLayer>(LayerParams {name, "Concat", prc});
    res->_axis = 1;

    res->insData.resize(num);
    res->outData.resize(1);

    auto out_data = DataPtr(new Data(name, TensorDesc {prc, dims, TensorDesc::getLayoutByDims(dims)}));
    getCreatorLayer(out_data) = res;

    res->outData[0] = out_data;
    return res;
}

static CNNLayerPtr _split(std::string name, Precision prc, SizeVector dims, int num) {
    auto res = std::make_shared<SplitLayer>(LayerParams {name, "Split", prc});
    res->_axis = 1;
    res->params["axis"] = std::to_string(res->_axis);

    res->insData.resize(1);
    res->outData.resize(num);

    for (int i = 0; i < num; i++) {
        auto out_data = DataPtr(
            new Data(name + "_part_" + std::to_string(i), TensorDesc {prc, dims, TensorDesc::getLayoutByDims(dims)}));
        getCreatorLayer(out_data) = res;

        res->outData[i] = out_data;
    }
    return res;
}

static CNNLayerPtr _fc(std::string name, Precision prc, SizeVector dims, Blob::Ptr& W, Blob::Ptr& B) {
    auto res = std::make_shared<FullyConnectedLayer>(LayerParams {name, "FullyConnected", prc});

    res->_weights = W;
    res->_biases = B;
    res->_out_num = static_cast<unsigned>(dims[1]);
    res->blobs["weights"] = W;
    res->blobs["biases"] = B;
    res->params["out-size"] = std::to_string(dims[1]);

    res->insData.resize(1);
    res->outData.resize(1);

    auto out_data = DataPtr(new Data(name, TensorDesc {prc, dims, TensorDesc::getLayoutByDims(dims)}));
    getCreatorLayer(out_data) = res;

    res->outData[0] = out_data;
    return res;
}

static std::shared_ptr<ClampLayer> _act(std::string name, Precision prc, SizeVector dims, std::string type) {
    auto res = std::make_shared<ClampLayer>(LayerParams {name, type, prc});

    res->params["type"] = type;

    res->insData.resize(1);
    res->outData.resize(1);

    auto out_data = DataPtr(new Data(name, TensorDesc {prc, dims, TensorDesc::getLayoutByDims(dims)}));
    getCreatorLayer(out_data) = res;

    res->outData[0] = out_data;
    return res;
}

static CNNLayerPtr _pwr(std::string name, Precision prc, SizeVector dims, float scale, float shift) {
    auto res = std::make_shared<PowerLayer>(LayerParams {name, "Power", prc});

    res->power = 1.0;
    res->scale = scale;
    res->offset = shift;
    res->params["power"] = CNNLayer::ie_serialize_float(res->power);
    res->params["scale"] = CNNLayer::ie_serialize_float(res->scale);
    res->params["shift"] = CNNLayer::ie_serialize_float(res->offset);

    res->insData.resize(1);
    res->outData.resize(1);

    auto out_data = DataPtr(new Data(name, TensorDesc {prc, dims, TensorDesc::getLayoutByDims(dims)}));
    getCreatorLayer(out_data) = res;

    res->outData[0] = out_data;
    return res;
}

static CNNLayerPtr _eltw(std::string name, Precision prc, SizeVector dims, std::string type) {
    auto res = std::make_shared<EltwiseLayer>(LayerParams {name, "Eltwise", prc});

    res->params["operation"] = type;
    res->_operation = type == "sum" ? EltwiseLayer::Sum : EltwiseLayer::Prod;

    res->insData.resize(2);
    res->outData.resize(1);

    auto out_data = DataPtr(new Data(name, TensorDesc {prc, dims, TensorDesc::getLayoutByDims(dims)}));
    getCreatorLayer(out_data) = res;

    res->outData[0] = out_data;
    return res;
}

static std::shared_ptr<ReshapeLayer> _resh(std::string name, Precision prc, SizeVector dims) {
    auto res = std::make_shared<ReshapeLayer>(LayerParams {name, "Reshape", prc});

    res->insData.resize(1);
    res->outData.resize(1);

    auto out_data = DataPtr(new Data(name, TensorDesc {prc, dims, TensorDesc::getLayoutByDims(dims)}));
    getCreatorLayer(out_data) = res;

    res->outData[0] = out_data;
    return res;
}

static std::shared_ptr<RNNCellBase> _cell(std::string name, Precision prc, SizeVector data_dims, SizeVector state_dims,
                                          RNNSequenceLayer::CellType type) {
    std::shared_ptr<RNNCellBase> res;
    size_t NS = 1;
    switch (type) {
    case RNNSequenceLayer::LSTM:
        res = std::make_shared<LSTMCell>(LayerParams {name, "LSTMCell", prc});
        NS = 2;
        break;
    case RNNSequenceLayer::GRU:
    case RNNSequenceLayer::GRU_LBR:
        res = std::make_shared<GRUCell>(LayerParams {name, "GRUCell", prc});
        break;
    case RNNSequenceLayer::RNN:
        res = std::make_shared<RNNCell>(LayerParams {name, "RNNCell", prc});
        break;
    }

    res->cellType = type;
    res->insData.resize(1 + NS);
    res->outData.resize(NS);

    auto out_data =
        DataPtr(new Data(name + ":out_data", TensorDesc {prc, data_dims, TensorDesc::getLayoutByDims(data_dims)}));
    getCreatorLayer(out_data) = res;
    res->outData[0] = out_data;

    for (size_t i = 0; i < NS; i++) {
        auto out_state = DataPtr(new Data(name + ":out_state_" + std::to_string(i),
                                          TensorDesc {prc, state_dims, TensorDesc::getLayoutByDims(state_dims)}));
        getCreatorLayer(out_state) = res;
        res->outData[i] = out_state;
    }

    return res;
}

static std::shared_ptr<TensorIterator> _ti(std::string name, Precision prc, size_t NS) {
    auto res = std::make_shared<TensorIterator>(LayerParams {name, "TensorIterator", prc});

    res->insData.resize(1 + NS);
    res->outData.resize(1 + NS);

    return res;
}

static void _link(CNNLayerPtr src, CNNLayerPtr dst, size_t src_port = 0, size_t dst_port = 0) {
    auto data = src->outData[src_port];
    getInputTo(data)[dst->name] = dst;
    dst->insData[dst_port] = data;
}

static void _link(DataPtr& data, CNNLayerPtr dst, size_t dst_port = 0) {
    getInputTo(data)[dst->name] = dst;
    dst->insData[dst_port] = data;
}

/** Link nodes with clipping data if required (clip_val != 0.0) */
static void _link_with_clip(CNNLayerPtr src, CNNLayerPtr dst, const float clip_val, size_t src_port = 0,
                            size_t dst_port = 0) {
    if (clip_val == 0.0f) {
        _link(src, dst, src_port, dst_port);
    } else {
        auto clip_name = dst->name + "_clip";
        auto clip_prc = dst->precision;
        auto clip_shape = src->outData[src_port]->getTensorDesc().getDims();
        auto clip = _act(clip_name, clip_prc, clip_shape, "clamp");
        clip->params["min"] = CNNLayer::ie_serialize_float(-clip_val);
        clip->params["max"] = CNNLayer::ie_serialize_float(clip_val);
        clip->min_value = -clip_val;
        clip->max_value = clip_val;

        _link(src, clip, src_port, 0);
        _link(clip, dst, 0, dst_port);
    }
}
static Blob::Ptr wrap_as_tensor(Blob::Ptr src, SizeVector dims) {
    auto res = make_blob_with_precision(
        TensorDesc {src->getTensorDesc().getPrecision(), dims, TensorDesc::getLayoutByDims(dims)}, src->buffer());
    IE_ASSERT(src->size() == res->size());
    return res;
}

static Blob::Ptr make_region_copy(Blob::Ptr src, SizeVector region, SizeVector offset) {
    IE_ASSERT(region.size() == offset.size());
    IE_ASSERT(region.size() == src->getTensorDesc().getDims().size());

    auto res = make_plain_blob(src->getTensorDesc().getPrecision(), region);
    res->allocate();

    size_t elem_size = src->getTensorDesc().getPrecision().size();
    auto src_ptr = src->buffer().as<uint8_t*>();
    auto dst_ptr = res->buffer().as<uint8_t*>();

    auto& dd = src->getTensorDesc().getDims();
    SizeVector src_dims {1, 1, 1};
    std::copy(dd.begin(), dd.end(), src_dims.end() - dd.size());

    SizeVector dims {1, 1, 1};
    std::copy(region.begin(), region.end(), dims.end() - region.size());

    SizeVector off {0, 0, 0};
    std::copy(offset.begin(), offset.end(), off.end() - offset.size());

    const auto D1 = dims[0];
    const auto D2 = dims[1];
    const auto D3 = dims[2];
    const auto off1 = off[0];
    const auto off2 = off[1];
    const auto off3 = off[2];
    const auto str1 = src_dims[1] * src_dims[2];
    const auto str2 = src_dims[2];

    for (size_t d1 = 0; d1 < D1; d1++)
        for (size_t d2 = 0; d2 < D2; d2++) {
            auto off_src = (off1 + d1) * str1 + (off2 + d2) * str2 + off3;
            auto off_dst = d1 * D2 * D3 + d2 * D3;
            ie_memcpy(dst_ptr + off_dst * elem_size, res->byteSize(), src_ptr + off_src * elem_size, D3 * elem_size);
        }

    return res;
}

static bool unrollRNNCellBody(CNNLayerPtr cur) {
    if (cur->type != "RNNCell") return true;

    auto cell = std::dynamic_pointer_cast<RNNCellBase>(cur);
    IE_ASSERT(cell) << "Cannot cast object with type ***Cell to WeightableLayer object";

    auto name = cell->name;

    auto in_data = cell->insData[0].lock();
    auto in_h_state = cell->insData[1].lock();
    auto out_h_state = cell->outData[0];

    auto d_dims = in_data->getTensorDesc().getDims();
    auto s_dims = in_h_state->getTensorDesc().getDims();

    size_t N = d_dims[0];
    size_t D = d_dims[1];
    size_t S = s_dims[1];

    auto prc = cell->precision;

    /** Release links on TI */
    for (auto& ins : cell->insData) getInputTo(ins.lock()).erase(cell->name);
    for (auto& outs : cell->outData) getCreatorLayer(outs).reset();

    // operations
    auto concat = _concat(name + ":concat", prc, {N, D + S}, 2);
    auto fc = _fc(name + ":fc", prc, {N, S}, cell->_weights, cell->_biases);
    auto act = _act(name + ":act", prc, {N, S}, cell->activations[0]);

    // Connection
    _link(in_data, concat, 0);
    _link(in_h_state, concat, 1);
    _link(concat, fc);
    _link_with_clip(fc, act, cell->clip);

    // Output
    act->outData[0] = out_h_state;
    getCreatorLayer(out_h_state) = act;

    return true;
}

static bool unrollLSTMCellBody(CNNLayerPtr cur) {
    if (cur->type != "LSTMCell") return true;

    auto cell = std::dynamic_pointer_cast<RNNCellBase>(cur);
    IE_ASSERT(cell) << "Cannot cast object with type ***Cell to WeightableLayer object";

    auto name = cell->name;

    auto in_data = cell->insData[0].lock();
    auto in_h_state = cell->insData[1].lock();
    auto in_c_state = cell->insData[2].lock();
    auto out_h_state = cell->outData[0];
    auto out_c_state = cell->outData[1];

    auto d_dims = in_data->getTensorDesc().getDims();
    auto s_dims = in_h_state->getTensorDesc().getDims();

    size_t N = d_dims[0];
    size_t D = d_dims[1];
    size_t S = s_dims[1];
    size_t G = 4;

    auto prc = cell->precision;

    /** Release links on TI */
    for (auto& ins : cell->insData) getInputTo(ins.lock()).erase(cell->name);
    for (auto& outs : cell->outData) getCreatorLayer(outs).reset();

    // operations
    auto concat = _concat(name + ":concat", prc, {N, D + S}, 2);
    auto split = _split(name + ":split", prc, {N, S}, static_cast<int>(G));
    auto fc = _fc(name + ":fc", prc, {N, S * G}, cell->_weights, cell->_biases);

    const std::string _f = cell->activations[0], _g = cell->activations[1], _h = cell->activations[2];

    auto act_f = _act(name + ":act_f", prc, {N, S}, _f);
    auto act_i = _act(name + ":act_i", prc, {N, S}, _f);
    auto act_c = _act(name + ":act_c", prc, {N, S}, _g);
    auto act_o = _act(name + ":act_o", prc, {N, S}, _f);
    auto act_x = _act(name + ":act_x", prc, {N, S}, _h);

    auto mul_ic = _eltw(name + ":mul_ic", prc, {N, S}, "mul");
    auto mul_f = _eltw(name + ":mul_f", prc, {N, S}, "mul");
    auto sum = _eltw(name + ":sum", prc, {N, S}, "sum");
    auto mul = _eltw(name + ":mul", prc, {N, S}, "mul");

    // Connection
    _link(in_data, concat, 0);
    _link(in_h_state, concat, 1);
    _link(concat, fc);

    _link_with_clip(fc, split, cell->clip);

    _link(split, act_f, 0, 0);
    _link(split, act_i, 1, 0);
    _link(split, act_c, 2, 0);
    _link(split, act_o, 3, 0);

    _link(act_i, mul_ic, 0, 0);
    _link(act_c, mul_ic, 0, 1);

    _link(act_f, mul_f, 0, 0);
    _link(in_c_state, mul_f, 1);

    _link(mul_f, sum, 0, 0);
    _link(mul_ic, sum, 0, 1);

    _link(sum, act_x);

    _link(act_x, mul, 0, 0);
    _link(act_o, mul, 0, 1);

    // Output
    mul->outData[0] = out_h_state;
    getCreatorLayer(out_h_state) = mul;

    CombineData(out_c_state, sum->outData[0]);
    sum->outData[0] = out_c_state;
    getCreatorLayer(out_c_state) = sum;

    return true;
}

static bool unrollGRUCellBody(CNNLayerPtr cur, bool linear_before_reset = false) {
    if (cur->type != "GRUCell") return true;

    auto cell = std::dynamic_pointer_cast<GRUCell>(cur);
    IE_ASSERT(cell) << "Cannot cast object with type ***Cell to WeightableLayer object";

    auto name = cell->name;

    auto in_data = cell->insData[0].lock();
    auto in_h_state = cell->insData[1].lock();
    auto out_h_state = cell->outData[0];

    auto d_dims = in_data->getTensorDesc().getDims();
    auto s_dims = in_h_state->getTensorDesc().getDims();

    size_t N = d_dims[0];
    size_t D = d_dims[1];
    size_t S = s_dims[1];

    // Split weights UR and O gates. Original gates are URO
    size_t bG = linear_before_reset ? 4 : 3;
    auto orig_W = wrap_as_tensor(cell->_weights, {3, S, D + S});
    auto orig_B = wrap_as_tensor(cell->_biases, {bG, S});

    auto ur_W = make_region_copy(orig_W, {2, S, D + S}, {0, 0, 0});
    auto o_W = make_region_copy(orig_W, {1, S, D + S}, {2, 0, 0});
    auto ur_B = make_region_copy(orig_B, {2, S}, {0, 0});
    auto o_B = make_region_copy(orig_B, {1, S}, {2, 0});

    auto prc = cell->precision;

    /** Release links on TI */
    for (auto& ins : cell->insData) getInputTo(ins.lock()).erase(cell->name);
    for (auto& outs : cell->outData) getCreatorLayer(outs).reset();

    // operations
    auto concat = _concat(name + ":concat", prc, {N, D + S}, 2);
    auto split = _split(name + ":split", prc, {N, S}, 2);
    auto fc_ur = _fc(name + ":fc_ur", prc, {N, S * 2}, ur_W, ur_B);

    const std::string _f = cell->activations[0], _g = cell->activations[1];

    auto act_ur = _act(name + ":act_ur", prc, {N, 2 * S}, _f);
    auto act_o = _act(name + ":act_o", prc, {N, S}, _g);

    auto mul_u = _eltw(name + ":mul_u", prc, {N, S}, "mul");
    auto mul_r = _eltw(name + ":mul_r", prc, {N, S}, "mul");

    auto pwr_m1 = _pwr(name + ":pwr", prc, {N, S}, -1.0, 1.0);

    auto mul = _eltw(name + ":mul", prc, {N, S}, "mul");
    auto sum = _eltw(name + ":sum", prc, {N, S}, "sum");

    /**
     * - zt = _f(Wz*[Xt + Ht-1] + Bz)
     * - rt = _f(Wr*[Xt + Ht-1] + Br)
     * - ht = _g(Wh*[Xt + (rt (.) Ht-1)] + Bh)    # default, when linear_before_reset = 0
     * - ht = _g(Whw*Xt + Bhw + (rt (.) (Whr*Ht-1 + Bhr))) # when linear_before_reset != 0
     * - Ht = (1 - zt) (.) ht + zt (.) Ht-1
     */
    _link(in_data, concat, 0);
    _link(in_h_state, concat, 1);
    _link(concat, fc_ur);
    _link_with_clip(fc_ur, act_ur, cell->clip);
    _link(act_ur, split);  // split[0] - zt,  split[1] - rt

    if (linear_before_reset) {
        auto lbr_B = wrap_as_tensor(orig_B, {4, S});

        auto whw_W = make_region_copy(o_W, {1, S, D}, {0, 0, 0});
        auto whr_W = make_region_copy(o_W, {1, S, S}, {0, 0, D});
        auto whw_B = make_region_copy(lbr_B, {1, S}, {2, 0});
        auto whr_B = make_region_copy(lbr_B, {1, S}, {3, 0});

        auto fc_whr = _fc(name + ":fc_whr", prc, {N, S}, whr_W, whr_B);
        auto fc_whw = _fc(name + ":fc_whw", prc, {N, S}, whw_W, whw_B);
        auto sum_h = _eltw(name + ":sum_h", prc, {N, S}, "sum");

        _link(in_h_state, fc_whr);                  //                            Whr*Ht-1 + Bhr
        _link(fc_whr, mul_r, 0);                    //
        _link(split, mul_r, 1, 1);                  //                    rt (.) (Whr*Ht-1 + Bhr)
        _link(in_data, fc_whw);                     //    Whw*Xt + Bhw
        _link(fc_whw, sum_h, 0, 0);                 //
        _link(mul_r, sum_h, 0, 1);                  //    Whw*Xt + Bhw + (rt (.) (Whr*Ht-1 + Bhr))
        _link_with_clip(sum_h, act_o, cell->clip);  // _g(Whw*Xt + Bhw + (rt (.) (Whr*Ht-1 + Bhr)))
    } else {
        auto fc_wh = _fc(name + ":fc_o", prc, {N, S}, o_W, o_B);
        auto concat_h = _concat(name + ":concat_h", prc, {N, D + S}, 2);

        _link(split, mul_r, 1, 0);                  //
        _link(in_h_state, mul_r, 1);                //              rt (.) Ht-1
        _link(in_data, concat_h, 0);                //
        _link(mul_r, concat_h, 0, 1);               //       [Xt + (rt (.) Ht-1)]
        _link(concat_h, fc_wh);                     //    Wh*[Xt + (rt (.) Ht-1)] + Bh
        _link_with_clip(fc_wh, act_o, cell->clip);  // _g(Wh*[Xt + (rt (.) Ht-1)] + Bh)
    }

    _link(split, pwr_m1, 0, 0);   //  1 - zt
    _link(act_o, mul, 0, 0);      //
    _link(pwr_m1, mul, 0, 1);     // (1 - zt) (.) ht
    _link(split, mul_u, 0, 0);    //
    _link(in_h_state, mul_u, 1);  //                   zt (.) Ht-1
    _link(mul, sum, 0, 0);        //
    _link(mul_u, sum, 0, 1);      // (1 - zt) (.) ht + zt (.) Ht-1

    // Output
    sum->outData[0] = out_h_state;
    getCreatorLayer(out_h_state) = sum;

    return true;
}

static bool unrollCell(CNNLayerPtr cur) {
    auto cell = std::dynamic_pointer_cast<RNNCellBase>(cur);
    switch (cell->cellType) {
    case RNNCellBase::LSTM:
        return unrollLSTMCellBody(cur);
    case RNNCellBase::GRU:
        return unrollGRUCellBody(cur);
    case RNNCellBase::GRU_LBR:
        return unrollGRUCellBody(cur, true);
    case RNNCellBase::RNN:
        return unrollRNNCellBody(cur);
    }
    return false;
}

static bool unrollSeq(CNNLayerPtr cur) {
    if (!one_of(cur->type, "LSTMSequence", "GRUSequence", "RNNSequence")) return true;

    auto seq = std::dynamic_pointer_cast<RNNSequenceLayer>(cur);
    IE_ASSERT(seq) << "Cannot cast object with type ***Sequence to RNNSequenceLayer object";

    auto name = seq->name;

    auto in_data = seq->insData[0].lock();
    auto in_h_state = seq->insData[1].lock();
    auto out_data = seq->outData[0];

    auto in_d_dims = in_data->getTensorDesc().getDims();
    auto state_dims = in_h_state->getTensorDesc().getDims();
    auto out_d_dims = out_data->getTensorDesc().getDims();

    const int axis = seq->axis;
    const auto direct = seq->direction;
    const auto prc = seq->precision;

    /** Release links on Seq */
    for (auto& ins : seq->insData) getInputTo(ins.lock()).erase(seq->name);
    for (auto& outs : seq->outData) getCreatorLayer(outs).reset();

    /** Body subgraph*/
    auto in_d_body_dims = in_d_dims;
    in_d_body_dims[axis] = 1;

    auto in_d_body_squeeze_dims = in_d_dims;
    in_d_body_squeeze_dims.erase(in_d_body_squeeze_dims.begin() + axis);

    auto out_d_body_dims = out_d_dims;
    out_d_body_dims[axis] = 1;

    auto out_d_body_squeeze_dims = out_d_dims;
    out_d_body_squeeze_dims.erase(out_d_body_squeeze_dims.begin() + axis);

    auto body_in_data = DataPtr(
        new Data(name + ":data_in", TensorDesc {prc, in_d_body_dims, TensorDesc::getLayoutByDims(in_d_body_dims)}));

    auto resh1 = _resh(name + ":resh1", prc, in_d_body_squeeze_dims);
    auto cell = _cell(name + ":cell", prc, out_d_body_squeeze_dims, state_dims, seq->cellType);
    auto resh2 = _resh(name + ":resh2", prc, out_d_body_dims);

    _link(body_in_data, resh1);
    _link(resh1, cell);
    _link(cell, resh2);

    cell->_weights = seq->_weights;
    cell->_biases = seq->_biases;
    cell->blobs["weights"] = cell->_weights;
    cell->blobs["biases"] = cell->_biases;
    cell->hidden_size = seq->hidden_size;
    cell->clip = seq->clip;
    cell->activations = seq->activations;
    cell->activation_alpha = seq->activation_alpha;
    cell->activation_beta = seq->activation_beta;

    const size_t NS = cell->outData.size();  // num of state

    /** TI layer */
    auto ti = _ti(name + ":ti", prc, NS);
    _link(in_data, ti, 0);

    ti->outData[0] = out_data;
    getCreatorLayer(out_data) = ti;

    ti->body.inputs.push_back(body_in_data);
    ti->body.outputs.push_back(resh2->outData[0]);

    int start = direct == RNNSequenceLayer::FWD ? 0 : -1;
    int end = direct == RNNSequenceLayer::FWD ? -1 : 0;
    int step = direct == RNNSequenceLayer::FWD ? 1 : -1;
    ti->input_port_map.push_back({0, 0, axis, step, start, end, 1});
    ti->output_port_map.push_back({0, 0, axis, step, start, end, 1});

    for (size_t i = 0; i < NS; i++) {
        auto in_state = seq->insData[1 + i].lock();
        _link(in_state, ti, 1 + i);

        auto out_state = seq->outData[1 + i];
        ti->outData[1 + i] = out_state;
        getCreatorLayer(out_state) = ti;

        auto body_in_state = DataPtr(new Data(name + ":state_in_" + std::to_string(i),
                                              TensorDesc {prc, state_dims, TensorDesc::getLayoutByDims(state_dims)}));

        _link(body_in_state, cell, 1 + i);

        ti->body.inputs.push_back(body_in_state);
        ti->body.outputs.push_back(cell->outData[i]);

        const int ii = 1 + static_cast<int>(i);
        ti->input_port_map.push_back({ii, ii, -1, 0, 0, 0, 0});
        ti->output_port_map.push_back({ii, ii, -1, 0, 0, 0, 0});
        ti->back_edges.push_back({ii, ii, -1, 0, 0, 0, 0});
    }

    return true;
}

/************************************************************/
/****  Converter API  ***************************************/
/************************************************************/

template <typename N>
std::vector<CNNLayerPtr> TopolSort(const N& net);

template <>
std::vector<CNNLayerPtr> TopolSort(const CNNNetwork& net) {
    return details::CNNNetSortTopologically(net);
}

template <>
std::vector<CNNLayerPtr> TopolSort(const TensorIterator::Body& net) {
    return details::CNNSubnetSortTopologically({net.inputs, net.outputs});
}

template <>
std::vector<CNNLayerPtr> TopolSort(const details::CNNSubnet& net) {
    return details::CNNSubnetSortTopologically(net);
}

static void restore_net_consistency(CNNNetwork& net) {
    IE_SUPPRESS_DEPRECATED_START
    auto & icnnnet = static_cast<ICNNNetwork&>(net);
    // ilavreno:
    // issues with RTTI on OSX once we compiled inference_engine_legacy as STATIC library
    // So, we use static_cast instead of dynamic_cast since we are sure that
    // icnnnet is always a details::CNNNetworkImpl
    auto inet = static_cast<details::CNNNetworkImpl*>(&icnnnet);
    IE_ASSERT(inet != nullptr);
    // At first all layers should be available via findByName() api.
    // In other words all layers should be present in internal map<name, layer>
    for (auto& l : TopolSort(net)) {
        inet->addLayer(l);
    }
    IE_SUPPRESS_DEPRECATED_END
}

template <typename N, typename T>
bool ApplyForAll(N& net, T action) {
    auto all_layers = TopolSort(net);
    bool sts = true;

    for (auto& layer : all_layers) sts &= action(layer, net);

    return sts;
}

template <typename N, typename T, typename P>
bool ApplyForAll_if(N& net, T action, P pred) {
    auto all_layers = TopolSort(net);
    bool sts = true;

    for (auto& layer : all_layers)
        if (pred(layer)) sts &= action(layer);

    return sts;
}

bool CombineRNNSeq(CNNNetwork& net) {
    auto res = ApplyForAll(net, convertToRNNSeq<CNNNetwork>);
    restore_net_consistency(net);
    return res;
}

bool CombineRNNSeq(TensorIterator::Body& net) {
    return ApplyForAll(net, convertToRNNSeq<TensorIterator::Body>);
}

bool UnrollTI(CNNNetwork& net) {
    auto res = ApplyForAll(net, unrollTI);
    restore_net_consistency(net);
    return res;
}

template <typename NET>
bool UnrollRNN_if_impl(NET& net, const std::function<bool(const RNNCellBase&)> pred) {
    // Filter layers by RNN specific type
    auto _seq_pred = [&](CNNLayerPtr layer) {
        auto rnn = std::dynamic_pointer_cast<RNNSequenceLayer>(layer);
        if (!rnn) return false;
        return pred(*rnn.get());
    };
    auto _cell_pred = [&](CNNLayerPtr layer) {
        auto rnn = std::dynamic_pointer_cast<RNNCellBase>(layer);
        if (!rnn || !one_of(rnn->type, "LSTMCell", "GRUCell", "RNNCell")) return false;
        return pred(*rnn.get());
    };

    bool res = true;
    res &= ApplyForAll_if(net, unrollSeq, _seq_pred);
    res &= ApplyForAll_if(net, unrollCell, _cell_pred);
    return res;
}

bool UnrollRNN_if(CNNNetwork& net, const std::function<bool(const RNNCellBase&)> pred) {
    auto res = UnrollRNN_if_impl(net, pred);
    restore_net_consistency(net);
    return res;
}

bool UnrollRNN_if(TensorIterator::Body& net, const std::function<bool(const RNNCellBase&)> pred) {
    return UnrollRNN_if_impl(net, pred);
}


/**
 * ===========================
 * Precision conversion passes
 * ===========================
 */

namespace {

template <Precision::ePrecision PREC_FROM, Precision::ePrecision PREC_TO>
void convertArrayPrecision(typename PrecisionTrait<PREC_TO>::value_type* dst,
                           const typename PrecisionTrait<PREC_FROM>::value_type* src, size_t nelem) {
    using dst_type = typename PrecisionTrait<PREC_TO>::value_type;

    for (size_t i = 0; i < nelem; i++) {
        dst[i] = PrecisionUtils::saturate_cast<dst_type>(static_cast<dst_type>(src[i]));
    }
}

template <>
void convertArrayPrecision<Precision::FP16, Precision::FP32>(float* dst, const short* src, size_t nelem) {
    PrecisionUtils::f16tof32Arrays(dst, src, nelem, 1.0f, 0.0f);
}

template <Precision::ePrecision PREC_FROM, Precision::ePrecision PREC_TO>
Blob::Ptr convertBlobPrecision(const Blob::Ptr& blob) {
    using from_d_type = typename PrecisionTrait<PREC_FROM>::value_type;
    using to_d_type = typename PrecisionTrait<PREC_TO>::value_type;

    auto tensor_desc = blob->getTensorDesc();
    Blob::Ptr new_blob = make_shared_blob<to_d_type>(TensorDesc {PREC_TO, tensor_desc.getDims(), tensor_desc.getLayout()});
    new_blob->allocate();
    auto target = new_blob->buffer().as<to_d_type*>();
    auto source = blob->buffer().as<from_d_type*>();
    convertArrayPrecision<PREC_FROM, PREC_TO>(target, source, blob->size());
    return new_blob;
}

// forward declaration to use in convertLayerPrecision<>()
template <Precision::ePrecision PREC_FROM, Precision::ePrecision PREC_TO, typename NET>
void convertPrecisionForAll(NET &net);

template <Precision::ePrecision PREC_FROM, Precision::ePrecision PREC_TO>
void convertLayerPrecision(const CNNLayerPtr& layer) {
    for (auto &out_data : layer->outData) {
        if (PREC_FROM == out_data->getPrecision())
            out_data->setPrecision(PREC_TO);
    }
    for (auto &in_data : layer->insData) {
        if (PREC_FROM == in_data.lock()->getPrecision())
            in_data.lock()->setPrecision(PREC_TO);
    }

    if (layer->precision == PREC_FROM)
        layer->precision = PREC_TO;

    if (HasInternalSubnet(layer)) {
        // apply the same conversion pass for internal graph
        auto layer_subnet = GetInternalSubnet(layer);
        convertPrecisionForAll<PREC_FROM, PREC_TO>(layer_subnet);
    }

    auto wLayer = dynamic_cast<InferenceEngine::WeightableLayer *>(layer.get());
    if (wLayer) {
        if (wLayer->_weights && wLayer->_weights->getTensorDesc().getPrecision() == PREC_FROM) {
            wLayer->_weights = convertBlobPrecision<PREC_FROM, PREC_TO>(wLayer->_weights);
        }
        if (wLayer->_biases && wLayer->_biases->getTensorDesc().getPrecision() == PREC_FROM) {
            wLayer->_biases = convertBlobPrecision<PREC_FROM, PREC_TO>(wLayer->_biases);
        }
    }

    for (auto &blob : layer->blobs) {
        auto &data = blob.second;
        if (nullptr != data) {
            if (data->getTensorDesc().getPrecision() == PREC_FROM) {
                data = convertBlobPrecision<PREC_FROM, PREC_TO>(data);
            }
        }
    }
}

template <typename NET>
void RemoveConverts(NET& net, std::vector<CNNLayerPtr>& to_remove) {
    for (auto& layer : to_remove) {
        RemoveLayer(layer, net);
    }
}

template <>
void RemoveConverts(CNNNetwork& net, std::vector<CNNLayerPtr>& to_remove) {
    OutputsDataMap outputs = net.getOutputsInfo();
    for (auto& layer : to_remove) {
        if (!std::any_of(outputs.begin(), outputs.end(),
            [layer](std::pair<std::string, DataPtr> p) { return p.second->getName() == layer->name; })) {
            RemoveLayer(layer, net);
        }
    }
}

template <typename NET>
void fixConvertLayers(NET &net) {
    std::vector<CNNLayerPtr> to_remove;
    auto all_layers = TopolSort(net);
    for (auto &layer : all_layers) {
        if (layer->type == "Convert") {
            auto out_precision = layer->outData[0]->getPrecision();
            auto in_precision = layer->input()->getPrecision();

            // Restore destination_type attribute after conversion
            auto found = layer->params.find("precision");
            IE_ASSERT(found != layer->params.end());
            found->second = out_precision.name();

            // Remove convert layer if it do nothing. After type conversion pass
            // some convert layers may lose actuality.
            if (in_precision == out_precision) {
                to_remove.push_back(layer);
            }
        }
    }
    RemoveConverts(net, to_remove);
}

template <Precision::ePrecision PREC_FROM, Precision::ePrecision PREC_TO, typename NET>
void convertPrecisionForAll(NET &net) {
    auto all_layers = TopolSort(net);
    for (auto &layer : all_layers) {
        convertLayerPrecision<PREC_FROM, PREC_TO>(layer);
    }
    fixConvertLayers(net);
}

}  // namespace

bool HasInternalSubnet(const CNNLayerPtr &layer) {
    return layer->type == "TensorIterator" && dynamic_cast<TensorIterator*>(layer.get()) != nullptr;
}

details::CNNSubnet GetInternalSubnet(const CNNLayerPtr &layer) {
    if (layer->type == "TensorIterator") {
        auto ti = static_cast<TensorIterator*>(layer.get());
        IE_ASSERT(ti);
        return {ti->body.inputs, ti->body.outputs};
    }
    return {};
}

void ConvertPrecision(CNNNetwork& net, Precision from, Precision to) {
    OV_ITT_SCOPED_TASK(itt::domains::IELegacy, "NetPass::ConvertPrecision");

    auto compare = getPrecisionMask(from, to);
    switch (compare) {
        case getPrecisionMask(Precision::U32, Precision::I32):
            convertPrecisionForAll<Precision::U32, Precision::I32>(net);
            break;
        case getPrecisionMask(Precision::U64, Precision::I32):
            convertPrecisionForAll<Precision::U64, Precision::I32>(net);
            break;
        case getPrecisionMask(Precision::I64, Precision::I32):
            convertPrecisionForAll<Precision::I64, Precision::I32>(net);
            break;
        case getPrecisionMask(Precision::BOOL, Precision::U8):
            convertPrecisionForAll<Precision::BOOL, Precision::U8>(net);
            break;
        case getPrecisionMask(Precision::BOOL, Precision::I32):
            convertPrecisionForAll<Precision::BOOL, Precision::I32>(net);
            break;
        case getPrecisionMask(Precision::FP16, Precision::FP32):
            convertPrecisionForAll<Precision::FP16, Precision::FP32>(net);
            break;
        case getPrecisionMask(Precision::FP64, Precision::FP32):
            convertPrecisionForAll<Precision::FP64, Precision::FP32>(net);
            break;
        case getPrecisionMask(Precision::U8, Precision::I32):
            convertPrecisionForAll<Precision::U8, Precision::I32>(net);
            break;
        case getPrecisionMask(Precision::U16, Precision::I32):
            convertPrecisionForAll<Precision::U16, Precision::I32>(net);
            break;
        case getPrecisionMask(Precision::I16, Precision::I32):
            convertPrecisionForAll<Precision::I16, Precision::I32>(net);
            break;
        default:
            IE_THROW() << "Precision conversion from " << from << " to " << to
                               << " currently is not supported. You may expand precision"
                                  " conversion pass.";
    }
}

void ConvertIOPrecision(CNNNetwork& net, Precision from, Precision to) {
    InputsDataMap inputDataMap = net.getInputsInfo();
    for (auto & i : inputDataMap) {
        if (i.second->getPrecision() == from) {
            i.second->setPrecision(to);
        }
    }

    OutputsDataMap outputDataMap = net.getOutputsInfo();
    for (auto & i : outputDataMap) {
        if (i.second->getPrecision() == from) {
            i.second->setPrecision(to);
        }
    }
}

}  // namespace NetPass
}  // namespace InferenceEngine
