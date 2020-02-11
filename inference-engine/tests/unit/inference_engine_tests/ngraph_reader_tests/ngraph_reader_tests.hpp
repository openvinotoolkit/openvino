// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <functional>
#include <ie_ir_reader.hpp>
#include "tests_common.hpp"
#include <ie_core.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <transform/transformations/fusion/conv_bias_fusion.hpp>
#include <transform/transformations/fusion/fc_bias_fusion.hpp>
#include <transform/transformations/quantizeconv_dequantize_fusion.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <ngraph/frontend/onnx_import/onnx.hpp>
#include <ie_util_internal.hpp>
#include <cpp/ie_cnn_net_reader.h>
#include <net_pass.h>

using namespace testing;
using namespace InferenceEngine;

class NGraphReaderTests : public TestsCommon {
protected:
    void TearDown() override {}
    void SetUp() override {}

    bool compareParamVal(const std::string& val1, const std::string& val2) {
        std::vector<std::string> vals1, vals2;
        std::stringstream ss1(val1);
        std::string field;
        while (getline(ss1, field, ',' )) {
            std::stringstream fs(field);
            std::string value;
            fs >> value;
            vals1.emplace_back(value);
        }

        std::stringstream ss2(val2);
        while (getline(ss2, field, ',' )) {
            std::stringstream fs(field);
            std::string value;
            fs >> value;
            vals2.emplace_back(value);
        }

        if (vals1.size() != vals2.size())
            return false;

        for (size_t i = 0; i < vals1.size(); i++) {
            try {
                float v1 = std::stof(vals1[i]);
                float v2 = std::stof(vals2[i]);
                if (std::fabs(v2 - v1) > 0.00001f)
                    return false;
            } catch (...) {
                if (vals1[i] != vals2[i])
                    return false;
            }
        }
        return true;
    };

    template <class T>
    void compareLayerByLayer(const T& network,const T& refNetwork) {
        std::vector<std::string> err_log;

        auto newIterator = network.begin();
        auto oldIterator = refNetwork.begin();
        for (; newIterator != network.end() && oldIterator != refNetwork.end(); newIterator++, oldIterator++) {
            CNNLayerPtr layer = *newIterator;
            CNNLayerPtr oldLayer = *oldIterator;
            if (layer->type != oldLayer->type) {
                err_log.push_back("Layer " + oldLayer->name + " and old layer " + oldLayer->name + " have different type: " +
                                  layer->type + " and " + oldLayer->type);
            } else if(layer->type == "TensorIterator") {
                compareTensorIterators(layer, oldLayer);
            }
            if (layer->precision != oldLayer->precision) {
                err_log.push_back("Layer " + layer->name + " and old layer " + oldLayer->name + " have different precisions: "
                                  + layer->precision.name() + " and " + oldLayer->precision.name());
            }
            bool success = layer->type == oldLayer->type &&
                layer->insData.size() == oldLayer->insData.size() &&
                layer->outData.size() == oldLayer->outData.size() &&
                layer->precision == oldLayer->precision;

            for (size_t i = 0; i < layer->insData.size() && success; i++) {
                auto lockedOldData = oldLayer->insData[i].lock();
                auto lockedData = layer->insData[i].lock();
                success = success && lockedOldData->getTensorDesc() == lockedData->getTensorDesc();
                if (lockedOldData->getTensorDesc() != lockedData->getTensorDesc()) {
                    err_log.push_back("Layer " + layer->name + " and old layer " + oldLayer->name + " have different tensor desc for locked data");
                }
            }
            for (size_t i = 0; i < layer->outData.size() && success; i++) {
                if (oldLayer->outData[i]->getTensorDesc() != layer->outData[i]->getTensorDesc()) {
                    err_log.push_back("Layer " + layer->name + " and old layer " + oldLayer->name + " have different tensor desc");
                }
                success = success && oldLayer->outData[i]->getTensorDesc() == layer->outData[i]->getTensorDesc();
            }
            for (const auto& item : layer->params) {
                if (!success)
                    break;
                if (oldLayer->params.find(item.first) != oldLayer->params.end()) {
                    if (!compareParamVal(oldLayer->params[item.first], item.second)) {
                        success = false;
                        err_log.push_back("Layer " + layer->name + " in new network differ from reference parameter " + item.first +
                                          " (new, old): " + item.second + ", " + oldLayer->params[item.first]);
                    }
                } else {
                    success = false;
                    err_log.push_back("Layer " + oldLayer->name + " in old net has no " + item.first + " attribute.");
                }
            }

            if (!success) {
                for (auto& it : err_log) {
                    std::cout << "ERROR: " << it << std::endl;
                }
                THROW_IE_EXCEPTION << "CNNNetworks have different layers!";
            }
        }
    }

    template<class T>
    void compareInfo(T& new_info, T& old_info, const std::string& err_msg) {
        bool success = new_info.size() == old_info.size();
        for (const auto& it : new_info) {
            if (!success)
                break;
            success = success && old_info.find(it.first) != old_info.end();
        }
        if (!success)
            THROW_IE_EXCEPTION << err_msg;
    }

    void compareCNNNetworks(const CNNNetwork& network, const CNNNetwork& refNetwork) {
        if (network.layerCount() != refNetwork.layerCount())
            THROW_IE_EXCEPTION << "CNNNetworks have different numbers of layers! " << std::to_string(network.layerCount()) <<
                           " and " << std::to_string(refNetwork.layerCount());

        compareLayerByLayer<CNNNetwork>(network, refNetwork);
        InputsDataMap newInput = network.getInputsInfo();
        InputsDataMap oldInput = refNetwork.getInputsInfo();
        OutputsDataMap newOutput = network.getOutputsInfo();
        OutputsDataMap oldOutput = refNetwork.getOutputsInfo();
        compareInfo<InputsDataMap>(newInput, oldInput, "CNNNetworks have different inputs!");
        compareInfo<OutputsDataMap>(newOutput, oldOutput, "CNNNetworks have different outputs!");
    }

    void compareIRs(const std::string& modelV10, const std::string& oldModel, size_t weightsSize = 0, const std::function<void(Blob::Ptr&)>& fillBlob = {}) {
        std::string plugins_path;
#ifndef _WIN32
        plugins_path = "lib/";
#endif
        plugins_path += "plugins.xml";
        Core ie(testing::FileUtils::makePath(getIELibraryPath(), plugins_path));
        Blob::Ptr weights;

        if (weightsSize) {
            weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {weightsSize}, Layout::C));
            weights->allocate();
            fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
            if (fillBlob)
                fillBlob(weights);
        }

        auto network = ie.ReadNetwork(modelV10, weights);
        network.begin();  // Call conversion from CNNNetwork NgraphImpl to CNNNetwork
        auto cnnNetwork = ie.ReadNetwork(oldModel, weights);

        compareCNNNetworks(network, cnnNetwork);
    }

    void compareTensorIterators(const CNNLayerPtr &new_layer, const CNNLayerPtr &old_layer) {
        auto ti_new = std::dynamic_pointer_cast<InferenceEngine::TensorIterator>(new_layer);
        auto ti_old = std::dynamic_pointer_cast<InferenceEngine::TensorIterator>(old_layer);

        if(!ti_new || !ti_old) {
            THROW_IE_EXCEPTION << "Cannot cast the layer to TensorIterator.";
        }

        auto get_port_map = [](const std::vector<TensorIterator::PortMap>& port_map_list, const std::vector<DataPtr>& data_from,
                               const std::vector<DataPtr>& data_to){
            std::map<std::pair<std::string, std::string>, TensorIterator::PortMap> ordered_port_maps;
            for (auto &port_map : port_map_list) {
                ordered_port_maps[{data_from[port_map.from]->getName(), data_to[port_map.to]->getName()}] = port_map;
            }

            return std::move(ordered_port_maps);
        };

        auto get_data_ptrs = [](std::vector<DataWeakPtr> &wk_data_ptrs) {
            std::vector<DataPtr> data_ptrs;
            for(auto &wk_data : wk_data_ptrs) {
                auto data_ptr = wk_data.lock();
                IE_ASSERT(data_ptr != nullptr);
                data_ptrs.push_back(data_ptr);
            }
            return std::move(data_ptrs);
        };

        auto compare_port_maps = [](std::map<std::pair<std::string, std::string>, TensorIterator::PortMap>& new_ordered_port_maps,
                                    std::map<std::pair<std::string, std::string>, TensorIterator::PortMap>& old_ordered_port_maps) {

            if(new_ordered_port_maps.size() != old_ordered_port_maps.size()) {
                THROW_IE_EXCEPTION << "PortMaps have different numbers of layers: " << new_ordered_port_maps.size() <<
                                   " and " << old_ordered_port_maps.size();
            }

            auto iterator_new = new_ordered_port_maps.begin();
            auto iterator_old = old_ordered_port_maps.begin();

            for (; iterator_new != new_ordered_port_maps.end() && iterator_old != old_ordered_port_maps.end();
                   iterator_new++, iterator_old++) {
                if(iterator_new->first != iterator_old->first) {
                    THROW_IE_EXCEPTION << R"(Names of "from" and "to" layers in the port maps do not match!)";
                }

                TensorIterator::PortMap& pm_new = iterator_new->second;
                TensorIterator::PortMap& pm_old = iterator_old->second;

                if (pm_new.part_size != pm_old.part_size || pm_new.axis != pm_old.axis || pm_new.stride != pm_old.stride ||
                    pm_new.end != pm_old.end || pm_new.start != pm_old.start) {
                    THROW_IE_EXCEPTION << "Parameters in the port maps do not match!";
                }
            }
        };

        auto output_port_mp_new = get_port_map(ti_new->output_port_map, ti_new->outData, ti_new->body.outputs);
        auto output_port_mp_old = get_port_map(ti_old->output_port_map, ti_old->outData, ti_old->body.outputs);
        compare_port_maps(output_port_mp_new, output_port_mp_old);

        auto input_port_mp_new = get_port_map(ti_new->input_port_map, get_data_ptrs(ti_new->insData), ti_new->body.inputs);
        auto input_port_mp_old = get_port_map(ti_old->input_port_map, get_data_ptrs(ti_old->insData), ti_old->body.inputs);
        compare_port_maps(input_port_mp_new, input_port_mp_old);

        auto back_edges_mp_new = get_port_map(ti_new->back_edges, ti_new->body.outputs, ti_new->body.inputs);
        auto back_edges_mp_old = get_port_map(ti_old->back_edges, ti_old->body.outputs, ti_old->body.inputs);
        compare_port_maps(back_edges_mp_new, back_edges_mp_old);

        auto holder = ti_new->body.inputs.back();
        ti_new->body.inputs.pop_back();

        // TI body comparison
        auto nodes_new = NetPass::TIBodySortTopologically(ti_new->body);
        auto nodes_old = NetPass::TIBodySortTopologically(ti_old->body);

        std::sort(nodes_new.begin(), nodes_new.end(), [](const CNNLayerPtr& l, const CNNLayerPtr& r){
            return l->name < r->name;
        });

        std::sort(nodes_old.begin(), nodes_old.end(), [](const CNNLayerPtr& l, const CNNLayerPtr& r){
            return l->name < r->name;
        });

        compareLayerByLayer<std::vector<CNNLayerPtr>>(nodes_new, nodes_old);

        auto get_map = [](const std::vector<DataPtr>& data) -> std::map<std::string, DataPtr> {
            std::map<std::string, DataPtr> name_to_data_map;
            for(auto &it : data) {
                name_to_data_map[it->getName()] = it;
            }
            return std::move(name_to_data_map);
        };

        auto new_inputs = get_map(ti_new->body.inputs);
        auto old_inputs = get_map(ti_old->body.inputs);
        compareInfo<std::map<std::string, DataPtr>>(new_inputs, old_inputs,
                                                    "Bodies of TensorIterator have different inputs!");

        auto new_outputs = get_map(ti_new->body.outputs);
        auto old_outputs = get_map(ti_old->body.outputs);
        compareInfo<std::map<std::string, DataPtr>>(new_outputs, old_outputs,
                                                    "Bodies of TensorIterator have different outputs!");

        ti_new->body.inputs.push_back(holder);
    }
};
