// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_common.h>
#include "cnn_network_stats_impl.hpp"
#include <memory>
#include <map>
#include <string>
#include <fstream>
#include <cassert>
#include <cfloat>
#include "debug.h"
#include <vector>

#include <pugixml.hpp>

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

string joinCommas(vector<float>& v) {
    string res;

    for (size_t i = 0; i < v.size(); ++i) {
        res += to_string(v[i]);
        if (i < v.size() - 1) {
            res += ", ";
        }
    }

    return res;
}

void CNNNetworkStatsImpl::SaveToFile(const string& xmlPath, const string& binPath) const {
    const map<string, NetworkNodeStatsPtr>& netNodesStats = this->netNodesStats;

    ofstream ofsBin(binPath, ofstream::out | ofstream::binary);

    pugi::xml_document doc;

    auto stats = doc.append_child("stats");
    stats.append_attribute("version").set_value("1");

    auto layers = stats.append_child("layers");

    size_t histOffset = 0;

    for (auto itStats : netNodesStats) {
        auto layer = layers.append_child("layer");

        layer.append_child("name").text().set(itStats.first.c_str());

        layer.append_child("min").text().set(joinCommas(itStats.second->_minOutputs).c_str());
        layer.append_child("max").text().set(joinCommas(itStats.second->_maxOutputs).c_str());
        /*layer.append_child("threshold").text().set(itStats.second->_threshold);

        auto hist = layer.append_child("histogram");
        hist.append_attribute("offset").set_value(histOffset);
        hist.append_attribute("size").set_value(itStats.second->_hist.size() * sizeof(size_t));
        hist.append_attribute("count").set_value(itStats.second->_hist.size());

        if (!itStats.second->_hist.empty()) {
            char* data = reinterpret_cast<char*>(itStats.second->_hist.data());
            int dataSize = (int) (itStats.second->_hist.size() * sizeof(size_t));

            ofsBin.write(data, dataSize);

            histOffset += dataSize;
        }*/
    }

    doc.save_file(xmlPath.c_str());
    ofsBin.close();
}

vector<float> splitParseCommas(const string& s) {
    vector<float> res;
    stringstream ss(s);

    float val;

    while (ss >> val) {
        res.push_back(val);

        if (ss.peek() == ',')
            ss.ignore();
    }

    return res;
}

void CNNNetworkStatsImpl::LoadFromFile(const string& xmlPath, const string& binPath) {
    map<string, NetworkNodeStatsPtr> newNetNodesStats;

    ifstream ifsBin(binPath, ofstream::in | ofstream::binary);

    pugi::xml_document doc;

    doc.load_file(xmlPath.c_str());

    auto stats = doc.child("stats");
    auto layers = stats.child("layers");

    NetworkNodeStatsPtr nodeStats;
    size_t offset;
    size_t size;
    size_t count;

    for (auto layer : layers.children("layer")) {
        nodeStats = NetworkNodeStatsPtr(new NetworkNodeStats());

        string name = layer.child("name").text().get();

        newNetNodesStats[name] = nodeStats;

        nodeStats->_minOutputs = splitParseCommas(layer.child("min").text().get());
        nodeStats->_maxOutputs = splitParseCommas(layer.child("max").text().get());
    }

    ifsBin.close();

    this->netNodesStats = newNetNodesStats;
}
