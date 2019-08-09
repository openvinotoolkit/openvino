// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdlib.h>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include <stdint.h>
#include <string>

#include "data_stats.hpp"

//----- dataStats -----//
void dataStats::registerLayer(const std::string& name, size_t batch, size_t channels) {
    _registeredLayers.push_back({name, batch, channels});
}

void dataStats::addStatistics(const std::string &name, size_t channel, uint8_t *data, size_t count) {
    float* dst = new float[count];
    for (size_t i = 0lu; i < count; i++) {
        dst[i] = static_cast<float>(data[i]);
    }
    addStatistics(name, channel, dst, count);
    delete[] dst;
}

void dataStats::addStatistics(const std::string &name, size_t channel, short *data, size_t count) {
    float* dst = new float[count];
    for (size_t i = 0lu; i < count; i++) {
        dst[i] = static_cast<float>(data[i]);
    }
    addStatistics(name, channel, dst, count);
    delete[] dst;
}

//----- simpleDataStats -----//
void simpleDataStats::registerLayer(const std::string& name, size_t batch, size_t channels) {
    dataStats::registerLayer(name, batch, channels);
    _data[name];
}

size_t simpleDataStats::getNumberChannels(const std::string& name) const {
    auto it = _data.find(name);
    if (it != _data.end()) {
        return it->second.size();
    }
    return 0lu;
}

void simpleDataStats::addStatistics(const std::string& name, size_t channel, float* data, size_t count) {
    auto& byChannel = _data[name][channel];
    // TODO: Investigate synchronization of _data usage
    // add_mutex.lock();
    for (size_t i = 0lu; i < count; i++) {
        if (byChannel._min > data[i]) {
            byChannel._min = data[i];
        }

        if (byChannel._max < data[i]) {
            byChannel._max = data[i];
        }
    }
    // add_mutex.unlock();
}

void simpleDataStats::getDataMinMax(const std::string& name, size_t channel, float& min, float& max, float threshold) {
    auto it = _data.find(name);
    if (it != _data.end()) {
        min = it->second[channel]._min;
        max = it->second[channel]._max;
    } else {
        min = max = 0.f;
    }
}

//----- TensorStatistic -----//
TensorStatistic::TensorStatistic(float* data, size_t count, size_t nbuckets) {
    _min = std::numeric_limits<float>::max();
    _max = std::numeric_limits<float>::min();
    for (size_t i = 0; i < count; i++) {
        float val = static_cast<float>(data[i]);
        if (_min > val) {
            _min = val;
        }

        if (_max < val) {
            _max = val;
        }
    }

    if (_min == _max) {
        return;
    }
}

float TensorStatistic::getMaxValue() const {
    return _max;
}

float TensorStatistic::getMinValue() const {
    return _min;
}
//----- AggregatedDataStats -----//
void AggregatedDataStats::registerLayer(const std::string& name, size_t batch, size_t channels) {
    dataStats::registerLayer(name , batch, channels);
    _data[name];
}

void AggregatedDataStats::addStatistics(const std::string& name, size_t channel, float* data, size_t count) {
    auto&& byChannel = _data[name];
    byChannel[channel].push_back(TensorStatistic(data, count));
}

size_t AggregatedDataStats::getNumberChannels(const std::string& name) const {
    auto it = _data.find(name);
    if (it != _data.end()) {
        return it->second.size();
    }
    return 0lu;
}

void AggregatedDataStats::getDataMinMax(const std::string& name, size_t channel, float& min, float& max, float threshold) {
    // take data by name
    auto it = _data.find(name);
    if (it != _data.end()) {
        auto stats = it->second[channel];
        // having absolute min/max values, we can create new statistic
        std::vector<float> maxValues;
        std::vector<float> minValues;
        for (size_t i = 0; i < stats.size(); i++) {
            const TensorStatistic& tsS = stats[i];
            maxValues.push_back(tsS.getMaxValue());
            minValues.push_back(tsS.getMinValue());
        }
        // define number of elements to throw out
        size_t elementToTake = static_cast<size_t>(maxValues.size() * (threshold / 100));
        int elementsToThrow = maxValues.size() - elementToTake;
        std::sort(maxValues.begin(), maxValues.end());
        std::sort(minValues.begin(), minValues.end());

        min = minValues[elementsToThrow];
        max = maxValues[elementToTake - 1];
    } else {
        min = max = 0.f;
    }
}
