// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <stdlib.h>
#include <cfloat>
#include <cmath>
#include <stdint.h>
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include <string>

#include "data_stats.h"


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

std::vector<std::string> AggregatedDataStats::registeredLayers() {
    std::vector<std::string> layers;
    for (auto l : _data) {
        layers.push_back(l.first);
    }
    return layers;
}

void AggregatedDataStats::registerLayer(std::string layer) {
    _data[layer];
}

void AggregatedDataStats::addTensorStatistics(const std::string& name, size_t channel, float* data, size_t count) {
    auto&& byChannel = _data[name];
    byChannel[channel].push_back(TensorStatistic(data, count));
}

void AggregatedDataStats::addTensorStatistics(const std::string &name, size_t channel, uint8_t *data, size_t count) {
    std::vector<float> intermediate;
    for (size_t i = 0; i < count; i++) {
        intermediate.push_back(data[i]);
    }
    addTensorStatistics(name, channel, intermediate.data(), count);
}

size_t AggregatedDataStats::getNumberChannels(const std::string& name) const {
    auto it = _data.find(name);
    if (it != _data.end()) {
        return it->second.size();
    }
    return 0;
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
        size_t elementToTake = maxValues.size() * threshold / 100;
        int elementsToThrow = maxValues.size() - elementToTake;
        std::sort(maxValues.begin(), maxValues.end());
        std::sort(minValues.begin(), minValues.end());

        min = minValues[elementsToThrow];
        max = maxValues[elementToTake - 1];
    } else {
        min = max = 0.f;
    }
}

