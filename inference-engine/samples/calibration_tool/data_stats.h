// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <map>
#include <string>

struct TensorStatistic {
    TensorStatistic(float* data, size_t count, size_t nbuckets = 1000);
    float getMaxValue() const;
    float getMinValue()const;
protected:
    float _min;
    float _max;
};

class AggregatedDataStats {
public:
    void addTensorStatistics(const std::string& name, size_t channel, float* data, size_t count);
    void addTensorStatistics(const std::string &name, size_t channel, uint8_t *data, size_t count);
    void getDataMinMax(const std::string& name, size_t channel, float& min, float& max, float threshold);
    size_t getNumberChannels(const std::string& name) const;
    std::vector <std::string> registeredLayers();
    void registerLayer(std::string layer);
protected:
    std::map<std::string, std::map<size_t, std::vector<TensorStatistic> > > _data;
};

