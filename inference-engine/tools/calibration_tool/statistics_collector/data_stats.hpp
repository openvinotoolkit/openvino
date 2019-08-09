// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <unordered_map>
#include <string>
#include <vector>

struct TensorStatistic {
    TensorStatistic(float* data, size_t count, size_t nbuckets = 1000);
    float getMaxValue() const;
    float getMinValue()const;
protected:
    float _min;
    float _max;
};

class dataStats {
public:
    struct layerInfo {
        std::string _name;
        size_t _batch;
        size_t _channels;
    };

    virtual void addStatistics(const std::string& name, size_t channel, float* data, size_t count) = 0;
    void addStatistics(const std::string& name, size_t channel, short* data, size_t count);
    void addStatistics(const std::string& name, size_t channel, uint8_t* data, size_t count);
    virtual void registerLayer(const std::string& name, size_t batch, size_t channels);
    inline const std::list<layerInfo>& registeredLayers() const {
        return _registeredLayers;
    }
    virtual void getDataMinMax(const std::string& name, size_t channel, float& min, float& max, float threshold = 100.f) = 0;
    virtual size_t getNumberChannels(const std::string& name) const = 0;
protected:
    std::list<layerInfo> _registeredLayers;
    std::mutex add_mutex;
};

class simpleDataStats : public dataStats {
public:
    void addStatistics(const std::string& name, size_t channel, float* data, size_t count);
    void registerLayer(const std::string& name, size_t batch, size_t channels);
    size_t getNumberChannels(const std::string& name) const;
    void getDataMinMax(const std::string& name, size_t channel, float& min, float& max, float threshold = 100.f);
protected:
    struct statsPair {
        float _min = std::numeric_limits<float>::max();;
        float _max = std::numeric_limits<float>::min();
    };
    std::unordered_map<std::string, std::unordered_map<size_t, statsPair>> _data;
};

class AggregatedDataStats : public dataStats {
public:
    typedef std::unordered_map<std::string, std::map<size_t, std::vector<TensorStatistic>>> internalData;

    void addStatistics(const std::string& name, size_t channel, float* data, size_t count);
    void getDataMinMax(const std::string& name, size_t channel, float& min, float& max, float threshold = 100.f);
    size_t getNumberChannels(const std::string& name) const;
    void registerLayer(const std::string& name, size_t batch, size_t channels);
protected:
    internalData _data;
};
