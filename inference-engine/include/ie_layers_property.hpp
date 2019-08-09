// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file for describing property style structure used by CNNLayers
 * @file ie_layers_property.h
 */
#pragma once

#include <vector>

namespace InferenceEngine {

constexpr const int MAX_DIMS_NUMBER = 12;

enum eDIMS_AXIS : uint8_t {
    X_AXIS = 0,
    Y_AXIS,
    Z_AXIS
};

template<class T, int N = MAX_DIMS_NUMBER>
class PropertyVector {
    T _axises[N] = {};
    bool _allocated[N] = {};
    size_t _length = 0;

public:
    PropertyVector() = default;

    PropertyVector(size_t len, T val) {
        if (len > N) {
            THROW_IE_EXCEPTION << "Property size exceeed limit of: " << N;
        }
        for (size_t i = 0; i < len; i++) {
            _axises[i] = val;
            _allocated[i] = true;
        }
        _length = len;
    }

    explicit PropertyVector(const std::vector<T>& values) {
        size_t i = 0;
        for (const auto val : values) {
            insert(i++, val);
        }
    }

    PropertyVector(std::initializer_list<int> init_list) {
        size_t i = 0;
        for (const auto val : init_list) {
            insert(i++, val);
        }
    }

    /**
     * @brief allows access up-to capacity size
     * @param index
     * @return
     */
    T &at(int index) {
        if (index >= N) {
            THROW_IE_EXCEPTION << "Property index is out of bounds (" << index << "/" << N;
        }
        return _axises[index];
    }

    const T &operator[](size_t index) const {
        if (index >= N ||!_allocated[index]) {
            THROW_IE_EXCEPTION << "Property index ("<< index <<") is out of bounds";
        }
        return _axises[index];
    }

    T &operator[](size_t index) {
        if (index >= N || !_allocated[index]) {
            THROW_IE_EXCEPTION << "Property index ("<< index <<") is out of bounds";
        }
        return _axises[index];
    }

    PropertyVector &operator=(const PropertyVector &src) {
        if (this != &src) {
            _length = src.size();
            for (size_t i = 0; i < N; i++) {
                _allocated[i] = src._allocated[i];
                if (_allocated[i]) {
                    _axises[i] = src[i];
                }
            }
        }
        return *this;
    }

    bool operator==(const PropertyVector& src) const {
        if (this == &src) return true;
        if (_length != src.size()) return false;
        for (size_t i = 0; i < N; i++)
            if ((_allocated[i] != src._allocated[i]) ||
                (_allocated[i] && _axises[i] != src._axises[i])) return false;
        return true;
    }

    size_t size() const {
        return _length;
    }

    void insert(size_t  axis, const T &val) {
        if (axis < N) {
            if (!_allocated[axis]) {
                _allocated[axis] = true;
                _length++;
            }
            _axises[axis] = val;
        } else {
            THROW_IE_EXCEPTION << "Layer Property insertion at(axis) should be in [0,"<< N<< ")";
        }
    }

    void remove(size_t axis) {
        if (axis < N && _allocated[axis]) {
            _allocated[axis] = false;
            _length--;
        }
    }

    void clear() {
        for (int i = 0; i != N; i++) {
            _allocated[i] = 0;
        }
        _length = 0u;
    }

    bool exist(size_t axis) const {
        return (axis < N && _allocated[axis]);
    }
};

}  // namespace InferenceEngine
