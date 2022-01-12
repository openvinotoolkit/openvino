// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "memory_desc/cpu_memory_desc.h"

namespace MKLDNNPlugin {
class PortConfig {
public:
    PortConfig() = default;

    PortConfig(const PortConfig& rhs) {
        this->_constant = rhs._constant;
        this->_inPlacePort = rhs._inPlacePort;
        if (rhs._desc) {
            this->_desc = rhs._desc;
        }
    }

    PortConfig& operator=(const PortConfig& rhs) {
        this->_constant = rhs._constant;
        this->_inPlacePort = rhs._inPlacePort;
        if (rhs._desc) {
            this->_desc = rhs._desc;
        }
        return *this;
    }

    PortConfig(PortConfig&& rhs) = default;
    PortConfig& operator=(PortConfig&& rhs) = default;

    int inPlace() const {
        return _inPlacePort;
    }

    void inPlace(int port) {
        _inPlacePort = port;
    }

    bool constant() const {
        return _constant;
    }

    void constant(bool constant) {
        _constant = constant;
    }

    MemoryDescPtr getMemDesc() const {
        return _desc;
    }

    void setMemDesc(MemoryDescPtr desc) {
        _desc = desc;
    }

private:
    bool _constant = false;
    int _inPlacePort = -1;
    MemoryDescPtr _desc;
};

struct NodeConfig {
    bool dynBatchSupport = false;
    std::vector<PortConfig> inConfs;
    std::vector<PortConfig> outConfs;
};
} // namespace MKLDNNPlugin