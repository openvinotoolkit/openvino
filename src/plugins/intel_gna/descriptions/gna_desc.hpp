// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <numeric>
#include <functional>
#include <unordered_set>

#include "ie_precision.hpp"
#include "ie_input_info.hpp"
#include "ie_algorithm.hpp"

#include "backend/dnn_types.h"
#include "gna_plugin_config.hpp"

namespace GNAPluginNS {

/*
 * This base structure accumulates all required information for network inputs and outputs
 */
struct GnaDesc {
    // common OV properties
    std::string name;
    std::unordered_set<std::string> tensor_names;
    InferenceEngine::Layout layout;
    InferenceEngine::SizeVector dims;
    InferenceEngine::Precision precision;
    InferenceEngine::Precision blob_precision;

    // gna specific properties
    double scale_factor = GNAPluginNS::kScaleFactorDefault;
    intel_dnn_orientation_t orientation = kDnnUnknownOrientation;
    uint32_t num_bytes_per_element = 0;
    uint32_t num_elements = 0;
    uint32_t allocated_size = 0;
    std::vector<void *> ptrs = {};  // ptr per each infer request

    // help methods
    uint32_t getRequiredSize() {
        return num_elements * num_bytes_per_element;
    }

    uint32_t getAllocatedSize() {
        return allocated_size;
    }

    void setPrecision(InferenceEngine::Precision precision) {
        this->precision = precision;
        this->num_bytes_per_element = precision.size();
    }

    InferenceEngine::DataPtr ToIEData() {
        return std::make_shared<InferenceEngine::Data>(name, InferenceEngine::TensorDesc(blob_precision, dims, layout));
    }
};

/*
 * This structure accumulates all required information for one the network input
 */
struct InputDesc : GnaDesc {
    InputDesc(const std::string &name) { this->name = name; }

    void Update(const InferenceEngine::InputInfo::Ptr inputInfo) {
        this->precision = inputInfo->getPrecision();
        this->blob_precision = inputInfo->getPrecision();
        this->layout = inputInfo->getLayout();
        this->dims = inputInfo->getTensorDesc().getDims();
        this->name = inputInfo->name();
        this->num_bytes_per_element = precision.size();
        this->num_elements = InferenceEngine::details::product(dims.begin(), dims.end());
    }

    InferenceEngine::InputInfo::Ptr ToIEInputInfo() {
        InferenceEngine::InputInfo::Ptr input_info = std::make_shared<InferenceEngine::InputInfo>();
        input_info->setInputData(this->ToIEData());
        return input_info;
    }
};

/*
 * This structure accumulates all required information for one network output
 */
struct OutputDesc : GnaDesc {
    OutputDesc(const std::string &name) { this->name = name; }

    void Update(const InferenceEngine::DataPtr outputData) {
        this->precision = outputData->getPrecision();
        this->blob_precision = outputData->getPrecision();
        this->layout = outputData->getLayout();
        this->dims = outputData->getTensorDesc().getDims();
        this->name = outputData->getName();
        this->num_bytes_per_element = precision.size();
        this->num_elements = InferenceEngine::details::product(dims.begin(), dims.end());
    }
};

/**
 * Wraps vectors of input/output structure to keep their order and simplify usage.
 * @tparam T - InputDesc/OutputDesc
 */
template <class T>
class GnaNetworkInfo {
private:
    std::vector<T> infos_;

public:
    GnaNetworkInfo(): infos_({}) { }

    const T& at(const std::string &key) const {
        if (key.empty()) {
            throw std::invalid_argument("The key cannot be empty");
        }
        auto desc_it = std::find_if(infos_.begin(), infos_.end(), [&key](const T& desc){return desc.name == key;});
        if (desc_it == infos_.end()) {
            throw std::out_of_range("The key cannot be found");
        }
        return *desc_it;
    }

    T& at(const std::string &key) {
      return const_cast<T&>( static_cast<const GnaNetworkInfo&>(*this).at(key) );
    }

    T& operator[](const std::string &key) {
        if (key.empty()) {
            throw std::invalid_argument("The key cannot be empty");
        }
        auto desc_it = std::find_if(infos_.begin(), infos_.end(), [&key](const T& desc){return desc.name == key;});
        if (desc_it == infos_.end()) {
            infos_.push_back(T(key));
            return infos_.back();
        }
        return *desc_it;
    }

    size_t size() const { return infos_.size(); }

    bool empty() const { return infos_.empty(); }

    const std::vector<T>& Get() const { return infos_; }

    std::vector<T>& Get() { return infos_; }
};

typedef GnaNetworkInfo<InputDesc> GnaInputs;
typedef GnaNetworkInfo<OutputDesc> GnaOutputs;

}  // namespace GNAPluginNS
