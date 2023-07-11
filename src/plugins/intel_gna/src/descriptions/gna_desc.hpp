// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "backend/dnn_types.hpp"
#include "gna_plugin_config.hpp"
#include "ie_algorithm.hpp"
#include "ie_input_info.hpp"
#include "ie_precision.hpp"

namespace ov {
namespace intel_gna {

/*
 * This base structure accumulates all required information for network inputs and outputs
 */
struct GnaDesc {
    // common OV properties
    std::string name = "";
    std::unordered_set<std::string> tensor_names = {};
    InferenceEngine::Layout model_layout = InferenceEngine::Layout::ANY;
    InferenceEngine::SizeVector dims = {};
    InferenceEngine::Precision model_precision = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::Precision tensor_precision = InferenceEngine::Precision::UNSPECIFIED;

    // gna specific properties
    float scale_factor = kScaleFactorDefault;
    intel_dnn_orientation_t orientation = kDnnUnknownOrientation;
    uint32_t num_elements = 0;
    uint32_t allocated_size = 0;
    std::vector<void*> ptrs = {};  // ptr per each infer request

    // pre/post processing model
    std::shared_ptr<ov::Model> pre_post_process_model = nullptr;

    // help methods
    uint32_t get_required_size() const {
        return num_elements * static_cast<uint32_t>(tensor_precision.size());
    }

    uint32_t get_allocated_size() const {
        return allocated_size;
    }

    void set_precision(InferenceEngine::Precision::ePrecision precision) {
        this->tensor_precision = precision;
    }

    // helps to get the precision for gna layers, because they use num_bytes instead of precision values
    void set_precision(uint32_t num_bytes) {
        switch (num_bytes) {
        case sizeof(int8_t): {
            set_precision(InferenceEngine::Precision::I8);
            break;
        }
        case sizeof(int16_t): {
            set_precision(InferenceEngine::Precision::I16);
            break;
        }
        case sizeof(int32_t): {
            set_precision(InferenceEngine::Precision::I32);
            break;
        }
        default:
            set_precision(InferenceEngine::Precision::UNSPECIFIED);
        }
    }

    InferenceEngine::DataPtr to_ie_data() {
        OPENVINO_SUPPRESS_DEPRECATED_START
        return std::make_shared<InferenceEngine::Data>(
            name,
            InferenceEngine::TensorDesc(model_precision, dims, model_layout));
        OPENVINO_SUPPRESS_DEPRECATED_END
    }
};

/*
 * This structure accumulates all required information for one the network input
 */
struct InputDesc : GnaDesc {
    InputDesc(const std::string& name) {
        this->name = name;
    }

    void Update(const InferenceEngine::InputInfo::Ptr inputInfo) {
        this->model_precision = inputInfo->getPrecision();
        this->tensor_precision = inputInfo->getPrecision();
        this->model_layout = inputInfo->getLayout();
        this->dims = inputInfo->getTensorDesc().getDims();
        this->name = inputInfo->name();
        this->num_elements = static_cast<uint32_t>(InferenceEngine::details::product(dims.begin(), dims.end()));
    }

    InferenceEngine::InputInfo::Ptr ToIEInputInfo() {
        OPENVINO_SUPPRESS_DEPRECATED_START
        InferenceEngine::InputInfo::Ptr input_info = std::make_shared<InferenceEngine::InputInfo>();
        input_info->setInputData(this->to_ie_data());
        return input_info;
        OPENVINO_SUPPRESS_DEPRECATED_END
    }
};

/*
 * This structure accumulates all required information for one network output
 */
struct OutputDesc : GnaDesc {
    OutputDesc(const std::string& name) {
        this->name = name;
    }

    void Update(const InferenceEngine::DataPtr outputData) {
        this->model_precision = outputData->getPrecision();
        this->tensor_precision = outputData->getPrecision();
        this->model_layout = outputData->getLayout();
        this->dims = outputData->getTensorDesc().getDims();
        this->name = outputData->getName();
        this->num_elements = static_cast<uint32_t>(InferenceEngine::details::product(dims.begin(), dims.end()));
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
    GnaNetworkInfo() : infos_({}) {}

    const T& at(const std::string& key) const {
        if (key.empty()) {
            throw std::invalid_argument("The key cannot be empty");
        }
        auto desc_it = find(key);
        if (desc_it == infos_.end()) {
            throw std::out_of_range("The key cannot be found");
        }
        return *desc_it;
    }

    T& at(const std::string& key) {
        return const_cast<T&>(static_cast<const GnaNetworkInfo&>(*this).at(key));
    }

    typename std::vector<T>::iterator end() {
        return infos_.end();
    }

    typename std::vector<T>::const_iterator find(const std::string& key) const {
        return std::find_if(infos_.cbegin(), infos_.cend(), [&key](const T& desc) {
            return desc.name == key;
        });
    }

    typename std::vector<T>::iterator find(const std::string& key) {
        return std::find_if(infos_.begin(), infos_.end(), [&key](const T& desc) {
            return desc.name == key;
        });
    }

    T& operator[](const std::string& key) {
        if (key.empty()) {
            throw std::invalid_argument("The key cannot be empty");
        }
        auto desc_it = std::find_if(infos_.begin(), infos_.end(), [&key](const T& desc) {
            return desc.name == key;
        });
        if (desc_it == infos_.end()) {
            infos_.push_back(T(key));
            return infos_.back();
        }
        return *desc_it;
    }

    size_t size() const {
        return infos_.size();
    }

    bool empty() const {
        return infos_.empty();
    }

    const std::vector<T>& Get() const {
        return infos_;
    }

    std::vector<T>& Get() {
        return infos_;
    }
};

typedef GnaNetworkInfo<InputDesc> GnaInputs;
typedef GnaNetworkInfo<OutputDesc> GnaOutputs;

}  // namespace intel_gna
}  // namespace ov
