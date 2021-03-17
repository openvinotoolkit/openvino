// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_test_net.hpp"

#include <ie_memcpy.h>
#include <blob_factory.hpp>

#include "vpu_ir_dumper.hpp"

namespace  {

std::vector<double> GetParamAsDoubles(const std::string& vals) {
    std::vector<double> result;
    std::istringstream stream(vals);
    std::string str;
    while (getline(stream, str, ',')) {
        try {
            result.push_back(std::stod(str));
        } catch (...) {
            THROW_IE_EXCEPTION << "Cannot parse parameter " << str
                               << ". Value " << vals << " cannot be casted to double.";
        }
    }
    return result;
}
template<class To>
struct PackStaticCastConverter {
    static To convert(double value) {
        return static_cast<To>(value);
    }
};

struct PackFP16Converter {
    static int16_t convert(double value) {
        return InferenceEngine::PrecisionUtils::f32tof16(static_cast<float>(value));
    }
};

template<typename T, typename Converter = PackStaticCastConverter<T>>
std::vector<uint8_t> PackData(const std::vector<double>& values) {
    std::vector<uint8_t> result(values.size() * sizeof (T));
    std::vector<T> tmp(values.size());
    std::transform(values.cbegin(), values.cend(), tmp.begin(), Converter::convert);
    ie_memcpy(result.data(), result.size(), tmp.data(), tmp.size() * sizeof(T));
    return result;
}

std::vector<uint8_t> PackData(const std::vector<double>& values, const InferenceEngine::Precision& precision) {
    if (precision == InferenceEngine::Precision::I64)
        return PackData<int64_t>(values);
    if (precision == InferenceEngine::Precision::FP16)
        return PackData<int16_t, PackFP16Converter>(values);

    THROW_IE_EXCEPTION << "unsupported pack format '" << precision << "'";
}

InferenceEngine::Layout getLayout(const IN_OUT_desc& inDim) {
    switch(inDim[0].size()) {
        case 2:
            return InferenceEngine::HW;
        case 3:
            return InferenceEngine::CHW;
        case 4:
            return InferenceEngine::NHWC;
        case 5:
            return InferenceEngine::NDHWC;
    }
    return InferenceEngine::ANY;
}

}

void VpuTestNet::ReferenceFunctionWrapper::setCallback(VpuTestNet::CallbackBasic&& f, const ParamsStruct& params) {
    if (f)
        _callback = std::bind(std::move(f), _input, _output,
                              params);
}

void VpuTestNet::ReferenceFunctionWrapper::setCallback(VpuTestNet::CallbackWithWeights&& f, const ParamsStruct& params) {
    if (f)
        _callback = std::bind(std::move(f), _input, _output,
                              _weights, _weightsSize, _biases, _biasesSize,
                              params);
}

void VpuTestNet::genInputOutput(VpuTestNet::ReferenceFunctionWrapper& obj, const LayerParams& params) {
    auto outW = params._outDim[0];
    if (_callbacks.empty()) {
        auto newW = params._inDim[0];
        const InferenceEngine::Layout inputLayout = getLayout(params._inDim);
        obj._input = InferenceEngine::make_shared_blob<uint16_t>({InferenceEngine::Precision::FP16, newW, inputLayout});
        obj._input->allocate();
    } else {
        auto val = _callbacks.back();
        ASSERT_EQ(params._inDim[0].size(), val._output->getTensorDesc().getDims().size());
        obj._input = val._output;
        auto inW = params._inDim[0];
        for (size_t i = 0; i < params._outDim[0].size(); ++i) {
            ASSERT_EQ(inW[i], val._output->getTensorDesc().getDims()[i]);
        }
    }
    const InferenceEngine::Layout outLayout = getLayout(params._outDim);
    obj._output = make_blob_with_precision(
            InferenceEngine::TensorDesc(params._outPrecision, {outW}, outLayout));
    obj._output->allocate();
}

VpuTestNet::ReferenceFunctionWrapper& VpuTestNet::addLayerImpl(const LayerParams& params) {
    _layers.push_back(params);
    ReferenceFunctionWrapper obj;
    genInputOutput(obj, params);
    obj._weightsSize= params._weightsSize;
    obj._biasesSize = params._biasesSize;
    if (params._weightsSize) {
        WeightsBlob* weights = new WeightsBlob({InferenceEngine::Precision::U8,
                                               {(params._weightsSize) * sizeof(uint16_t)},
                                               InferenceEngine::C});
        weights->allocate();
        obj._weightsPtr = WeightsBlob::Ptr(weights);
        obj._weights = weights->data().as<uint16_t *>();
    }
    if (params._biasesSize) {
        WeightsBlob* biases = new WeightsBlob({InferenceEngine::Precision::U8,
                                              {(params._biasesSize) * sizeof(uint16_t)},
                                              InferenceEngine::C});
        biases->allocate();
        obj._biasesPtr = WeightsBlob::Ptr(biases);
        obj._biases = biases->data().as<uint16_t *>();
    }
    _callbacks.push_back(obj);
    return *_callbacks.rbegin();
}

void VpuTestNet::addLayer(const VpuTestNet::LayerParams& params) {
    addLayerImpl(params);
}

void VpuTestNet::addLayer(const VpuTestNet::LayerParams& params, VpuTestNet::CallbackBasic&& callback) {
    addLayerImpl(params).setCallback(std::move(callback), params._params);
}
void VpuTestNet::addLayer(const VpuTestNet::LayerParams& params, VpuTestNet::CallbackWithWeights&& callback) {
    addLayerImpl(params).setCallback(std::move(callback), params._params);
}

void VpuTestNet::run() const {
    for (auto& elem : _callbacks) {
        if (elem._callback)
            elem._callback();
    }
}

void VpuTestNet::clear() {
    _callbacks.clear();
    _layers.clear();
}

VpuTestNet::NetworkSerializedData VpuTestNet::genNetwork(IRVersion version) {
    IE_ASSERT(!_layers.empty());
    IRDumperNetwork IRDumper(version);
    IRDumper.addInput("input"  , _layers.begin()->_inDim);

    size_t testNetIndex = 0;
    for (auto& elem : _layers) {
        auto & layer = IRDumper.addLayer(elem._layerName + "_" + std::to_string(testNetIndex),
                                         elem._layerType, elem._inDim, elem._outDim);
        layer._outputPrecision = elem._outPrecision;

        auto params = elem._params;
        if (!params.empty()) {
            if (version == IRVersion::v10) {
                static const std::map<std::string, std::vector<std::string>> constLayerParams {
                    {"Transpose", {"order"}},
                    {"Pad"      , {"pads_begin", "pads_end", "pad_value"}},
                };
                auto paramsIt = constLayerParams.find(elem._layerType);
                if (paramsIt != constLayerParams.cend()) {
                    for (const auto& paramName : paramsIt->second) {
                        if (params.find(paramName) == params.cend())
                            continue;

                        const auto paramValues = GetParamAsDoubles(params[paramName]);
                        IRWeightsDescription weights;
                        weights._precision = InferenceEngine::Precision::I64;
                        if (paramName == "pad_value")
                            weights._precision = InferenceEngine::Precision::FP16;

                        weights._data = PackData(paramValues, weights._precision);

                        weights._desc = {paramValues.size()};
                        if (paramValues.size() == 1)
                            weights._isScalar = true;

                        weights._description = paramName;
                        params.erase(paramName);
                        layer._paramWeights.emplace_back(std::move(weights));
                    }
                }
            }
            layer._dataParams = params;
        }
        if (elem._weightsSize) {
            IE_ASSERT(layer._weights.empty());
            layer._weights._data.resize(elem._weightsSize * sizeof(int16_t));
            if (!elem._weightsDim.empty())
                layer._weights._desc = elem._weightsDim[0];

            if (elem._fillWeights) {
                auto& refLayer = _callbacks[testNetIndex];
                elem._fillWeights(reinterpret_cast<uint16_t*>(layer._weights._data.data()), elem._weightsSize);
                ie_memcpy(refLayer._weights, refLayer._weightsSize * sizeof(uint16_t), layer._weights._data.data(), elem._weightsSize * sizeof(uint16_t));
            }
        }
        if (elem._biasesSize) {
            IE_ASSERT(layer._biases.empty());
            layer._biases._data.resize(elem._biasesSize * sizeof(int16_t));
            if (!elem._biasesDim.empty())
                layer._biases._desc = elem._biasesDim[0];

            if (elem._fillBiases) {
                auto& refLayer = _callbacks[testNetIndex];
                elem._fillBiases(reinterpret_cast<uint16_t*>(layer._biases._data.data()), elem._biasesSize);
                ie_memcpy(refLayer._biases, refLayer._biasesSize * sizeof(uint16_t), layer._biases._data.data(), elem._biasesSize * sizeof(uint16_t));
            }
        }
        ++testNetIndex;
    }

    IRDumper.addOutput("output", _layers.rbegin()->_outDim);
    IRDumper.finalize();

    // separate lines here for debugging purpose.
    auto modelNode = IRDumper.dump();
    auto modelText = formatXmlNode(modelNode);
    return {std::move(modelText), IRDumper.getWeights()};
}

void VpuTestNet::setWeightsCallbackForLayer(size_t index, VpuTestNet::CalcWeights&& callback) {
    _layers[index]._fillWeights = std::move(callback);
}

void VpuTestNet::setBiasesCallbackForLayer(size_t index, VpuTestNet::CalcWeights&& callback) {
    _layers[index]._fillBiases = std::move(callback);
}

InferenceEngine::Blob::Ptr VpuTestNet::getFirstInput() const {
    IE_ASSERT(!empty());
    return _callbacks.begin()->_input;
}

InferenceEngine::Blob::Ptr VpuTestNet::getLastOutput() const {
    IE_ASSERT(!empty());
    return _callbacks.rbegin()->_output;
}
