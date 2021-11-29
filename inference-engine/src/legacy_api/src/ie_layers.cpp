// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <legacy/ie_layers.h>

using namespace InferenceEngine;

CNNLayer::CNNLayer(const LayerParams& prms)
    : node(nullptr), name(prms.name), type(prms.type), precision(prms.precision), userValue({0}) {}

CNNLayer::CNNLayer(const CNNLayer& other)
    : node(other.node), name(other.name), type(other.type), precision(other.precision),
    outData(other.outData), insData(other.insData), _fusedWith(other._fusedWith),
    userValue(other.userValue), affinity(other.affinity),
    params(other.params), blobs(other.blobs) {}

LayerParams::LayerParams() {}

LayerParams::LayerParams(const std::string & name, const std::string & type, Precision precision)
    : name(name), type(type), precision(precision) {}

LayerParams::LayerParams(const LayerParams & other)
    : name(other.name), type(other.type), precision(other.precision) {}

LayerParams & LayerParams::operator= (const LayerParams & other) {
    if (&other != this) {
        name = other.name;
        type = other.type;
        precision = other.precision;
    }
    return *this;
}

WeightableLayer::WeightableLayer(const LayerParams& prms) : CNNLayer(prms) {}

const DataPtr CNNLayer::input() const {
    if (insData.empty()) {
        IE_THROW() << "Internal error: input data is empty";
    }
    auto lockedFirstInsData = insData[0].lock();
    if (!lockedFirstInsData) {
        IE_THROW() << "Internal error: unable to lock weak_ptr\n";
    }
    return lockedFirstInsData;
}

float CNNLayer::ie_parse_float(const std::string& str) {
    if (str == "-inf") {
        return -std::numeric_limits<float>::infinity();
    } else if (str == "inf") {
        return std::numeric_limits<float>::infinity();
    } else {
        float res;
        std::stringstream val_stream(str);
        val_stream.imbue(std::locale("C"));
        val_stream >> res;
        if (!val_stream.eof()) IE_THROW();
        return res;
    }
}

std::string CNNLayer::ie_serialize_float(float value) {
    std::stringstream val_stream;
    val_stream.imbue(std::locale("C"));
    val_stream << value;
    return val_stream.str();
}

float CNNLayer::GetParamAsFloat(const char* param, float def) const {
    std::string val = GetParamAsString(param, ie_serialize_float(def).c_str());
    try {
        return ie_parse_float(val);
    } catch (...) {
        IE_THROW() << "Cannot parse parameter " << param << " from IR for layer " << name << ". Value "
                           << val << " cannot be casted to float.";
    }
}

float CNNLayer::GetParamAsFloat(const char* param) const {
    std::string val = GetParamAsString(param);
    try {
        return ie_parse_float(val);
    } catch (...) {
        IE_THROW() << "Cannot parse parameter " << param << " from IR for layer " << name << ". Value "
                           << val << " cannot be casted to float.";
    }
}

std::vector<float> CNNLayer::GetParamAsFloats(const char* param, std::vector<float> def) const {
    std::string vals = GetParamAsString(param, "");
    std::vector<float> result;
    std::istringstream stream(vals);
    std::string str;
    if (vals.empty()) return def;
    while (getline(stream, str, ',')) {
        try {
            float val = ie_parse_float(str);
            result.push_back(val);
        } catch (...) {
            IE_THROW() << "Cannot parse parameter " << param << " " << str << " from IR for layer " << name
                               << ". Value " << vals << " cannot be casted to floats.";
        }
    }
    return result;
}

std::vector<float> CNNLayer::GetParamAsFloats(const char* param) const {
    std::string vals = GetParamAsString(param);
    std::vector<float> result;
    std::istringstream stream(vals);
    std::string str;
    while (getline(stream, str, ',')) {
        try {
            float val = ie_parse_float(str);
            result.push_back(val);
        } catch (...) {
            IE_THROW() << "Cannot parse parameter " << param << " " << str << " from IR for layer " << name
                               << ". Value " << vals << " cannot be casted to floats.";
        }
    }
    return result;
}

int CNNLayer::GetParamAsInt(const char* param, int def) const {
    std::string val = GetParamAsString(param, std::to_string(def).c_str());
    try {
        return std::stoi(val);
    } catch (...) {
        IE_THROW() << "Cannot parse parameter " << param << " from IR for layer " << name << ". Value "
                           << val << " cannot be casted to int.";
    }
}

int CNNLayer::GetParamAsInt(const char* param) const {
    std::string val = GetParamAsString(param);
    try {
        return std::stoi(val);
    } catch (...) {
        IE_THROW() << "Cannot parse parameter " << param << " from IR for layer " << name << ". Value "
                           << val << " cannot be casted to int.";
    }
}

std::vector<int> CNNLayer::GetParamAsInts(const char* param, std::vector<int> def) const {
    std::string vals = GetParamAsString(param, "");
    std::vector<int> result;
    std::istringstream stream(vals);
    std::string str;
    if (vals.empty()) return def;
    while (getline(stream, str, ',')) {
        try {
            result.push_back(std::stoi(str));
        } catch (...) {
            IE_THROW() << "Cannot parse parameter " << param << " " << str << " from IR for layer " << name
                               << ". Value " << vals << " cannot be casted to int.";
        }
    }
    return result;
}

std::vector<int> CNNLayer::GetParamAsInts(const char* param) const {
    std::string vals = GetParamAsString(param);
    std::vector<int> result;
    std::istringstream stream(vals);
    std::string str;
    while (getline(stream, str, ',')) {
        try {
            result.push_back(std::stoi(str));
        } catch (...) {
            IE_THROW() << "Cannot parse parameter " << param << " " << str << " from IR for layer " << name
                               << ". Value " << vals << " cannot be casted to int.";
        }
    }
    return result;
}

unsigned int CNNLayer::GetParamAsUInt(const char* param, unsigned int def) const {
    std::string val = GetParamAsString(param, std::to_string(def).c_str());
    std::string message = "Cannot parse parameter " + std::string(param) + " from IR for layer " + name +
                          ". Value " + val + " cannot be casted to unsigned int.";
    try {
        long long value = std::stoll(val);
        if ((value < 0) || (value > std::numeric_limits<unsigned int>::max())) {
            IE_THROW() << message;
        }
        return static_cast<unsigned int>(value);
    } catch (...) {
        IE_THROW() << message;
    }
}

unsigned int CNNLayer::GetParamAsUInt(const char* param) const {
    std::string val = GetParamAsString(param);
    std::string message = "Cannot parse parameter " + std::string(param) + " from IR for layer " + name +
                          ". Value " + val + " cannot be casted to unsigned int.";
    try {
        long long value = std::stoll(val);
        if ((value < 0) || (value > std::numeric_limits<unsigned int>::max())) {
            IE_THROW() << message;
        }
        return static_cast<unsigned int>(value);
    } catch (...) {
        IE_THROW() << message;
    }
}

std::vector<unsigned int> CNNLayer::GetParamAsUInts(const char* param, std::vector<unsigned int> def) const {
    std::string vals = GetParamAsString(param, "");
    std::vector<unsigned int> result;
    std::istringstream stream(vals);
    std::string str;
    std::string message = "Cannot parse parameter " + std::string(param) + " " + str + " from IR for layer " +
                          name + ". Value " + vals + " cannot be casted to unsigned int.";
    if (vals.empty()) return def;
    while (getline(stream, str, ',')) {
        try {
            long long value = std::stoll(str);
            if ((value < 0) || (value > std::numeric_limits<unsigned int>::max())) {
                IE_THROW() << message;
            }
            result.push_back(static_cast<unsigned int>(value));
        } catch (...) {
            IE_THROW() << message;
        }
    }
    return result;
}

std::vector<unsigned int> CNNLayer::GetParamAsUInts(const char* param) const {
    std::string vals = GetParamAsString(param);
    std::vector<unsigned int> result;
    std::istringstream stream(vals);
    std::string str;
    std::string message = "Cannot parse parameter " + std::string(param) + " " + str + " from IR for layer " +
                          name + ". Value " + vals + " cannot be casted to unsigned int.";
    while (getline(stream, str, ',')) {
        try {
            long long value = std::stoll(str);
            if ((value < 0) || (value > std::numeric_limits<unsigned int>::max())) {
                IE_THROW() << message;
            }
            result.push_back(static_cast<unsigned int>(value));
        } catch (...) {
            IE_THROW() << message;
        }
    }
    return result;
}

size_t CNNLayer::GetParamAsSizeT(const char* param, size_t def) const {
    std::string val = GetParamAsString(param, std::to_string(def).c_str());
    std::string message = "Cannot parse parameter " + std::string(param) + " from IR for layer " + name +
                          ". Value " + val + " cannot be casted to size_t.";
    try {
        long long value = std::stoll(val);
        if ((value < 0) || (static_cast<unsigned long long>(value) > std::numeric_limits<size_t>::max())) {
            IE_THROW() << message;
        }
        return static_cast<size_t>(value);
    } catch (...) {
        IE_THROW() << message;
    }
}

size_t CNNLayer::GetParamAsSizeT(const char* param) const {
    std::string val = GetParamAsString(param);
    std::string message = "Cannot parse parameter " + std::string(param) + " from IR for layer " + name +
                          ". Value " + val + " cannot be casted to size_t.";
    try {
        long long value = std::stoll(val);
        if ((value < 0) || (static_cast<unsigned long long>(value) > std::numeric_limits<size_t>::max())) {
            IE_THROW() << message;
        }
        return static_cast<size_t>(value);
    } catch (...) {
        IE_THROW() << message;
    }
}

bool CNNLayer::GetParamAsBool(const char* param, bool def) const {
    std::string val = GetParamAsString(param, std::to_string(def).c_str());
    std::string loweredCaseValue;
    std::transform(val.begin(), val.end(), std::back_inserter(loweredCaseValue), [](char value) {
        return static_cast<char>(std::tolower(value));
    });

    bool result = false;

    if (!(std::istringstream(loweredCaseValue) >> std::boolalpha >> result)) {
        // attempting parse using non alpha bool
        return (GetParamAsInt(param, def) != 0);
    }

    return result;
}

bool CNNLayer::GetParamAsBool(const char* param) const {
    std::string val = GetParamAsString(param);
    std::string loweredCaseValue;
    std::transform(val.begin(), val.end(), std::back_inserter(loweredCaseValue), [](char value) {
        return static_cast<char>(std::tolower(value));
    });

    bool result = false;

    if (!(std::istringstream(loweredCaseValue) >> std::boolalpha >> result)) {
        // attempting parse using non alpha bool
        return (GetParamAsInt(param) != 0);
    }

    return result;
}

std::string CNNLayer::GetParamAsString(const char* param, const char* def) const {
    auto it = params.find(param);
    if (it == params.end() || it->second.empty()) {
        return def;
    }
    return (*it).second;
}

bool CNNLayer::CheckParamPresence(const char* param) const {
    auto it = params.find(param);
    if (it == params.end()) {
        return false;
    }
    return true;
}

std::string CNNLayer::GetParamAsString(const char* param) const {
    auto it = params.find(param);
    if (it == params.end()) {
        IE_THROW() << "No such parameter name '" << param << "' for layer " << name;
    }
    return (*it).second;
}

std::string CNNLayer::getBoolStrParamAsIntStr(const char *param) const {
    std::string val = GetParamAsString(param);
    if (val == "true" || val == "True") {
        return "1";
    } else if (val == "false" || val == "False") {
        return "0";
    }
    return val;
}

std::vector<std::string> CNNLayer::GetParamAsStrings(const char* param, std::vector<std::string> def) const {
    std::string vals = GetParamAsString(param, "");
    std::vector<std::string> result;
    std::istringstream stream(vals);
    std::string str;
    if (vals.empty()) return def;
    while (getline(stream, str, ',')) {
        try {
            result.push_back(str);
        } catch (...) {
            IE_THROW() << "Cannot parse parameter " << param << " from IR for layer " << name << ".";
        }
    }
    return result;
}

CNNLayer::~CNNLayer() {}
WeightableLayer::~WeightableLayer() {}
ConvolutionLayer::~ConvolutionLayer() {}
DeconvolutionLayer::~DeconvolutionLayer() {}
DeformableConvolutionLayer::~DeformableConvolutionLayer() {}
PoolingLayer::~PoolingLayer() {}
BinaryConvolutionLayer::~BinaryConvolutionLayer() {}
FullyConnectedLayer::~FullyConnectedLayer() {}
ConcatLayer::~ConcatLayer() {}
SplitLayer::~SplitLayer() {}
NormLayer::~NormLayer() {}
SoftMaxLayer::~SoftMaxLayer() {}
GRNLayer::~GRNLayer() {}
MVNLayer::~MVNLayer() {}
ReLULayer::~ReLULayer() {}
ClampLayer::~ClampLayer() {}
ReLU6Layer::~ReLU6Layer() {}
EltwiseLayer::~EltwiseLayer() {}
CropLayer::~CropLayer() {}
ReshapeLayer::~ReshapeLayer() {}
TileLayer::~TileLayer() {}
ScaleShiftLayer::~ScaleShiftLayer() {}
TensorIterator::~TensorIterator() {}
RNNCellBase::~RNNCellBase() {}
LSTMCell::~LSTMCell() {}
GRUCell::~GRUCell() {}
RNNCell::~RNNCell() {}
RNNSequenceLayer::~RNNSequenceLayer() {}
PReLULayer::~PReLULayer() {}
PowerLayer::~PowerLayer() {}
BatchNormalizationLayer::~BatchNormalizationLayer() {}
GemmLayer::~GemmLayer() {}
PadLayer::~PadLayer() {}
GatherLayer::~GatherLayer() {}
StridedSliceLayer::~StridedSliceLayer() {}
ShuffleChannelsLayer::~ShuffleChannelsLayer() {}
DepthToSpaceLayer::~DepthToSpaceLayer() {}
SpaceToDepthLayer::~SpaceToDepthLayer() {}
SpaceToBatchLayer::~SpaceToBatchLayer() {}
BatchToSpaceLayer::~BatchToSpaceLayer() {}
SparseFillEmptyRowsLayer::~SparseFillEmptyRowsLayer() {}
SparseSegmentReduceLayer::~SparseSegmentReduceLayer() {}
ExperimentalSparseWeightedReduceLayer::~ExperimentalSparseWeightedReduceLayer() {}
SparseToDenseLayer::~SparseToDenseLayer() {}
BucketizeLayer::~BucketizeLayer() {}
ReverseSequenceLayer::~ReverseSequenceLayer() {}
OneHotLayer::~OneHotLayer() {}
RangeLayer::~RangeLayer() {}
FillLayer::~FillLayer() {}
SelectLayer::~SelectLayer() {}
BroadcastLayer::~BroadcastLayer() {}
QuantizeLayer::~QuantizeLayer() {}
MathLayer::~MathLayer() {}
ReduceLayer::~ReduceLayer() {}
TopKLayer::~TopKLayer() {}
UniqueLayer::~UniqueLayer() {}
NonMaxSuppressionLayer::~NonMaxSuppressionLayer() {}
ScatterUpdateLayer::~ScatterUpdateLayer() {}
ScatterElementsUpdateLayer::~ScatterElementsUpdateLayer() {}
ExperimentalDetectronPriorGridGeneratorLayer::~ExperimentalDetectronPriorGridGeneratorLayer() {}
ExperimentalDetectronGenerateProposalsSingleImageLayer::~ExperimentalDetectronGenerateProposalsSingleImageLayer() {}
ExperimentalDetectronTopKROIs::~ExperimentalDetectronTopKROIs() {}
