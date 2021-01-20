// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <tuple>
#include <memory>
#include <map>

#include <details/ie_exception.hpp>
#include <ie_blob.h>
#include <ie_parameter.hpp>
#include <ie_iextension.h>
#include <ie_extension.h>
#include <exec_graph_info.hpp>

#include <ngraph/opsets/opset.hpp>

using namespace InferenceEngine;

//
// exec_graph_info.hpp
//
constexpr ngraph::NodeTypeInfo ExecGraphInfoSerialization::ExecutionNode::type_info;

const ngraph::NodeTypeInfo&
ExecGraphInfoSerialization::ExecutionNode::get_type_info() const {
    return type_info;
}

//
// ie_blob.h
//

Blob::~Blob() {}
MemoryBlob::~MemoryBlob() {}

//
// ie_iextension.h
//
ILayerImpl::~ILayerImpl() {}
ILayerExecImpl::~ILayerExecImpl() {}
std::map<std::string, ngraph::OpSet> IExtension::getOpSets() {
    return {};
}

//
// ie_extension.h
//
std::map<std::string, ngraph::OpSet> Extension::getOpSets() {
    return actual->getOpSets();
}

//
// details/ie_exception.hpp
//

details::InferenceEngineException::~InferenceEngineException() noexcept {}

details::InferenceEngineException::InferenceEngineException(const std::string& filename, const int line, const std::string& message) noexcept :
    std::exception(), _file(filename), _line(line) {
    if (!message.empty()) {
        exception_stream = std::make_shared<std::stringstream>(message);
    }
}

details::InferenceEngineException::InferenceEngineException(const InferenceEngineException& that) noexcept :
    std::exception() {
    errorDesc = that.errorDesc;
    status_code = that.status_code;
    _file = that._file;
    _line = that._line;
    exception_stream = that.exception_stream;
}
//
// ie_parameter.hpp
//

Parameter::~Parameter() {
    clear();
}

#ifdef __clang__
Parameter::Any::~Any() {}

template struct InferenceEngine::Parameter::RealData<int>;
template struct InferenceEngine::Parameter::RealData<bool>;
template struct InferenceEngine::Parameter::RealData<float>;
template struct InferenceEngine::Parameter::RealData<double>;
template struct InferenceEngine::Parameter::RealData<uint32_t>;
template struct InferenceEngine::Parameter::RealData<std::string>;
template struct InferenceEngine::Parameter::RealData<unsigned long>;
template struct InferenceEngine::Parameter::RealData<std::vector<int>>;
template struct InferenceEngine::Parameter::RealData<std::vector<std::string>>;
template struct InferenceEngine::Parameter::RealData<std::vector<unsigned long>>;
template struct InferenceEngine::Parameter::RealData<std::tuple<unsigned int, unsigned int>>;
template struct InferenceEngine::Parameter::RealData<std::tuple<unsigned int, unsigned int, unsigned int>>;
template struct InferenceEngine::Parameter::RealData<InferenceEngine::Blob::Ptr>;
#endif  // __clang__
//
// ie_blob.h
//

#ifdef __clang__
template <typename T, typename U>
TBlob<T, U>::~TBlob() {
    free();
}

template class InferenceEngine::TBlob<float>;
template class InferenceEngine::TBlob<double>;
template class InferenceEngine::TBlob<int8_t>;
template class InferenceEngine::TBlob<uint8_t>;
template class InferenceEngine::TBlob<int16_t>;
template class InferenceEngine::TBlob<uint16_t>;
template class InferenceEngine::TBlob<int32_t>;
template class InferenceEngine::TBlob<uint32_t>;
template class InferenceEngine::TBlob<long>;
template class InferenceEngine::TBlob<long long>;
template class InferenceEngine::TBlob<unsigned long>;
template class InferenceEngine::TBlob<unsigned long long>;
#endif  // __clang__
