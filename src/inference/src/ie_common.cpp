// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_common.h"

#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "exec_graph_info.hpp"
#include "ie_blob.h"
#include "ie_extension.h"
#include "ie_iextension.h"
#include "ie_parameter.hpp"
#include "ngraph/opsets/opset.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/exception.hpp"

namespace InferenceEngine {
IE_SUPPRESS_DEPRECATED_START

//
// ie_iextension.h
//
ILayerImpl::~ILayerImpl() {}
ILayerExecImpl::~ILayerExecImpl() {}
std::map<std::string, ngraph::OpSet> IExtension::getOpSets() {
    return {};
}

namespace details {

void Rethrow() {
    try {
        throw;
    } catch (const ov::NotImplemented& e) {
        IE_THROW(NotImplemented) << e.what();
    } catch (const InferenceEngine::GeneralError& e) {
        throw e;
    } catch (const InferenceEngine::NotImplemented& e) {
        throw e;
    } catch (const InferenceEngine::NetworkNotLoaded& e) {
        throw e;
    } catch (const InferenceEngine::ParameterMismatch& e) {
        throw e;
    } catch (const InferenceEngine::NotFound& e) {
        throw e;
    } catch (const InferenceEngine::OutOfBounds& e) {
        throw e;
    } catch (const InferenceEngine::Unexpected& e) {
        throw e;
    } catch (const InferenceEngine::RequestBusy& e) {
        throw e;
    } catch (const InferenceEngine::ResultNotReady& e) {
        throw e;
    } catch (const InferenceEngine::NotAllocated& e) {
        throw e;
    } catch (const InferenceEngine::InferNotStarted& e) {
        throw e;
    } catch (const InferenceEngine::NetworkNotRead& e) {
        throw e;
    } catch (const InferenceEngine::InferCancelled& e) {
        throw e;
    } catch (const ov::Cancelled& e) {
        IE_THROW(InferCancelled) << e.what();
    } catch (const std::exception& e) {
        IE_THROW() << e.what();
    } catch (...) {
        IE_THROW(Unexpected);
    }
}

IE_SUPPRESS_DEPRECATED_START

StatusCode InferenceEngineException::getStatus() const {
    if (dynamic_cast<const GeneralError*>(this) != nullptr || dynamic_cast<const ::ov::Exception*>(this) != nullptr) {
        return GENERAL_ERROR;
    } else if (dynamic_cast<const NotImplemented*>(this) != nullptr ||
               dynamic_cast<const ::ov::NotImplemented*>(this) != nullptr) {
        return NOT_IMPLEMENTED;
    } else if (dynamic_cast<const NetworkNotLoaded*>(this) != nullptr) {
        return NETWORK_NOT_LOADED;
    } else if (dynamic_cast<const ParameterMismatch*>(this) != nullptr) {
        return PARAMETER_MISMATCH;
    } else if (dynamic_cast<const NotFound*>(this) != nullptr) {
        return NOT_FOUND;
    } else if (dynamic_cast<const OutOfBounds*>(this) != nullptr) {
        return OUT_OF_BOUNDS;
    } else if (dynamic_cast<const Unexpected*>(this) != nullptr) {
        return UNEXPECTED;
    } else if (dynamic_cast<const RequestBusy*>(this) != nullptr) {
        return REQUEST_BUSY;
    } else if (dynamic_cast<const ResultNotReady*>(this) != nullptr) {
        return RESULT_NOT_READY;
    } else if (dynamic_cast<const NotAllocated*>(this) != nullptr) {
        return NOT_ALLOCATED;
    } else if (dynamic_cast<const InferNotStarted*>(this) != nullptr) {
        return INFER_NOT_STARTED;
    } else if (dynamic_cast<const NetworkNotRead*>(this) != nullptr) {
        return NETWORK_NOT_READ;
    } else if (dynamic_cast<const InferCancelled*>(this) != nullptr) {
        return INFER_CANCELLED;
    } else {
        assert(!"Unreachable");
        return OK;
    }
}
}  // namespace details
IE_SUPPRESS_DEPRECATED_END

}  // namespace InferenceEngine
