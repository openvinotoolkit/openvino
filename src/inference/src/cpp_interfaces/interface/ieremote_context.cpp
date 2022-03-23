// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp_interfaces/interface/iremote_context.hpp"
#include "cpp_interfaces/interface/itensor.hpp"
#include "ie_blob.h"
#include "ie_ngraph_utils.hpp"
#include "ie_remote_blob.hpp"
#include "openvino/core/except.hpp"

namespace ov {

std::string IRemoteContext::get_device_name() const {
    OPENVINO_NOT_IMPLEMENTED;
}

ITensor::Ptr IRemoteContext::create_tensor(const element::Type type, const Shape& shape, const AnyMap& params) {
    OPENVINO_NOT_IMPLEMENTED;
}

ITensor::Ptr IRemoteContext::create_host_tensor(const element::Type element_type, const Shape& shape) {
    OPENVINO_NOT_IMPLEMENTED;
}

AnyMap IRemoteContext::get_params() const {
    OPENVINO_NOT_IMPLEMENTED;
}

IERemoteContext::IERemoteContext(const InferenceEngine::RemoteContext::Ptr& impl_) : impl{impl_} {}

std::string IERemoteContext::get_device_name() const {
    return impl->getDeviceName();
}

ITensor::Ptr IERemoteContext::create_tensor(const element::Type type, const Shape& shape, const AnyMap& params) {
    auto blob =
        impl->CreateBlob({ie::details::convertPrecision(type), shape, ie::TensorDesc::getLayoutByRank(shape.size())},
                         params);
    blob->allocate();
    return blob_to_tensor(blob);
}

AnyMap IERemoteContext::get_params() const {
    return impl->getParams();
}

ITensor::Ptr IERemoteContext::create_host_tensor(const element::Type type, const Shape& shape) {
    auto blob = impl->CreateHostBlob(
        {ie::details::convertPrecision(type), shape, ie::TensorDesc::getLayoutByRank(shape.size())});
    blob->allocate();
    return blob_to_tensor(blob);
}
}  // namespace ov

namespace InferenceEngine {
std::string OVRemoteContext::getDeviceName() const noexcept {
    return impl->get_device_name();
}
RemoteBlob::Ptr OVRemoteContext::CreateBlob(const TensorDesc& tensorDesc, const ParamMap& params) {
    return std::dynamic_pointer_cast<RemoteBlob>(ov::tensor_to_blob(
        impl->create_tensor(details::convertPrecision(tensorDesc.getPrecision()), tensorDesc.getDims(), params)));
}
MemoryBlob::Ptr OVRemoteContext::CreateHostBlob(const TensorDesc& tensorDesc) {
    return std::dynamic_pointer_cast<RemoteBlob>(ov::tensor_to_blob(
        impl->create_host_tensor(details::convertPrecision(tensorDesc.getPrecision()), tensorDesc.getDims())));
}
ParamMap OVRemoteContext::getParams() const {
    return impl->get_params();
}
}  // namespace InferenceEngine