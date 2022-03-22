// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp_interfaces/interface/iremote_context.hpp"

namespace ov {


std::string RemoteContext::get_device_name() const {
    OV_REMOTE_CONTEXT_STATEMENT(return _impl->getDeviceName());
}

ITensor::Ptr IRemoteContext::create_tensor(const element::Type& type, const Shape& shape, const AnyMap& params) {
    OV_REMOTE_CONTEXT_STATEMENT({
        auto blob = _impl->CreateBlob(
            {ie::details::convertPrecision(type), shape, ie::TensorDesc::getLayoutByRank(shape.size())},
            params);
        blob->allocate();
        return {blob, _so};
    });
}

ITensor::Ptr IRemoteContext::create_host_tensor(const element::Type element_type, const Shape& shape) {
    OV_REMOTE_CONTEXT_STATEMENT({
        auto blob = _impl->CreateHostBlob(
            {ie::details::convertPrecision(element_type), shape, ie::TensorDesc::getLayoutByRank(shape.size())});
        blob->allocate();
        return {blob, _so};
    });
}

AnyMap IRemoteContext::get_params() const {
    AnyMap paramMap;
    OV_REMOTE_CONTEXT_STATEMENT({
        for (auto&& param : _impl->getParams()) {
            paramMap.emplace(param.first, Any{param.second, _so});
        }
    });
    return paramMap;
}


}  // namespace ov