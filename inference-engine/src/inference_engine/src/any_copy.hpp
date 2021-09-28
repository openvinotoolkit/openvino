// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_parameter.hpp>
#include <istream>
#include <map>
#include <openvino/core/shared_any.hpp>
#include <openvino/runtime/common.hpp>
#include <openvino/runtime/parameter.hpp>
#include <ostream>
#include <string>

namespace ov {

ie::Parameter any_copy(const SharedAny& shared_any);
ie::Parameter any_copy(const SharedAnyImpl::Ptr& shared_any_impl);
runtime::ConfigMap any_copy(const std::map<std::string, std::string>& config_map);

struct AnyCopyIEParamter {
    operator SharedAny();
    operator std::shared_ptr<SharedAnyImpl>();
    const ie::Parameter& parameter;
};
AnyCopyIEParamter any_copy(const ie::Parameter& parameter);

struct AnyCopyConfigMap {
    operator ie::ParamMap();
    std::string to_string(const SharedAny& shared_any);
    operator std::map<std::string, std::string>();
    const runtime::ConfigMap& config_map;
};
AnyCopyConfigMap any_copy(const runtime::ConfigMap& config_map);

}  // namespace ov
