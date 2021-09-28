// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "any_copy.hpp"

namespace ov {

ie::Parameter any_copy(const SharedAny& shared_any) {
#define COPY_TYPE(...)                         \
    if (shared_any.is<__VA_ARGS__>()) {        \
        return {shared_any.is<__VA_ARGS__>()}; \
    }
    COPY_TYPE(ie::Blob::Ptr);
    COPY_TYPE(int);
    COPY_TYPE(bool);
    COPY_TYPE(float);
    COPY_TYPE(uint32_t);
    COPY_TYPE(std::string);
    COPY_TYPE(unsigned long);
    COPY_TYPE(std::vector<int>);
    COPY_TYPE(std::vector<std::string>);
    COPY_TYPE(std::vector<unsigned long>);
    COPY_TYPE(std::tuple<unsigned int, unsigned int>);
    COPY_TYPE(std::tuple<unsigned int, unsigned int, unsigned int>);
#undef COPY_TYPE
    OPENVINO_UNREACHABLE("No type found ", shared_any.get_type_info().name());
}

ie::Parameter any_copy(const SharedAnyImpl::Ptr& shared_any_impl) {
    return any_copy(SharedAny{shared_any_impl});
}

AnyCopyIEParamter::operator SharedAny() {
#define COPY_TYPE(...)                        \
    if (parameter.is<__VA_ARGS__>()) {        \
        return {parameter.as<__VA_ARGS__>()}; \
    }
    COPY_TYPE(ie::Blob::Ptr);
    COPY_TYPE(int);
    COPY_TYPE(bool);
    COPY_TYPE(float);
    COPY_TYPE(uint32_t);
    COPY_TYPE(std::string);
    COPY_TYPE(unsigned long);
    COPY_TYPE(std::vector<int>);
    COPY_TYPE(std::vector<std::string>);
    COPY_TYPE(std::vector<unsigned long>);
    COPY_TYPE(std::tuple<unsigned int, unsigned int>);
    COPY_TYPE(std::tuple<unsigned int, unsigned int, unsigned int>);
#undef COPY_TYPE
    OPENVINO_UNREACHABLE("No type found");
}

AnyCopyIEParamter::operator std::shared_ptr<SharedAnyImpl>() {
    return operator SharedAny().get();
}

AnyCopyIEParamter any_copy(const ie::Parameter& parameter) {
    return {parameter};
}

AnyCopyConfigMap::operator ie::ParamMap() {
    ie::ParamMap result;
    for (auto&& value : config_map) {
        result.emplace(value.first, any_copy(value.second));
    }
    return result;
}

std::string AnyCopyConfigMap::to_string(const SharedAny& shared_any) {
    std::stringstream strm;
    shared_any.print(strm);
    return strm.str();
}

AnyCopyConfigMap::operator std::map<std::string, std::string>() {
    std::map<std::string, std::string> result;
    for (auto&& value : config_map) {
        result.emplace(value.first, to_string(value.second));
    }
    return result;
}

AnyCopyConfigMap any_copy(const runtime::ConfigMap& config_map) {
    return {config_map};
}

runtime::ConfigMap any_copy(const std::map<std::string, std::string>& config_map) {
    runtime::ConfigMap result;
    for (auto&& value : config_map) {
        result.emplace(value.first, SharedAny{value.second});
    }
    return result;
}
}  // namespace ov
