// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock.h>
#include "ie_icore.hpp"
#include "plugin.hpp"
#include <iostream>
#include <type_traits>

using namespace MockMultiDevicePlugin;
namespace MockMultiDevice {
static constexpr ov::Property<bool, ov::PropertyMutability::RW> exclusive_async_requests{"EXCLUSIVE_ASYNC_REQUESTS"};
enum class TestParamType {
    VALID = 0,
    INVALID = 1,
    DEFAULT = 2
};

class BaseGenerator {
public:
    using Ptr = std::shared_ptr<BaseGenerator>;
    virtual std::vector<ov::Any> f(TestParamType& ptype) = 0;
};
// PropertyTypeValidator ensures that value can be converted to given property type
template<typename T>
class PropertyValueGenerator : public BaseGenerator {
public:
    std::vector<ov::Any> f(TestParamType& ptype) override {
        return generate<T>(ptype);
    }
    PropertyValueGenerator(const ov::Any& default_v = {}) : default_value(default_v) {
    }
template <typename U = T,
          typename std::enable_if<std::is_same<U, std::string>::value, bool>::type = true>
    std::vector<ov::Any> generate(TestParamType& ptype) {
        switch (ptype) {
            case TestParamType::VALID:
                return {"xyz", "/test_cache/"};
            case TestParamType::INVALID:
                return {};
            case TestParamType::DEFAULT:
                return {default_value};
            default:
                return {};
        }
    }

template <typename U = T,
          typename std::enable_if<std::is_same<U, const char*>::value, bool>::type = true>
    std::vector<ov::Any> generate(TestParamType& ptype) {
        switch (ptype) {
            case TestParamType::VALID:
                return {"CPU", "GPU", "CPU,GPU", "VPUX,mock", "TEMPLATE", "HETERO", "CPU(2),GPU(3)"}; // WA, need to update when whitelist removed
            case TestParamType::INVALID:
                return {"NONE", "_", "INVALID", "HDDL", "MYRIAD-2.0"};
            case TestParamType::DEFAULT:
                return {default_value};
            default:
                return {};
        }
    }

template<typename U = T, typename std::enable_if<std::is_same<U, bool>::value, bool>::type = true>
    std::vector<ov::Any> generate(TestParamType& ptype) {
        switch (ptype) {
            case TestParamType::VALID:
                return {true, false};
            case TestParamType::INVALID:
                return {"NONE", "_", "INVALID", 2, 3};
            case TestParamType::DEFAULT:
                return {default_value};
            default:
                return {};
        }
    }
template<typename U = T, typename std::enable_if<std::is_integral<U>::value, bool>::type = true>
    std::vector<ov::Any> generate(TestParamType& ptype) const {
        switch (ptype) {
            case TestParamType::VALID:
                return {0, 2147483647};
            case TestParamType::INVALID:
                return {"NONE", "_", "INVALID", -1};
            case TestParamType::DEFAULT:
                return {default_value};
            default:
                return {};
        }
    }

template<typename U = T, typename std::enable_if<std::is_same<U, ov::hint::Priority>::value, bool>::type = true>
    std::vector<ov::Any> generate(TestParamType& ptype) const {
        switch (ptype) {
            case TestParamType::VALID:
                return {ov::hint::Priority::HIGH, ov::hint::Priority::MEDIUM, ov::hint::Priority::LOW};
            case TestParamType::INVALID:
                return {"NONE", "_", "INVALID", -1};
            case TestParamType::DEFAULT:
                return {default_value};
            default:
                return {};
        }
    }

template<typename U = T, typename std::enable_if<std::is_same<U, ov::log::Level>::value, bool>::type = true>
    std::vector<ov::Any> generate(TestParamType& ptype) const {
        switch (ptype) {
            case TestParamType::VALID:
                return {ov::log::Level::NO, ov::log::Level::ERR, ov::log::Level::DEBUG, ov::log::Level::INFO, ov::log::Level::TRACE, ov::log::Level::WARNING};
            case TestParamType::INVALID:
                return {"NONE", "_", "INVALID", -1};
            case TestParamType::DEFAULT:
                return {default_value};
            default:
                return {};
        }
    }

template<typename U = T, typename std::enable_if<std::is_same<U, ov::hint::PerformanceMode>::value, bool>::type = true>
    std::vector<ov::Any> generate(TestParamType& ptype) const {
        switch (ptype) {
            case TestParamType::VALID:
                return {ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT,
                        ov::hint::PerformanceMode::LATENCY,
                        ov::hint::PerformanceMode::THROUGHPUT,
                        ov::hint::PerformanceMode::UNDEFINED};
            case TestParamType::INVALID:
                return {"NONE", "_", "INVALID", -1};
            case TestParamType::DEFAULT:
                return {default_value};
            default:
                return {};
        }
    }

template<typename U = T, typename std::enable_if<std::is_same<U, std::vector<std::string>>::value, bool>::type = true>
    std::vector<ov::Any> generate(TestParamType& ptype) const {
        switch (ptype) {
            case TestParamType::VALID:
                return {{"fp32", "fp16", "i8"}};
            case TestParamType::INVALID:
                return {};
            case TestParamType::DEFAULT:
                return {default_value};
            default:
                return {};
        }
    }

private:
    ov::Any default_value;
};

template <TestParamType pType>
class ParamSet {
public:
    ParamSet() = default;
    template <typename T,
              ov::PropertyMutability mutability>
    void register_param_impl(const ov::Property<T, mutability>& property, TestParamType param_type, std::shared_ptr<BaseGenerator> value_generator) {
        std::vector<ov::Any> values = value_generator->f(param_type);
        std::map<ov::PropertyName, ov::Any> res;
        for (const auto &iter : values) {
            res = {{ov::PropertyName(property.name(), mutability), iter}};
            res_param.push_back(res);
        }
    }

    void register_param_impl(const ov::device::Priorities& priorities, TestParamType param_type, std::shared_ptr<BaseGenerator> value_generator) {
        std::vector<ov::Any> values = value_generator->f(param_type);
        std::map<ov::PropertyName, ov::Any> res;
        for (const auto &iter : values) {
            res = {{ov::PropertyName(priorities.name(), priorities.mutability), iter}};
            res_param.push_back(res);
        }
    }

    template <TestParamType param_type, typename... Args, typename std::enable_if<(sizeof...(Args) == 0), bool>::type = true>
    void register_param_impl() { }

    template <TestParamType param_type,
              typename T,
              ov::PropertyMutability mutability,
              typename ValueT,
              typename... PropertyInitializer>
    void register_param_impl(const std::tuple<ov::Property<T, mutability>, ValueT>& property, PropertyInitializer&&... properties) {
        auto p = std::get<0>(property);
        auto v = std::dynamic_pointer_cast<BaseGenerator>(std::make_shared<PropertyValueGenerator<T>>(std::get<1>(property)));
        register_param_impl<T, mutability>(std::move(p), param_type, std::move(v));
        register_param_impl<param_type>(properties...);
    }
    template <TestParamType param_type,
              typename ValueT,
              typename... PropertyInitializer>
    void register_param_impl(const std::tuple<ov::device::Priorities, ValueT>& property, PropertyInitializer&&... properties) {
        auto p = std::get<0>(property);
        auto v = std::dynamic_pointer_cast<BaseGenerator>(std::make_shared<PropertyValueGenerator<ValueT>>(std::get<1>(property)));
        register_param_impl(std::move(p), param_type, std::move(v));
        register_param_impl<param_type>(properties...);
    }
    template <TestParamType param_type, typename... Args>
    void register_param(Args&&... properties) {
        register_param_impl<param_type>(properties...);
    }
    std::vector<std::map<ov::PropertyName, ov::Any>> get_params() {
        register_param<pType>(
            std::make_tuple(ov::device::full_name, ""),
            std::make_tuple(ov::device::capabilities, ""),
            std::make_tuple(ov::enable_profiling, false),
            std::make_tuple(ov::hint::performance_mode, ov::hint::PerformanceMode::UNDEFINED),
            std::make_tuple(ov::hint::num_requests, 0),
            std::make_tuple(ov::hint::model_priority, ov::hint::Priority::MEDIUM),
            std::make_tuple(ov::intel_auto::device_bind_buffer, false),
            //std::make_tuple(exclusive_async_requests, false),  // not supported in supported_properties
            std::make_tuple(ov::log::level, ov::log::Level::NO),
            // special args population function
            std::make_tuple(ov::device::priorities, ""),
            // below to be removed when core implementation ready
            std::make_tuple(ov::cache_dir, ""),
            std::make_tuple(ov::hint::allow_auto_batching, true),
            std::make_tuple(ov::auto_batch_timeout, 1000));
        // generate params with invalid key
        if (pType == TestParamType::INVALID) {
             // Read Only
            res_param.push_back({{ov::PropertyName(ov::device::architecture.name(), ov::device::architecture.mutability), nullptr}});
             // Read Write
            res_param.push_back({{ov::PropertyName(ov::compilation_num_threads.name(), ov::compilation_num_threads.mutability), nullptr}});
        }
        return res_param;
    }

private:
    std::vector<std::map<ov::PropertyName, ov::Any>> res_param;
    //std::vector<std::map<ov::PropertyName, ov::Any>> default_param;
};
}// namespace MockMultiDevice
