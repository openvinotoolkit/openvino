// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <openvino/core/function.hpp>
#include <shared_node_info.hpp>

#include <memory>
#include <string>
#include <utility>

namespace ov {
namespace test {

class TestsCommon : virtual public ::testing::Test {
protected:
    TestsCommon();
    ~TestsCommon() override;

    static std::string GetTimestamp();
    std::string GetTestName() const;
};

}  // namespace test

class FunctionAccessor {
     std::shared_ptr<Function> m_function;
public:
     FunctionAccessor(std::shared_ptr<Function> f) : m_function(std::move(f)) {}

     bool get_cache_flag() const {
         return m_function->m_shared_rt_info->get_use_topological_cache();
     }
};
}  // namespace ov
