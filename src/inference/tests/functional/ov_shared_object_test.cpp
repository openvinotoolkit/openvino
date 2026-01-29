// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/file_utils.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace ov::test {
class SharedObjectOVTests : public ::testing::Test {
protected:
    void loadDll(const std::filesystem::path& library_name) {
        shared_object = ov::util::load_shared_object(library_name);
    }
    std::shared_ptr<void> shared_object;

    using CreateF = void(std::shared_ptr<ov::IPlugin>&);

    std::function<CreateF> make_std_function(const std::string& functionName) {
        std::function<CreateF> ptr(
            reinterpret_cast<CreateF*>(ov::util::get_symbol(shared_object, functionName.c_str())));
        return ptr;
    }
};

TEST_F(SharedObjectOVTests, canLoadExistedPlugin) {
    loadDll(utils::get_mock_engine_path());
    EXPECT_NE(nullptr, shared_object.get());
}

TEST_F(SharedObjectOVTests, loaderThrowsIfNoPlugin) {
    EXPECT_THROW(loadDll("wrong_name"), std::runtime_error);
}

TEST_F(SharedObjectOVTests, canFindExistedMethod) {
    loadDll(utils::get_mock_engine_path());

    auto factory = make_std_function(ov::create_plugin_function);
    EXPECT_NE(nullptr, factory);
}

TEST_F(SharedObjectOVTests, throwIfMethodNofFoundInLibrary) {
    loadDll(utils::get_mock_engine_path());
    EXPECT_THROW(make_std_function("wrong_function"), std::runtime_error);
}

TEST_F(SharedObjectOVTests, canCallExistedMethod) {
    loadDll(utils::get_mock_engine_path());

    auto factory = make_std_function(ov::create_plugin_function);
    std::shared_ptr<ov::IPlugin> ptr;
    EXPECT_NO_THROW(factory(ptr));
}
}  // namespace ov::test
