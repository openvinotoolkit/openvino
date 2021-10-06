// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <gmock/gmock-spec-builders.h>

#include <file_utils.h>

#include <memory>
#include <common_test_utils/test_assertions.hpp>
#include <details/ie_so_pointer.hpp>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace ::testing;
using ::testing::InSequence;

namespace InferenceEngine {

namespace details {

struct UnknownPlugin : std::enable_shared_from_this<UnknownPlugin> {};

template<>
class SOCreatorTrait<InferenceEngine::details::UnknownPlugin> {
public:
    static constexpr auto name = "CreateUnknownPlugin";
};

}  // namespace details

}  // namespace InferenceEngine

class SoPointerTests : public ::testing::Test {};

TEST_F(SoPointerTests, UnknownPlugin) {
    ASSERT_THROW(SOPointer<InferenceEngine::details::UnknownPlugin>{std::string{"UnknownPlugin"}}, Exception);
}

TEST_F(SoPointerTests, UnknownPluginExceptionStr) {
    try {
        SOPointer<InferenceEngine::details::UnknownPlugin>(std::string{"UnknownPlugin"});
    }
    catch (Exception &e) {
        ASSERT_STR_CONTAINS(e.what(), "Cannot load library 'UnknownPlugin':");
        ASSERT_STR_DOES_NOT_CONTAIN(e.what(), "path:");
        ASSERT_STR_DOES_NOT_CONTAIN(e.what(), "from CWD:");
    }
}
