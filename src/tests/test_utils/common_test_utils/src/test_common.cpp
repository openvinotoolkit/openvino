// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_common.hpp"

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"

#ifdef ENABLE_CONFORMANCE_PGQL
#    include "common_test_utils/postgres_link.hpp"
#endif

namespace ov {
namespace test {

TestsCommon::~TestsCommon() {
    ov::threading::executor_manager()->clear();

#ifdef ENABLE_CONFORMANCE_PGQL
    delete PGLink;
    PGLink = nullptr;
#endif
}

TestsCommon::TestsCommon()
#ifdef ENABLE_CONFORMANCE_PGQL
    : PGLink(new utils::PostgreSQLLink(this))
#endif
{
#ifndef __APPLE__  // TODO: add getVmSizeInKB() for Apple platform
    auto memsize = ov::test::utils::getVmSizeInKB();
    if (memsize != 0) {
        std::cout << "\nMEM_USAGE=" << memsize << "KB\n";
    }
#endif
    ov::threading::executor_manager()->clear();
}

std::string TestsCommon::GetTimestamp() {
    return ov::test::utils::GetTimestamp();
}

std::string TestsCommon::GetTestName() const {
    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::replace_if(
        test_name.begin(),
        test_name.end(),
        [](char c) {
            return !std::isalnum(c);
        },
        '_');
    return test_name;
}

std::string TestsCommon::GetFullTestName() const {
    std::string suite_name = ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name();
    std::replace_if(
        suite_name.begin(),
        suite_name.end(),
        [](char c) {
            return !std::isalnum(c);
        },
        '_');

    std::string test_name = GetTestName();

    return suite_name + "_" + test_name;
}

std::filesystem::path UnicodePathTest::get_path_param() const {
    return std::visit(
        [](const auto& p) {
            // Use OV util to hide some platform details with path creation
            return ov::util::make_path(p);
        },
        GetParam());
}
std::filesystem::path UnicodePathTest::fs_path_from_variant() const {
    return std::visit(
        [](const auto& p) {
            return std::filesystem::path(p);
        },
        GetParam());
}

INSTANTIATE_TEST_SUITE_P(string_paths,
                         UnicodePathTest,
                         testing::Values("test_encoder/test_encoder.encrypted/",
                                         "test_encoder/test_encoder.encrypted"));

INSTANTIATE_TEST_SUITE_P(u16_paths,
                         UnicodePathTest,
                         testing::Values(u"test_encoder/dot.folder", u"test_encoder/dot.folder/"));

INSTANTIATE_TEST_SUITE_P(u32_paths,
                         UnicodePathTest,
                         testing::Values(U"test_encoder/dot.folder", U"test_encoder/dot.folder/"));

INSTANTIATE_TEST_SUITE_P(wstring_paths,
                         UnicodePathTest,
                         testing::Values(L"test_encoder/test_encoder.encrypted",
                                         L"test_encoder/test_encoder.encrypted/"));

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
INSTANTIATE_TEST_SUITE_P(
    unicode_paths,
    UnicodePathTest,
    testing::Values("这是.folder", L"这是_folder", L"这是_folder/", u"这是_folder/", U"这是_folder/"));
#endif

}  // namespace test
std::shared_ptr<SharedRTInfo> ModelAccessor::get_shared_info() const {
    if (auto f = m_function.lock()) {
        return f->m_shared_rt_info;
    }
    OPENVINO_THROW("Original model is not available");
}

std::set<std::shared_ptr<SharedRTInfo>> NodeAccessor::get_shared_info() const {
    if (auto node = m_node.lock()) {
        return node->m_shared_rt_info;
    }
    OPENVINO_THROW("Original node is not available");
}
}  // namespace ov
