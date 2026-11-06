// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/equal.hpp"
#include "openvino/op/topk.hpp"
#include "transformations/convert_precision.hpp"

#ifdef SELECTIVE_BUILD_ANALYZER
#    define SELECTIVE_BUILD_ANALYZER_ON
#    undef SELECTIVE_BUILD_ANALYZER
#elif defined(SELECTIVE_BUILD)
#    define SELECTIVE_BUILD_ON
#    undef SELECTIVE_BUILD
#endif

#define SELECTIVE_BUILD_ANALYZER

#include "itt.hpp"

namespace {
bool fuse_returns_true(const std::shared_ptr<ov::Node>&, const precisions_map&) {
    return true;
}

bool fuse_returns_false(const std::shared_ptr<ov::Node>&, const precisions_map&) {
    return false;
}
}  // namespace

TEST(conditional_compilation, collect_fuse_map_keeps_all_entries) {
#define ov_pass_fuse_type_to_TopK_v3 1
    const ov::DiscreteTypeInfo custom_op("CustomOp", "extension");
    type_to_fuse_map type_to_fuse;
    ADD_FUSE_ENTRY(type_to_fuse, TopK_v3, ov::op::v3::TopK, fuse_returns_true);
    ADD_FUSE_ENTRY(type_to_fuse, Equal_v1, ov::op::v1::Equal, fuse_returns_false);
    ADD_FUSE_ENTRY_FOR_TYPE_INFO(type_to_fuse, CustomOp, custom_op, fuse_returns_true);

    // Entries are kept regardless of macros, the wrapper only records the call
    ASSERT_EQ(type_to_fuse.size(), 3u);
    EXPECT_TRUE(type_to_fuse.at(ov::op::v3::TopK::get_type_info_static())(nullptr, {}));
    EXPECT_FALSE(type_to_fuse.at(ov::op::v1::Equal::get_type_info_static())(nullptr, {}));
    EXPECT_TRUE(type_to_fuse.at(custom_op)(nullptr, {}));
#undef ov_pass_fuse_type_to_TopK_v3
}

#undef SELECTIVE_BUILD_ANALYZER

#ifdef SELECTIVE_BUILD_ANALYZER_ON
#    define SELECTIVE_BUILD_ANALYZER
#elif defined(SELECTIVE_BUILD_ON)
#    define SELECTIVE_BUILD
#endif
