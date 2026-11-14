// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/variadic_split.hpp"
#include "pass/lower_set_rows_stateless.hpp"

namespace {

using namespace ov::op;

class GGUFSliceSharedKVReads : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::frontend::gguf::pass::SliceSharedKVReads>();
        comparator.enable(FunctionsComparator::ACCURACY);
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }
};

std::shared_ptr<v0::Constant> i64(std::vector<int64_t> values) {
    return v0::Constant::create(ov::element::i64, {values.size()}, values);
}

// One cache [1, ctx, 2, 4] read for its first `length` tokens by `readers` attention layers, the form
// translate_kv_view emits.
std::shared_ptr<ov::Model> kv_reads(size_t readers, bool sliced) {
    const auto cache = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 2, 4});
    const auto length = std::make_shared<v0::Parameter>(ov::element::i64, ov::Shape{1});
    ov::OutputVector reads;
    for (size_t i = 0; i < readers; ++i) {
        if (sliced) {
            const auto rows = std::make_shared<v1::Reshape>(cache, i64({1, -1, 8}), false);
            const auto slice = std::make_shared<v8::Slice>(rows, i64({0}), length, i64({1}), i64({1}));
            reads.push_back(std::make_shared<v1::Reshape>(slice, i64({1, -1, 2, 4}), false));
        } else {
            const auto lengths = std::make_shared<v0::Concat>(ov::OutputVector{length, i64({-1})}, 0);
            const auto split =
                std::make_shared<v1::VariadicSplit>(cache, v0::Constant::create(ov::element::i64, {}, {1}), lengths);
            reads.push_back(split->output(0));
        }
    }
    return std::make_shared<ov::Model>(reads, ov::ParameterVector{cache, length});
}

TEST_F(GGUFSliceSharedKVReads, SharedCacheReadsBecomeSlices) {
    model = kv_reads(3, false);
    model_ref = kv_reads(3, true);
}

TEST_F(GGUFSliceSharedKVReads, SingleCacheReadKeepsVariadicSplit) {
    model = kv_reads(1, false);
}

}  // namespace
