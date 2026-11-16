// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <numeric>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/runtime/core.hpp"
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
    cache->set_friendly_name("cache");
    cache->output(0).set_names({"cache"});
    const auto length = std::make_shared<v0::Parameter>(ov::element::i64, ov::Shape{1});
    length->set_friendly_name("length");
    length->output(0).set_names({"length"});
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

TEST(GGUFSliceSharedKVReadsRuntime, SharedCacheSlicesMatchVariadicSplit) {
    auto reference = kv_reads(3, false);
    auto transformed = kv_reads(3, false);
    ov::frontend::gguf::pass::SliceSharedKVReads pass;
    ASSERT_TRUE(pass.run_on_model(transformed));

    ov::Core core;
    auto reference_model = core.compile_model(reference, "CPU");
    auto transformed_model = core.compile_model(transformed, "CPU");
    std::vector<float> cache_values(40);
    std::iota(cache_values.begin(), cache_values.end(), 0.0f);
    ov::Tensor cache(ov::element::f32, {1, 5, 2, 4});
    std::copy(cache_values.begin(), cache_values.end(), cache.data<float>());

    for (const int64_t length : {0, 3}) {
        ov::Tensor read_length(ov::element::i64, {1});
        read_length.data<int64_t>()[0] = length;
        auto reference_request = reference_model.create_infer_request();
        auto transformed_request = transformed_model.create_infer_request();
        reference_request.set_tensor(reference_model.input("cache"), cache);
        reference_request.set_tensor(reference_model.input("length"), read_length);
        transformed_request.set_tensor(transformed_model.input("cache"), cache);
        transformed_request.set_tensor(transformed_model.input("length"), read_length);
        reference_request.infer();
        transformed_request.infer();

        for (size_t output = 0; output < 3; ++output) {
            const auto expected = reference_request.get_output_tensor(output);
            const auto actual = transformed_request.get_output_tensor(output);
            EXPECT_EQ(actual.get_shape(), (ov::Shape{1, static_cast<size_t>(length), 2, 4}));
            ASSERT_EQ(actual.get_shape(), expected.get_shape());
            for (size_t i = 0; i < expected.get_size(); ++i) {
                EXPECT_FLOAT_EQ(actual.data<const float>()[i], expected.data<const float>()[i]);
            }
        }
    }
}

}  // namespace
