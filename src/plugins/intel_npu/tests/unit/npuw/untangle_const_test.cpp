// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/ops.hpp"
#include "partitioning/patterns/opt.hpp"

// Tests for ov::npuw::patterns::opt::untangleConst.
//
// Regression: the copies were built from the original Constant's non-owning
// get_tensor_view(). Once NPUW freed the original (e.g. a folded ShapeOf value),
// the copies read freed memory. On Gemma-3n this turned the sliding-window mask
// Slice start from 0 into garbage, and prefill compilation failed with
// "BitwiseAnd ... boolean[1,1,1024,1024], Reshape ... boolean[0,1]".

namespace {

using namespace ov;

struct SharedConstModel {
    std::shared_ptr<ov::Model> model;
    std::vector<std::shared_ptr<op::v1::Add>> readers;
};

// Param(i64)[shape] --+--> Add_0 --> Result
//                     |--> Add_1 --> Result
//                     ...
// Const(type)[shape] -'  (one Constant shared by all Adds)
SharedConstModel make_shared_const_model(const element::Type& type,
                                         const Shape& shape,
                                         const std::vector<int64_t>& values,
                                         size_t num_readers) {
    auto shared = op::v0::Constant::create(type, shape, values);
    shared->set_friendly_name("shared_const");

    SharedConstModel m;
    ParameterVector params;
    ResultVector results;
    for (size_t i = 0; i < num_readers; ++i) {
        auto param = std::make_shared<op::v0::Parameter>(type, shape);
        auto add = std::make_shared<op::v1::Add>(param, shared);
        params.push_back(param);
        results.push_back(std::make_shared<op::v0::Result>(add));
        m.readers.push_back(add);
    }
    m.model = std::make_shared<ov::Model>(results, params, "untangle_const_test");
    return m;
}

std::shared_ptr<op::v0::Constant> const_input(const std::shared_ptr<ov::Node>& reader) {
    return ov::as_type_ptr<op::v0::Constant>(reader->input_value(1).get_node_shared_ptr());
}

TEST(UntangleConstTest, EachReaderGetsItsOwnConstant) {
    auto m = make_shared_const_model(element::i64, Shape{1}, {0}, 3);
    const auto original = const_input(m.readers[0]);

    ov::npuw::patterns::opt::untangleConst(m.model);

    EXPECT_EQ(const_input(m.readers[0]), original);
    for (size_t i = 1; i < m.readers.size(); ++i) {
        const auto copy = const_input(m.readers[i]);
        ASSERT_NE(copy, nullptr);
        EXPECT_NE(copy, original);
        EXPECT_EQ(copy->output(0).get_target_inputs().size(), 1u);
        EXPECT_EQ(copy->get_friendly_name().rfind("NPUW::Untangled_Const", 0), 0u) << copy->get_friendly_name();
        EXPECT_EQ(copy->get_element_type(), element::i64);
        EXPECT_EQ(copy->get_shape(), Shape{1});
        EXPECT_EQ(copy->cast_vector<int64_t>(), std::vector<int64_t>{0});
    }
    EXPECT_NO_THROW(m.model->validate_nodes_and_infer_types());
}

TEST(UntangleConstTest, CopiesDoNotAliasOriginalData) {
    auto m = make_shared_const_model(element::i64, Shape{1}, {42}, 3);
    const auto original = const_input(m.readers[0]);

    ov::npuw::patterns::opt::untangleConst(m.model);

    for (size_t i = 1; i < m.readers.size(); ++i) {
        EXPECT_NE(const_input(m.readers[i])->get_data_ptr(), original->get_data_ptr());
    }
}

TEST(UntangleConstTest, CopiesKeepValueAfterOriginalIsReleased) {
    auto m = make_shared_const_model(element::i64, Shape{1}, {1024}, 3);
    std::weak_ptr<op::v0::Constant> original = const_input(m.readers[0]);

    ov::npuw::patterns::opt::untangleConst(m.model);

    // Detach the only remaining reader of the original, so that it gets destroyed
    m.readers[0]->input(1).replace_source_output(op::v0::Constant::create(element::i64, Shape{1}, {7}));
    m.model->get_ordered_ops();  // drop the model's cached reference to the original
    ASSERT_TRUE(original.expired());

    for (size_t i = 1; i < m.readers.size(); ++i) {
        EXPECT_EQ(const_input(m.readers[i])->cast_vector<int64_t>(), std::vector<int64_t>{1024});
    }
}

TEST(UntangleConstTest, ScalarConstIsUntangled) {
    auto m = make_shared_const_model(element::i64, Shape{}, {5}, 2);
    const auto original = const_input(m.readers[0]);

    ov::npuw::patterns::opt::untangleConst(m.model);

    const auto copy = const_input(m.readers[1]);
    EXPECT_NE(copy, original);
    EXPECT_NE(copy->get_data_ptr(), original->get_data_ptr());
    EXPECT_EQ(copy->get_shape(), Shape{});
    EXPECT_EQ(copy->cast_vector<int64_t>(), std::vector<int64_t>{5});
}

TEST(UntangleConstTest, NonSingleElementConstIsKeptShared) {
    auto m = make_shared_const_model(element::i64, Shape{2}, {0, 1}, 2);
    const auto original = const_input(m.readers[0]);

    ov::npuw::patterns::opt::untangleConst(m.model);

    EXPECT_EQ(const_input(m.readers[1]), original);
}

TEST(UntangleConstTest, NonI64ConstIsKeptShared) {
    auto m = make_shared_const_model(element::i32, Shape{1}, {0}, 2);
    const auto original = const_input(m.readers[0]);

    ov::npuw::patterns::opt::untangleConst(m.model);

    EXPECT_EQ(const_input(m.readers[1]), original);
}

}  // namespace
