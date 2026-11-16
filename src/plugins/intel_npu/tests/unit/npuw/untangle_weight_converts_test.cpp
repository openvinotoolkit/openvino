// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <set>

#include "openvino/op/ops.hpp"
#include "partitioning/patterns/opt.hpp"

namespace {

using namespace ov;

struct SharedWeightModel {
    std::shared_ptr<Model> model;
    std::shared_ptr<op::v0::Constant> weight;
    std::shared_ptr<op::v0::Convert> convert;
    std::vector<std::shared_ptr<op::v1::Add>> readers;
};

SharedWeightModel make_model(std::size_t reader_count) {
    SharedWeightModel m;
    m.weight = op::v0::Constant::create(element::u8, Shape{4}, {0, 1, 127, 255});
    m.convert = std::make_shared<op::v0::Convert>(m.weight, element::f32);
    m.convert->set_friendly_name("weight_convert");
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
    ResultVector results;
    for (std::size_t i = 0; i < reader_count; ++i) {
        auto add = std::make_shared<op::v1::Add>(input, m.convert);
        add->set_friendly_name("reader_" + std::to_string(i));
        m.readers.push_back(add);
        results.push_back(std::make_shared<op::v0::Result>(add));
    }
    m.model = std::make_shared<Model>(results, ParameterVector{input});
    return m;
}

TEST(UntangleWeightConvertsTest, GivesEachReaderIndependentConvertWithSharedWeight) {
    auto m = make_model(3);
    m.convert->get_rt_info()["test_attribute"] = std::string("preserved");

    npuw::patterns::opt::untangleWeightConverts(m.model);

    std::set<const Node*> converts;
    std::set<std::string> names;
    for (const auto& reader : m.readers) {
        const auto convert = as_type_ptr<op::v0::Convert>(reader->input_value(1).get_node_shared_ptr());
        ASSERT_NE(convert, nullptr);
        converts.insert(convert.get());
        names.insert(convert->get_friendly_name());
        EXPECT_EQ(convert->input_value(0).get_node_shared_ptr(), m.weight);
        EXPECT_EQ(convert->get_destination_type(), element::f32);
        EXPECT_EQ(convert->get_shape(), Shape{4});
        EXPECT_EQ(convert->output(0).get_target_inputs().size(), 1u);
        EXPECT_EQ(convert->get_rt_info().at("test_attribute").as<std::string>(), "preserved");
    }
    EXPECT_EQ(converts.size(), 3u);
    EXPECT_EQ(names.size(), 3u);
    EXPECT_EQ(m.readers.front()->input_value(1).get_node_shared_ptr(), m.convert);
    EXPECT_NO_THROW(m.model->validate_nodes_and_infer_types());
}

TEST(UntangleWeightConvertsTest, PreservesNumericalOutputs) {
    auto m = make_model(3);
    Tensor input(element::f32, Shape{4});
    for (std::size_t i = 0; i < input.get_size(); ++i) {
        input.data<float>()[i] = static_cast<float>(i) - 2.f;
    }
    TensorVector before(3), after(3);
    for (std::size_t i = 0; i < before.size(); ++i) {
        before[i] = Tensor(element::f32, Shape{4});
        after[i] = Tensor(element::f32, Shape{4});
    }
    ASSERT_TRUE(m.model->evaluate(before, TensorVector{input}));

    npuw::patterns::opt::untangleWeightConverts(m.model);

    ASSERT_TRUE(m.model->evaluate(after, TensorVector{input}));
    for (std::size_t i = 0; i < after.size(); ++i) {
        for (std::size_t j = 0; j < input.get_size(); ++j) {
            EXPECT_EQ(after[i].data<float>()[j], before[i].data<float>()[j]);
        }
    }
}

TEST(UntangleWeightConvertsTest, WeightCutOffDoesNotModifyOtherReaders) {
    auto m = make_model(2);
    npuw::patterns::opt::untangleWeightConverts(m.model);
    auto first = m.readers[0]->input_value(1).get_node_shared_ptr();
    auto second = m.readers[1]->input_value(1).get_node_shared_ptr();
    auto closure = std::make_shared<op::v0::Parameter>(element::u8, Shape{4});

    first->input(0).replace_source_output(closure);

    EXPECT_EQ(first->input_value(0).get_node_shared_ptr(), closure);
    EXPECT_EQ(second->input_value(0).get_node_shared_ptr(), m.weight);
    auto first_model = std::make_shared<Model>(ResultVector{m.model->get_results()[0]},
                                               ParameterVector{m.model->get_parameters()[0], closure});
    auto second_model =
        std::make_shared<Model>(ResultVector{m.model->get_results()[1]}, ParameterVector{m.model->get_parameters()[0]});
    EXPECT_NO_THROW(first_model->validate_nodes_and_infer_types());
    EXPECT_NO_THROW(second_model->validate_nodes_and_infer_types());
}

TEST(UntangleWeightConvertsTest, LeavesSingleReaderUnchangedAndIsIdempotent) {
    for (std::size_t count : {1u, 3u}) {
        auto m = make_model(count);
        npuw::patterns::opt::untangleWeightConverts(m.model);
        NodeVector converts;
        for (const auto& reader : m.readers) {
            converts.push_back(reader->input_value(1).get_node_shared_ptr());
        }

        npuw::patterns::opt::untangleWeightConverts(m.model);

        for (std::size_t i = 0; i < count; ++i) {
            EXPECT_EQ(m.readers[i]->input_value(1).get_node_shared_ptr(), converts[i]);
        }
        EXPECT_EQ(converts.front(), m.convert);
    }
}

TEST(UntangleWeightConvertsTest, LeavesActivationConvertShared) {
    auto input = std::make_shared<op::v0::Parameter>(element::u8, Shape{4});
    auto convert = std::make_shared<op::v0::Convert>(input, element::f32);
    auto add = std::make_shared<op::v1::Add>(convert, convert);
    auto model = std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(add)}, ParameterVector{input});

    npuw::patterns::opt::untangleWeightConverts(model);

    EXPECT_EQ(add->input_value(0).get_node_shared_ptr(), convert);
    EXPECT_EQ(add->input_value(1).get_node_shared_ptr(), convert);
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
}

TEST(UntangleWeightConvertsTest, HandlesMultiplePortsOfSameReader) {
    auto m = make_model(1);
    m.readers[0]->input(0).replace_source_output(m.convert);

    npuw::patterns::opt::untangleWeightConverts(m.model);

    EXPECT_NE(m.readers[0]->input_value(0), m.readers[0]->input_value(1));
    for (const auto& input : m.readers[0]->input_values()) {
        EXPECT_EQ(input.get_node_shared_ptr()->input_value(0).get_node_shared_ptr(), m.weight);
    }
    EXPECT_NO_THROW(m.model->validate_nodes_and_infer_types());
}

TEST(UntangleWeightConvertsTest, ProducesStableNamesForEquivalentGraphs) {
    auto first = make_model(3);
    auto second = make_model(3);
    npuw::patterns::opt::untangleWeightConverts(first.model);
    npuw::patterns::opt::untangleWeightConverts(second.model);

    for (std::size_t i = 0; i < first.readers.size(); ++i) {
        EXPECT_EQ(first.readers[i]->input_value(1).get_node()->get_friendly_name(),
                  second.readers[i]->input_value(1).get_node()->get_friendly_name());
    }
}

}  // namespace
