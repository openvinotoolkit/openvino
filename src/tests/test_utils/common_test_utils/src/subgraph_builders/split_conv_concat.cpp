// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/subgraph_builders/split_conv_concat.hpp"

#include "common_test_utils/node_builders/convolution.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_split_conv_concat(ov::Shape input_shape, ov::element::Type type) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(type, input_shape)};
    params.front()->set_friendly_name("Param_1");
    params.front()->get_output_tensor(0).set_names({"input_tensor"});

    auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto split = std::make_shared<ov::op::v1::Split>(params[0], split_axis_op, 2);

    auto conv1 = ov::test::utils::make_convolution(split->output(0),
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    auto relu1 = std::make_shared<ov::op::v0::Relu>(conv1);

    auto conv2 = ov::test::utils::make_convolution(split->output(1),
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);

    auto relu2 = std::make_shared<ov::op::v0::Relu>(conv2);

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{relu1->output(0), relu2->output(0)}, 1);
    concat->get_output_tensor(0).set_names({"concat_tensor"});

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat)};

    std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(results, params);
    model->set_friendly_name("SplitConvConcat");
    return model;
}

std::shared_ptr<ov::Model> make_cplit_conv_concat_input_in_branch(ov::Shape input_shape, ov::element::Type type) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(type, input_shape),
                               std::make_shared<ov::op::v0::Parameter>(type, input_shape)};
    auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto split = std::make_shared<ov::op::v1::Split>(params[0], split_axis_op, 2);

    auto conv1 = ov::test::utils::make_convolution(split->output(0),
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    auto relu1 = std::make_shared<ov::op::v0::Relu>(conv1);

    auto conv2 = ov::test::utils::make_convolution(split->output(1),
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    auto relu2 = std::make_shared<ov::op::v0::Relu>(conv2);

    auto conv4 = ov::test::utils::make_convolution(params[1]->output(0),
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    auto relu4 = std::make_shared<ov::op::v0::Relu>(conv4);

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{relu4->output(0), relu2->output(0)}, 1);

    auto conv3 = ov::test::utils::make_convolution(concat,
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {0, 0},
                                                   {0, 0},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    auto relu3 = std::make_shared<ov::op::v0::Relu>(conv3);

    auto concat1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{relu1->output(0), relu3->output(0)}, 1);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat1)};

    std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(results, params);
    model->set_friendly_name("SplitConvConcatInputInBranch");
    return model;
}

std::shared_ptr<ov::Model> make_cplit_conv_concat_nested_in_branch(ov::Shape input_shape, ov::element::Type type) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(type, input_shape),
                               std::make_shared<ov::op::v0::Parameter>(type, input_shape)};

    int localId = 0;
#define SET_NAME(node) node->set_friendly_name(#node + std::to_string(localId++));

    auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});

    auto split = std::make_shared<ov::op::v1::Split>(params[0], split_axis_op, 2);
    SET_NAME(split);

    auto conv1 = ov::test::utils::make_convolution(split->output(0),
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    SET_NAME(conv1);

    auto relu1 = std::make_shared<ov::op::v0::Relu>(conv1);
    SET_NAME(relu1);

    auto conv2 = ov::test::utils::make_convolution(split->output(1),
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    SET_NAME(conv2);

    auto relu2 = std::make_shared<ov::op::v0::Relu>(conv2);
    SET_NAME(relu2);

    auto nested_subgraph = [&] {
        auto split_axis_op =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
        auto split = std::make_shared<ov::op::v1::Split>(params[1], split_axis_op, 2);
        SET_NAME(split);

        auto conv1 = ov::test::utils::make_convolution(split->output(0),
                                                       type,
                                                       {3, 3},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       ov::op::PadType::EXPLICIT,
                                                       5);
        SET_NAME(conv1);

        auto relu1 = std::make_shared<ov::op::v0::Relu>(conv1);
        SET_NAME(relu1);

        auto conv2 = ov::test::utils::make_convolution(split->output(1),
                                                       type,
                                                       {3, 3},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       ov::op::PadType::EXPLICIT,
                                                       10);
        SET_NAME(conv2);

        auto relu2 = std::make_shared<ov::op::v0::Relu>(conv2);
        SET_NAME(relu2);

        auto split2_axis_op =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
        auto split2 = std::make_shared<ov::op::v1::Split>(relu2, split2_axis_op, 2);
        SET_NAME(split2);

        auto conv3 = ov::test::utils::make_convolution(split2->output(0),
                                                       type,
                                                       {3, 3},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       ov::op::PadType::EXPLICIT,
                                                       5);
        SET_NAME(conv3);

        auto relu3 = std::make_shared<ov::op::v0::Relu>(conv3);
        SET_NAME(relu3);

        auto conv4 = ov::test::utils::make_convolution(split2->output(1),
                                                       type,
                                                       {3, 3},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       ov::op::PadType::EXPLICIT,
                                                       5);
        SET_NAME(conv4);

        auto relu4 = std::make_shared<ov::op::v0::Relu>(conv4);
        SET_NAME(relu4);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{relu3->output(0), relu4->output(0)}, 1);
        SET_NAME(concat);

        auto concat1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{relu1->output(0), concat}, 1);
        SET_NAME(concat1);

        auto conv5 = ov::test::utils::make_convolution(concat1,
                                                       type,
                                                       {3, 3},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       ov::op::PadType::EXPLICIT,
                                                       5);
        SET_NAME(conv5);

        auto relu5 = std::make_shared<ov::op::v0::Relu>(conv5);
        SET_NAME(relu5);

        return relu5;
    }();

    auto concat =
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{nested_subgraph->output(0), relu2->output(0)}, 1);
    SET_NAME(concat);

    auto conv3 = ov::test::utils::make_convolution(concat,
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    SET_NAME(conv3);

    auto relu3 = std::make_shared<ov::op::v0::Relu>(conv3);
    SET_NAME(relu3);

    auto concat1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{relu1->output(0), relu3->output(0)}, 1);
    SET_NAME(concat1);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat1)};

    std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(results, params);
    model->set_friendly_name("SplitConvConcatNestedInBranch");
    return model;
}

std::shared_ptr<ov::Model> make_cplit_conv_concat_nested_in_branch_nested_out(ov::Shape input_shape,
                                                                              ov::element::Type type) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(type, input_shape),
                               std::make_shared<ov::op::v0::Parameter>(type, input_shape)};

    int localId = 0;
#define SET_NAME(node) node->set_friendly_name(#node + std::to_string(localId++));

    auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto split = std::make_shared<ov::op::v1::Split>(params[0], split_axis_op, 2);
    SET_NAME(split);

    auto conv1 = ov::test::utils::make_convolution(split->output(0),
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    SET_NAME(conv1);

    auto relu1 = std::make_shared<ov::op::v0::Relu>(conv1);
    SET_NAME(relu1);

    auto conv2 = ov::test::utils::make_convolution(split->output(1),
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   10);
    SET_NAME(conv2);

    auto relu2 = std::make_shared<ov::op::v0::Relu>(conv2);
    SET_NAME(relu2);

    auto split3_axis_op =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto split3 = std::make_shared<ov::op::v1::Split>(relu2, split3_axis_op, 2);
    SET_NAME(split3);

    auto conv32 = ov::test::utils::make_convolution(split3->output(1),
                                                    type,
                                                    {3, 3},
                                                    {1, 1},
                                                    {1, 1},
                                                    {1, 1},
                                                    {1, 1},
                                                    ov::op::PadType::EXPLICIT,
                                                    10);
    SET_NAME(conv32);

    auto relu32 = std::make_shared<ov::op::v0::Relu>(conv32);
    SET_NAME(relu32);

    auto nested_subgraph = [&] {
        auto split_axis_op =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
        auto split = std::make_shared<ov::op::v1::Split>(params[1], split_axis_op, 2);
        SET_NAME(split);

        auto conv1 = ov::test::utils::make_convolution(split->output(0),
                                                       type,
                                                       {3, 3},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       ov::op::PadType::EXPLICIT,
                                                       5);
        SET_NAME(conv1);

        auto relu1 = std::make_shared<ov::op::v0::Relu>(conv1);
        SET_NAME(relu1);

        auto conv2 = ov::test::utils::make_convolution(split->output(1),
                                                       type,
                                                       {3, 3},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       ov::op::PadType::EXPLICIT,
                                                       10);
        SET_NAME(conv2);

        auto relu2 = std::make_shared<ov::op::v0::Relu>(conv2);
        SET_NAME(relu2);

        auto split2_axis_op =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
        auto split2 = std::make_shared<ov::op::v1::Split>(relu2, split2_axis_op, 2);
        SET_NAME(split2);

        auto conv3 = ov::test::utils::make_convolution(split2->output(0),
                                                       type,
                                                       {3, 3},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       ov::op::PadType::EXPLICIT,
                                                       5);
        SET_NAME(conv3);

        auto relu3 = std::make_shared<ov::op::v0::Relu>(conv3);
        SET_NAME(relu3);

        auto conv4 = ov::test::utils::make_convolution(split2->output(1),
                                                       type,
                                                       {3, 3},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       ov::op::PadType::EXPLICIT,
                                                       5);
        SET_NAME(conv4);

        auto relu4 = std::make_shared<ov::op::v0::Relu>(conv4);
        SET_NAME(relu4);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{relu3->output(0), relu4->output(0)}, 1);
        SET_NAME(concat);

        auto concat1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{relu1->output(0), concat}, 1);
        SET_NAME(concat1);

        auto conv5 = ov::test::utils::make_convolution(concat1,
                                                       type,
                                                       {3, 3},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       ov::op::PadType::EXPLICIT,
                                                       5);
        SET_NAME(conv5);

        auto relu5 = std::make_shared<ov::op::v0::Relu>(conv5);
        SET_NAME(relu5);
        return relu5;
    }();

    auto nested_subgraph1 = [&] {
        auto split_axis_op =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
        auto split = std::make_shared<ov::op::v1::Split>(relu32, split_axis_op, 2);
        SET_NAME(split);

        auto conv1 = ov::test::utils::make_convolution(split->output(0),
                                                       type,
                                                       {3, 3},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       ov::op::PadType::EXPLICIT,
                                                       5);
        SET_NAME(conv1);

        auto relu1 = std::make_shared<ov::op::v0::Relu>(conv1);
        SET_NAME(relu1);

        auto conv2 = ov::test::utils::make_convolution(split->output(1),
                                                       type,
                                                       {3, 3},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       ov::op::PadType::EXPLICIT,
                                                       10);
        SET_NAME(conv2);

        auto relu2 = std::make_shared<ov::op::v0::Relu>(conv2);
        SET_NAME(relu2);

        auto split2_axis_op =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
        auto split2 = std::make_shared<ov::op::v1::Split>(relu2, split2_axis_op, 2);
        SET_NAME(split2);

        auto conv3 = ov::test::utils::make_convolution(split2->output(0),
                                                       type,
                                                       {3, 3},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       ov::op::PadType::EXPLICIT,
                                                       5);
        SET_NAME(conv3);

        auto relu3 = std::make_shared<ov::op::v0::Relu>(conv3);
        SET_NAME(relu3);

        auto conv4 = ov::test::utils::make_convolution(split2->output(1),
                                                       type,
                                                       {3, 3},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       ov::op::PadType::EXPLICIT,
                                                       5);
        SET_NAME(conv4);

        auto relu4 = std::make_shared<ov::op::v0::Relu>(conv4);
        SET_NAME(relu4);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{relu3->output(0), relu4->output(0)}, 1);
        SET_NAME(concat);

        auto concat1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{relu1->output(0), concat}, 1);
        SET_NAME(concat1);

        auto conv5 = ov::test::utils::make_convolution(concat1,
                                                       type,
                                                       {3, 3},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       {1, 1},
                                                       ov::op::PadType::EXPLICIT,
                                                       5);
        SET_NAME(conv5);

        auto relu5 = std::make_shared<ov::op::v0::Relu>(conv5);
        SET_NAME(relu5);
        return relu5;
    }();

    auto concat =
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{nested_subgraph->output(0), split3->output(0)}, 1);
    SET_NAME(concat);

    auto conv3 = ov::test::utils::make_convolution(concat,
                                                   type,
                                                   {3, 3},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   {1, 1},
                                                   ov::op::PadType::EXPLICIT,
                                                   5);
    SET_NAME(conv3);

    auto relu3 = std::make_shared<ov::op::v0::Relu>(conv3);
    SET_NAME(relu3);

    auto concat1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{relu1->output(0), relu3->output(0)}, 1);
    SET_NAME(concat1);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat1),
                             std::make_shared<ov::op::v0::Result>(nested_subgraph1)};

    std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(results, params);
    model->set_friendly_name("SplitConvConcatNestedInBranchNestedOut");
    return model;
}
}  // namespace utils
}  // namespace test
}  // namespace ov