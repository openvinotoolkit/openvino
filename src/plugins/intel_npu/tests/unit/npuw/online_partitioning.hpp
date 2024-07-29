// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <iostream>

#include "partitioning/online/compiler.hpp"
#include "partitioning/online/snapshot.hpp"

#include "openvino/openvino.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cosh.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/op_types.hpp"

namespace {

bool isEqualEns(ov::npuw::Ensemble& ens1, ov::npuw::Ensemble& ens2) {
    if (ens1.groups.size() != ens2.groups.size()) {
        return false;
    }

    for (auto& g : ens1.groups) {
        std::sort(g.input_layers.begin(), g.input_layers.end());
        std::sort(g.output_layers.begin(), g.output_layers.end());
        std::sort(g.all_layers.begin(), g.all_layers.end());
    }

    for (auto& g : ens2.groups) {
        std::sort(g.input_layers.begin(), g.input_layers.end());
        std::sort(g.output_layers.begin(), g.output_layers.end());
        std::sort(g.all_layers.begin(), g.all_layers.end());
    }

    std::sort(ens1.groups.begin(), ens1.groups.end(), [](const ov::npuw::Group& g1,
                                                         const ov::npuw::Group& g2){
                                                                return g1.all_layers.front() < g2.all_layers.front();
                                                        });

    std::sort(ens2.groups.begin(), ens2.groups.end(), [](const ov::npuw::Group& g1,
                                                         const ov::npuw::Group& g2){
                                                                return g1.all_layers.front() < g2.all_layers.front();
                                                        });

    for (size_t i = 0; i < ens1.groups.size(); ++i) {
        const auto& g1 = ens1.groups.at(i);
        const auto& g2 = ens2.groups.at(i);

        if (g1.avoid_list != g2.avoid_list ||
            g1.input_layers != g2.input_layers ||
            g1.output_layers != g2.output_layers ||
            g1.all_layers != g2.all_layers) {
            return false;
        }

        // Can't compare them directly since they are random, but dont't affect the structure
        if ((g1.repeated_id.empty() && !g2.repeated_id.empty()) ||
            (!g1.repeated_id.empty() && g2.repeated_id.empty())) {
            return false;
        }
    }

    if (ens1.repeated.size() != ens2.repeated.size()) {
        return false;
    }

    auto get_sorted_rep = [](const std::map<std::string, ov::npuw::RepeatedBlock>& rep) {
        std::vector<std::vector<std::set<std::string>>> sorted_rep;

        std::transform(rep.begin(), rep.end(), std::back_inserter(sorted_rep), [](const auto& v) {
            return v.second.matches;
        });

        for (auto& g : sorted_rep) {
            std::sort(g.begin(), g.end(),
                    [](const auto& a, const auto& b) {return *a.begin() < *b.begin();});
        }

        std::sort(sorted_rep.begin(), sorted_rep.end(),
                    [](const auto& a, const auto& b) {return *a.front().begin() < *b.front().begin();});

        return sorted_rep;
    };


    if (get_sorted_rep(ens1.repeated) != get_sorted_rep(ens2.repeated)) {
        return false;
    }

    return true;
}

class ModelGenerator {
public:
    ModelGenerator() = default;

    std::shared_ptr<ov::Model> get_model_without_repeated_blocks() {
        std::shared_ptr<ov::op::v0::Parameter> input = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 1, 40});
        m_nodes.push_back(input);
        set_name(input);

        std::shared_ptr<ov::Node> res = get_block(input);

        auto result = std::make_shared<ov::op::v0::Result>(res);
        m_nodes.push_back(result);
        set_name(result);

        ov::ParameterVector params = {input};
        ov::ResultVector results = {result};

        return std::make_shared<ov::Model>(results, params);
    }

    std::shared_ptr<ov::Model> get_model_with_repeated_blocks() {
        // Generate head
        std::shared_ptr<ov::op::v0::Parameter> input = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 1, 40});
        m_nodes.push_back(input);
        set_name(input);

        std::vector<std::shared_ptr<ov::Node>> head(7, nullptr);
        head[0] = std::make_shared<ov::op::v1::Add>(input, input);
        head[1] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int>{2});
        head[2] = std::make_shared<ov::op::v1::Divide>(head[0], head[1], true);
        head[3] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int>{1, 1, 4, 10});
        head[4] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int>{1, 1, 40});
        head[5] = std::make_shared<ov::op::v1::Reshape>(head[2], head[3], false);
        head[6] = std::make_shared<ov::op::v1::Reshape>(head[5], head[4], false);

        for (const auto& h : head) {
            m_nodes.push_back(h);
            set_name(h);
        }

        // Generate repeated blocks
        std::shared_ptr<ov::Node> output = get_block(head[6]);
        std::vector<std::shared_ptr<ov::Node>> outputs;
        outputs.push_back(output);

        for (size_t i = 0; i < 9; ++i) {
            output = get_block(output);
            outputs.push_back(output);
        }

        // Generate tail
        std::vector<std::shared_ptr<ov::Node>> tail(6, nullptr);
        tail[0] = std::make_shared<ov::op::v0::Concat>(outputs, -1);
        tail[1] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int>{1, 20, 20});
        tail[2] = std::make_shared<ov::op::v1::Reshape>(tail[0], tail[1], false);
        tail[3] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1, 1, 1});
        tail[4] = std::make_shared<ov::op::v1::Multiply>(tail[2], tail[3]);
        tail[5] = std::make_shared<ov::op::v1::Add>(tail[4], tail[4]);

        for (const auto& t : tail) {
            m_nodes.push_back(t);
            set_name(t);
        }

        // Create model
        auto result = std::make_shared<ov::op::v0::Result>(tail[5]);
        m_nodes.push_back(result);
        set_name(result);

        ov::ParameterVector params = {input};
        ov::ResultVector results = {result};

        return std::make_shared<ov::Model>(results, params);
    }

    std::shared_ptr<ov::Node> get_block(const std::shared_ptr<ov::Node>& input) {
        // Parameters
        // input

        // Constants
        std::vector<std::shared_ptr<ov::Node>> model_c(18, nullptr);
        model_c[0] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int>{0, 2, 1, 3});
        model_c[1] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{1});
        model_c[2] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{0});
        model_c[3] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{2});
        model_c[4] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{0});
        model_c[5] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int>{1, 1, 1, 1});
        model_c[6] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{1});
        model_c[7] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{0});
        model_c[8] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int>{1, 1, 1, 1});
        model_c[9] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int>{1, 1, 1, 2});
        model_c[10] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int>{1, 1, 1, 1});
        model_c[11] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int>{1, 1, 1, 2});
        model_c[12] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1, 1, 1, 1});
        model_c[13] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1, 1, 1, 1});
        model_c[14] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1, 1, 1, 1});
        model_c[15] = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{40, 40});
        model_c[16] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int>{1, 1, 4, 10});
        model_c[17] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int>{1, 1, 40});

        for (const auto& c : model_c) {
            m_nodes.push_back(c);
            set_name(c);
        }

        // Converts
        std::vector<std::shared_ptr<ov::Node>> convert(3, nullptr);
        convert[0] = std::make_shared<ov::op::v0::Convert>(model_c[15], ov::element::f16);
        convert[1] = std::make_shared<ov::op::v0::Convert>(convert[0], ov::element::i32);
        convert[2] = std::make_shared<ov::op::v0::Convert>(model_c[12], ov::element::i32);

        for (const auto& c : convert) {
            m_nodes.push_back(c);
            set_name(c);
        }

        // Ops
        std::vector<std::shared_ptr<ov::Node>> op(16, nullptr);
        op[0] = std::make_shared<ov::op::v0::MatMul>(input, convert[1], false, true);
        op[1] = std::make_shared<ov::op::v1::Reshape>(op[0], model_c[16], false);
        op[2] = std::make_shared<ov::op::v1::Transpose>(op[1], model_c[0]);
        op[3] = std::make_shared<ov::op::v0::ShapeOf>(op[2]);
        op[4] = std::make_shared<ov::op::v1::Gather>(op[3], model_c[1], model_c[2]);
        op[5] = std::make_shared<ov::op::v1::Divide>(op[4], model_c[3], true);
        op[6] = std::make_shared<ov::op::v0::Floor>(op[5]);
        op[7] = std::make_shared<ov::op::v3::ScatterUpdate>(model_c[5], model_c[6], op[6], model_c[7]);
        op[8] = std::make_shared<ov::op::v1::StridedSlice>(op[2],
                                                            model_c[8],
                                                            op[7],
                                                            model_c[9],
                                                            std::vector<int64_t>{1, 1, 1, 1},
                                                            std::vector<int64_t>{1, 1, 1, 1});
        op[9] = std::make_shared<ov::op::v1::StridedSlice>(op[2],
                                                            op[7],
                                                            model_c[10],
                                                            model_c[11],
                                                            std::vector<int64_t>{1, 1, 1, 1},
                                                            std::vector<int64_t>{1, 1, 1, 1});
        op[10] = std::make_shared<ov::op::v1::Multiply>(op[9], convert[2]);
        op[11] = std::make_shared<ov::op::v0::Concat>(std::vector<std::shared_ptr<ov::Node>>{op[10], op[8]}, -1);
        op[12] = std::make_shared<ov::op::v1::Multiply>(model_c[13], op[11]);
        op[13] = std::make_shared<ov::op::v1::Multiply>(model_c[14], op[2]);
        op[14] = std::make_shared<ov::op::v1::Add>(op[13], op[12]);
        op[15] = std::make_shared<ov::op::v1::Reshape>(op[14], model_c[17], false);

        for (const auto& o : op) {
            m_nodes.push_back(o);
            set_name(o);
        }

        return op[15];
    }

private:
    void set_name(const std::shared_ptr<ov::Node>& node) {
        node->set_friendly_name("node_" + std::to_string(m_name_idx++));
    }

    std::vector<std::shared_ptr<ov::Node>> m_nodes;
    size_t m_name_idx;
};

TEST(OnlinePartitioning, PartitioningIsTheSameSmallModel) {
    ModelGenerator mg;
    auto model = mg.get_model_without_repeated_blocks();

    auto cfg = ::intel_npu::Config(std::make_shared<::intel_npu::OptionsDesc>());

    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    for (size_t i = 0; i < 100; ++i) {
        auto ens_again = ov::npuw::online::buildPartitioning(model, cfg);
        EXPECT_TRUE(isEqualEns(ens, ens_again));
    }
}

TEST(OnlinePartitioning, PartitioningIsTheSameRepeatedModel) {
    ModelGenerator mg;
    auto model = mg.get_model_with_repeated_blocks();

    auto cfg = ::intel_npu::Config(std::make_shared<::intel_npu::OptionsDesc>());

    auto ens = ov::npuw::online::buildPartitioning(model, cfg);

    for (size_t i = 0; i < 100; ++i) {
        auto ens_again = ov::npuw::online::buildPartitioning(model, cfg);
        EXPECT_TRUE(isEqualEns(ens, ens_again));
    }
}

} // anonymous namespace
