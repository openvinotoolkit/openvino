// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/function.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

namespace FuncTestUtils {

using ComparingNodesPair = typename std::pair<std::shared_ptr<ngraph::Node>, std::shared_ptr<ngraph::Node>>;
using ComparingNodesBFSQueue = typename std::queue<ComparingNodesPair>;

//
// This function compares two nGraph functions and requires them to have exactly one output
// Check nodes types
// Check number of inputs
// Check shapes of each Node
//
void CompareFunctions(const std::shared_ptr<ngraph::Function>& fActual,
                      const std::shared_ptr<ngraph::Function>& fExpected) {
    const auto fActualResults = fActual->get_results();
    const auto fExpectedResults = fExpected->get_results();

    ASSERT_EQ(fActualResults.size(), 1);
    ASSERT_EQ(fExpectedResults.size(), 1);

    const auto typeInfoToStr = [](const ngraph::Node::type_info_t& typeInfo) {
        return std::string(typeInfo.name) + "/" + std::to_string(typeInfo.version);
    };

    ComparingNodesBFSQueue comparingNodes;
    comparingNodes.push({fActualResults[0], fExpectedResults[0]});
    while (!comparingNodes.empty()) {
        const auto node1 = comparingNodes.front().first;
        const auto node2 = comparingNodes.front().second;
        comparingNodes.pop();

        ASSERT_EQ(node1->get_type_info(), node2->get_type_info())
                                    << "Functions compare: data types must be equal "
                                    << typeInfoToStr(node1->get_type_info()) << " != "
                                    << typeInfoToStr(node2->get_type_info());

        ASSERT_EQ(node1->inputs().size(), node2->inputs().size())
                                    << "Functions compare: numbers of inputs are different: "
                                    << node1->inputs().size() << " and " << node2->inputs().size();

        for (int i = 0; i < node1->inputs().size(); ++i) {
            const auto partialShape1 = node1->input(i).get_partial_shape();
            const auto partialShape2 = node2->input(i).get_partial_shape();
            ASSERT_TRUE(partialShape1.relaxes(partialShape2) && partialShape1.refines(partialShape2))
                                        << "Functions compare: Different shape detected "
                                        << partialShape1 << " and " << partialShape2;

            comparingNodes.push({node1->input_value(i).get_node_shared_ptr(),
                                 node2->input_value(i).get_node_shared_ptr()});
        }
    }
}

}  // namespace FuncTestUtils
