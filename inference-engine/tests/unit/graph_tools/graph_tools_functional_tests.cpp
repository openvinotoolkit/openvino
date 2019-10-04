// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <details/ie_cnn_network_tools.h>
#include <cpp/ie_cnn_net_reader.h>
#include <cpp/ie_cnn_network.h>
#include <memory>
#include <test_model_path.hpp>

using namespace testing;
using namespace InferenceEngine::details;
using namespace InferenceEngine;
using namespace std;

class GraphToolsFncTest : public ::testing::Test {
public:
    template <typename T>
    static void checkSort(const T &sorted) {
        for (int i = 0; i < sorted.size(); i++) {
            //check that all input already visited:
            for (auto &inputs : sorted[i]->insData) {
                auto inputName = inputs.lock()->getCreatorLayer().lock()->name;

                bool bFound = false;
                for (int j = 0; j < i; j++) {
                    if (sorted[j]->name == inputName) {
                        bFound = true;
                        break;
                    }
                }
                ASSERT_TRUE(bFound) << "order is not correct, layer " << sorted[i]->name << " has missed input: "
                                    << inputName;
            }
        }
    }
};
