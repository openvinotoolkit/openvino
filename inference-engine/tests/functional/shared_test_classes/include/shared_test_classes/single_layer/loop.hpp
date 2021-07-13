// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <ngraph/op/util/attr_types.hpp>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {
enum LOOP_IN_TYPE {
    INVARIANT,
    MERGED
};

using LoopParams = typename std::tuple<
        bool,                                                              // ExecuteFirstIteration
        bool,                                                              // BodyCondition is a constant?
        bool,                                                              // BodyCondition value, if it is a Const
        int64_t,                                                           // TripCount, -1 means infinity
        std::vector<std::pair<std::vector<size_t>, LOOP_IN_TYPE>>,         // inputs
        InferenceEngine::Precision,                                        // Network precision
        std::string>;                                                      // Device name

class LoopTest : public testing::WithParamInterface<LoopParams>,
                 virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LoopParams> &obj);

protected:
    void SetUp() override;
};


using StaticShapeLoopParams = typename std::tuple<
        bool,
        bool,
        std::tuple<
            bool,
            int64_t,
            int64_t,
            int64_t
            >,
        int64_t,
        InferenceEngine::SizeVector,
        InferenceEngine::Precision,
        std::string
        >;

/**
 * Test case with static SHAPE version of loop operation.
 * Total iteration count is dynamic.
 */
class StaticShapeLoopTest : public testing::WithParamInterface<StaticShapeLoopParams>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<StaticShapeLoopParams> &obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;
    std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> PredefinedRefs();

private:
    bool unrolling;             // unroll Loop
    bool static_iter_num;       // trip count provided by constant node
    bool static_continue_cond;  // initial_cond provided by constant node
    int64_t max_iter_num;       // -1 means infinity loop (expected dynamic exit condition in body)
    int64_t dynamic_exit;       // -1 means always true
    int64_t axis;               // -1 means no auto concatenation
    int64_t start_value;
    InferenceEngine::SizeVector data_shape;
    InferenceEngine::Precision data_prc;

    int64_t actual_n_iter();

protected:
    void SetUp() override;
};


class TrivialLoopTest : public testing::WithParamInterface<LayerTestsUtils::basicParams>,
                        virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    using RefBlobGenerator = std::function<InferenceEngine::Blob::Ptr (const InferenceEngine::TensorDesc &info)>;
    std::map<std::string, RefBlobGenerator> inputGens, outputGens;

    void CreateSlicedLoop(size_t batch_size, size_t num_iteration, InferenceEngine::Precision iePrc,
                          InferenceEngine::SizeVector& ieShape);
    void CreateSlicedLoopDynCondition(size_t batch_size, size_t num_iteration, InferenceEngine::Precision iePrc,
                          InferenceEngine::SizeVector& ieShape, size_t trip_count);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
        auto found = inputGens.find(info.name());
        if (found != inputGens.end()) {
            return found->second(info.getTensorDesc());
        }

        found = inputGens.find("");
        if (found != inputGens.end()) {
            return found->second(info.getTensorDesc());
        }

        return LayerTestsCommon::GenerateInput(info);
    }

    std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> CalculateRefs() override {
        if (outputGens.empty())
            return LayerTestsCommon::CalculateRefs();

        const auto results = function->get_results();
        const auto outs_info = cnnNetwork.getOutputsInfo();
        const auto num_out_blob = results.size();

        std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> res_collection(num_out_blob);

        for (int i = 0; i < num_out_blob; i++) {
            // TODO: name of original NG result doesn't match with outs after conversion.
            //       Expected : auto name = results[i]->get_friendly_name();
            auto name = results[i]->get_input_node_ptr(0)->get_friendly_name();
            auto data = outs_info.at(name);
            IE_ASSERT(data != nullptr);

            RefBlobGenerator generator;
            auto found = outputGens.find(name);
            if (found != outputGens.end()) {
                generator = found->second;
            } else {
                found = outputGens.find("");
                if (found != outputGens.end()) {
                    generator = found->second;
                }
            }

            IE_ASSERT(generator != nullptr) << "Test output generator is not specified";
            auto blob = generator(data->getTensorDesc());
            auto blob_size = blob->byteSize();
            auto blob_ptr = blob->buffer().as<uint8_t*>();

            auto &res = res_collection[i];
            res.second.resize(blob_size);
            std::copy(blob_ptr, blob_ptr + blob_size, res.second.begin());
        }
        return res_collection;
    }
};

}  // namespace LayerTestsDefinitions
