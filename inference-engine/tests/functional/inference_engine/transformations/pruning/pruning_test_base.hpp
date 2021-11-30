// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/coordinate_transform.hpp>

#include <common_test_utils/ngraph_test_utils.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

using namespace testing;
using namespace ngraph;

class PruningTestsCommon: public CommonTestUtils::TestsCommon {
public:
    void Run() {
        try {
            std::vector<std::vector<uint8_t>> input_data;
            ngraph::element::TypeVector types;
            for (auto param : function_ref->get_parameters()) {
                types.push_back(param->get_element_type());

                InferenceEngine::TensorDesc td(InferenceEngine::Precision::FP32, param->get_shape(), InferenceEngine::Layout::ANY);
                const auto &input = FuncTestUtils::createAndFillBlob(td);
                const auto &input_size = input->byteSize();

                std::vector<uint8_t> data;
                data.resize(input_size);

                auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(input);
                IE_ASSERT(memory);

                const auto lockedMemory = memory->wmap();
                const auto buffer = lockedMemory.as<const std::uint8_t *>();
                std::copy(buffer, buffer + input_size, data.data());

                input_data.push_back(std::move(data));
            }

            auto ref_outputs = ngraph::helpers::interpreterFunction(function_ref, input_data, types);
            auto outputs = ngraph::helpers::interpreterFunction(function, input_data, types);

            IE_ASSERT(ref_outputs.size() == outputs.size());

            for (size_t i = 0; i < ref_outputs.size(); ++i) {
                IE_ASSERT(ref_outputs[i].second.size() == outputs[i].second.size());
                auto * ref = reinterpret_cast<float *>(ref_outputs[i].second.data());
                auto * out = reinterpret_cast<float *>(outputs[i].second.data());
                IE_ASSERT(ref_outputs[i].second.size() / 8);
                size_t size = ref_outputs[i].second.size() / sizeof(float);
                LayerTestsUtils::LayerTestsCommon::Compare<float, float>(ref, out, size, 1e-5);
            }
        }
        catch (const std::runtime_error &re) {
            GTEST_FATAL_FAILURE_(re.what());
        } catch (const std::exception &ex) {
            GTEST_FATAL_FAILURE_(ex.what());
        } catch (...) {
            GTEST_FATAL_FAILURE_("Unknown failure occurred.");
        }
    }

protected:
    std::shared_ptr<ov::Function> function, function_ref;
};