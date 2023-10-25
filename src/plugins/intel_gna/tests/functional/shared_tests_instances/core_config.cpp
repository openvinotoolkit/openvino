// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/core_config.hpp"

#include <ie_ngraph_utils.hpp>
#include <string>

#include "functional_test_utils/blob_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

void CoreConfiguration(LayerTestsUtils::LayerTestsCommon* test) {
    const float MAX_VAL_2B_FEAT = 16384.0f;
    auto inputParameters = test->GetFunction()->get_parameters();
    auto& configuration = test->GetConfiguration();
    for (size_t i = 0; i < inputParameters.size(); ++i) {
        std::string scaleFactorConfigKey = "GNA_SCALE_FACTOR" + std::string("_") + std::to_string(i);
        if (configuration.find(scaleFactorConfigKey) != configuration.end()) {
            continue;
        }

        auto elementType = inputParameters[i]->get_element_type();
        auto shape = inputParameters[i]->get_shape();
        auto precision = InferenceEngine::details::convertPrecision(elementType);
        precision = (precision.getPrecVal() == InferenceEngine::Precision::FP16)
                        ? InferenceEngine::Precision(InferenceEngine::Precision::FP32)
                        : precision;

        InferenceEngine::SizeVector size(shape);
        InferenceEngine::TensorDesc tensor(precision, size, InferenceEngine::Layout::ANY);
        InferenceEngine::DataPtr dataPtr = std::make_shared<InferenceEngine::Data>("tmp", tensor);

        InferenceEngine::InputInfo info;
        info.setInputData(dataPtr);
        info.setPrecision(precision);

        auto blob = test->GenerateInput(info);
        float floatScaleFactor = 1.0f;

        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
        IE_ASSERT(memory);

        const auto lockedMemory = memory->wmap();
        if (precision == InferenceEngine::Precision::FP32) {
            float* ptrFloatFeat = lockedMemory.as<float*>();
            float max = 0.0;

            for (size_t i = 0; i < blob->size(); i++) {
                if (fabs(ptrFloatFeat[i]) > max) {
                    max = fabs(ptrFloatFeat[i]);
                }
            }

            floatScaleFactor = (max == 0) ? 1.0f : MAX_VAL_2B_FEAT / max;
        }

        configuration[scaleFactorConfigKey] = std::to_string(floatScaleFactor);
    }
}

namespace ov {
namespace test {

void core_configuration(ov::test::SubgraphBaseTest* test) {}

}  // namespace test
}  // namespace ov
