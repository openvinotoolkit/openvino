// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <memory>
#include <string>
#include <algorithm>
#include <vector>

#include "functional_test_utils/test_model/test_model.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include <ngraph_functions/subgraph_builders.hpp>
#include <ngraph/pass/manager.hpp>
#include "transformations/serialize.hpp"
#include "ie_ngraph_utils.hpp"

namespace FuncTestUtils {
namespace TestModel {

/**
 * @brief generates IR files (XML and BIN files) with the test model.
 *        Passed reference vector is filled with CNN layers to validate after the network reading.
 * @param modelPath used to serialize the generated network
 * @param weightsPath used to serialize the generated weights
 * @param netPrc precision of the generated network
 * @param inputDims dims on the input layer of the generated network
 */
void generateTestModel(const std::string &modelPath,
                       const std::string &weightsPath,
                       const InferenceEngine::Precision &netPrc,
                       const InferenceEngine::SizeVector &inputDims) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::Serialize>(
            modelPath, weightsPath,
            ngraph::pass::Serialize::Version::IR_V10);
    manager.run_passes(ngraph::builder::subgraph::makeConvPoolRelu(
            inputDims, InferenceEngine::details::convertPrecision(netPrc)));
}

}  // namespace TestModel
}  // namespace FuncTestUtils
