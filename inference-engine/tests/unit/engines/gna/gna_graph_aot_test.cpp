/*
 * INTEL CONFIDENTIAL
 * Copyright (C) 2018-2019 Intel Corporation.
 *
 * The source code contained or described herein and all documents
 * related to the source code ("Material") are owned by Intel Corporation
 * or its suppliers or licensors. Title to the Material remains with
 * Intel Corporation or its suppliers and licensors. The Material may
 * contain trade secrets and proprietary and confidential information
 * of Intel Corporation and its suppliers and licensors, and is protected
 * by worldwide copyright and trade secret laws and treaty provisions.
 * No part of the Material may be used, copied, reproduced, modified,
 * published, uploaded, posted, transmitted, distributed, or disclosed
 * in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other
 * intellectual property right is granted to or conferred upon you by
 * disclosure or delivery of the Materials, either expressly, by implication,
 * inducement, estoppel or otherwise. Any license under such intellectual
 * property rights must be express and approved by Intel in writing.
 *
 * Include any supplier copyright notices as supplier requires Intel to use.
 *
 * Include supplier trademarks or logos as supplier requires Intel to use,
 * preceded by an asterisk. An asterisked footnote can be added as follows:
 * *Third Party trademarks are the property of their respective owners.
 *
 * Unless otherwise agreed by Intel in writing, you may not remove or alter
 * this notice or any other notice embedded in Materials by Intel or Intel's
 * suppliers or licensors in any way.
 */


#include <vector>
#include <gtest/gtest.h>
#include <inference_engine/layer_transform.hpp>
#include <gna_plugin/quantization/model_quantizer.hpp>
#include "gna_plugin/quantization/layer_quantizer.hpp"
#include "gna_matcher.hpp"

using namespace InferenceEngine;
using namespace GNAPluginNS;
using namespace GNATestIRs;

class GNAAOTTests : public GNATest {
 protected:
    std::list<std::string> files_to_remove;
    std::string registerFileForRemove(std::string file_to_remove) {
        files_to_remove.push_back(file_to_remove);
        return file_to_remove;
    }
    void TearDown() override {
        for (auto & file : files_to_remove) {
            std::remove(file.c_str());
        }
    }

    void SetUp() override  {
    }
};

TEST_F(GNAAOTTests, AffineWith2AffineOutputs_canbe_export_imported) {

    const std::string X = registerFileForRemove("unit_tests.bin");

    // running export to a file
    export_network(AffineWith2AffineOutputsModel())
        .inNotCompactMode().as().gna().model().to(X);

    // running infer using imported model instead of IR
    assert_that().onInferModel().importedFrom(X)
        .inNotCompactMode().gna().propagate_forward().called().once();
}


TEST_F(GNAAOTTests, AffineWith2AffineOutputs_canbe_imported_verify_structure) {

    auto & nnet_type = storage<intel_nnet_type_t>();

    // saving pointer to nnet - todo probably deep copy required
    save_args().onInferModel(AffineWith2AffineOutputsModel())
        .inNotCompactMode().from().gna().propagate_forward().to(&nnet_type);

    const std::string X = registerFileForRemove("unit_tests.bin");

    // running export to a file
    export_network(AffineWith2AffineOutputsModel())
        .inNotCompactMode().as().gna().model().to(X);

    // running infer using imported model instead of IR
    assert_that().onInferModel().importedFrom(X)
        .inNotCompactMode().gna().propagate_forward().called_with().exact_nnet_structure(&nnet_type);

}

TEST_F(GNAAOTTests, CanConvertFromAOTtoSueModel) {

    auto & nnet_type = storage<intel_nnet_type_t>();

    // saving pointer to nnet - todo probably deep copy required
    save_args().onInferModel(AffineWith2AffineOutputsModel())
        .inNotCompactMode().from().gna().propagate_forward().to(&nnet_type);

    const std::string X = registerFileForRemove("unit_tests.bin");

    // running export to a file
    export_network(AffineWith2AffineOutputsModel())
        .inNotCompactMode().as().gna().model().to(X);

    // running infer using imported model instead of IR
    assert_that().onInferModel().importedFrom(X)
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE), "sue.dump").gna().dumpXNN().called();
}

