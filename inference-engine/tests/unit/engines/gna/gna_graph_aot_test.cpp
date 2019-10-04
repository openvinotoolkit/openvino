// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

TEST_F(GNAAOTTests, DISABLED_AffineWith2AffineOutputs_canbe_export_imported) {

    const std::string X = registerFileForRemove("unit_tests.bin");

    // running export to a file
    export_network(AffineWith2AffineOutputsModel())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f).as().gna().model().to(X);

    // running infer using imported model instead of IR
    assert_that().onInferModel().importedFrom(X)
        .inNotCompactMode().gna().propagate_forward().called().once();
}


TEST_F(GNAAOTTests, DISABLED_AffineWith2AffineOutputs_canbe_imported_verify_structure) {

    auto & nnet_type = storage<intel_nnet_type_t>();

    // saving pointer to nnet - todo probably deep copy required
    save_args().onInferModel(AffineWith2AffineOutputsModel())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f).from().gna().propagate_forward().to(&nnet_type);

    const std::string X = registerFileForRemove("unit_tests.bin");

    // running export to a file
    export_network(AffineWith2AffineOutputsModel())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f).as().gna().model().to(X);

    // running infer using imported model instead of IR
    assert_that().onInferModel().importedFrom(X)
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f).gna()
        .propagate_forward().called_with().exact_nnet_structure(&nnet_type);

}

TEST_F(GNAAOTTests, CanConvertFromAOTtoSueModel) {

    auto & nnet_type = storage<intel_nnet_type_t>();

    // saving pointer to nnet - todo probably deep copy required
    save_args().onInferModel(AffineWith2AffineOutputsModel())
        .inNotCompactMode().inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .from().gna().propagate_forward().to(&nnet_type);

    const std::string X = registerFileForRemove("unit_tests.bin");

    // running export to a file
    export_network(AffineWith2AffineOutputsModel())
        .inNotCompactMode().inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .as().gna().model().to(X);

    // running infer using imported model instead of IR
    assert_that().onInferModel().importedFrom(X)
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f).withGNAConfig(GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE), "sue.dump")
        .gna().dumpXNN().called();
}

