// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <thread>
#include <chrono>
#include <gtest/gtest.h>
#include <legacy/layer_transform.hpp>
#include "gna_matcher.hpp"

using namespace InferenceEngine;
using namespace GNAPluginNS;
using namespace GNATestIRs;

class GNAAOTTests : public GNATest<>{
 protected:
    std::list<std::string> files_to_remove;
    std::string registerFileForRemove(std::string file_to_remove) {
        files_to_remove.push_back(file_to_remove);
        return file_to_remove;
    }

    std::string generateFileName(const std::string& baseName) const {
        using namespace std::chrono;
        std::stringstream ss;
        auto ts = duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch());
        ss << std::this_thread::get_id() << "_" << ts.count() << "_" << baseName;
        return ss.str();
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

    const std::string X = registerFileForRemove(generateFileName("unit_tests.bin"));

    // running export to a file
    export_network(AffineWith2AffineOutputsModel())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f).as().gna().model().to(X);

    // running infer using imported model instead of IR
    assert_that().onInferModel().importedFrom(X)
        .inNotCompactMode().gna().propagate_forward().called().once();
}

TEST_F(GNAAOTTests, DISABLED_AffineWith2AffineOutputs_canbe_imported_verify_structure) {
// Disabled because of random fails: Issue-23611
#if GNA_LIB_VER == 1
    GTEST_SKIP();
#endif
    auto & nnet_type = storage<gna_nnet_type_t>();

    // saving pointer to nnet - todo probably deep copy required
    save_args().onInferModel(AffineWith2AffineOutputsModel())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f).from().gna().propagate_forward().to(&nnet_type);

    const std::string X = registerFileForRemove(generateFileName("unit_tests.bin"));

    // running export to a file
    export_network(AffineWith2AffineOutputsModel())
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f).as().gna().model().to(X);

    // running infer using imported model instead of IR
    assert_that().onInferModel().importedFrom(X)
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f).gna()
        .propagate_forward().called_with().exact_nnet_structure(&nnet_type);

}

TEST_F(GNAAOTTests, DISABLED_TwoInputsModel_canbe_export_imported) {
    // Disabled because of random conflicts with other tests: Issue-54220
#if GNA_LIB_VER == 1
    GTEST_SKIP();
#endif

    const std::string X = registerFileForRemove(generateFileName("unit_tests.bin"));

    // running export to a file
    export_network(TwoInputsModelForIO())
            .inNotCompactMode()
            .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_0"), 1.0f)
            .withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_1"), 1.0f)
            .as().gna().model().to(X);

    // running infer using imported model instead of IR
    assert_that().onInferModel().importedFrom(X)
            .inNotCompactMode().gna().propagate_forward().called().once();
}

TEST_F(GNAAOTTests, DISABLED_PermuteModel_canbe_export_imported) {
    // Disabled because of random conflicts with other tests: Issue-54220
#if GNA_LIB_VER == 1
    GTEST_SKIP();
#endif

    const std::string X = registerFileForRemove(generateFileName("unit_tests.bin"));

    // running export to a file
    export_network(PermuteModelForIO())
            .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f).as().gna().model().to(X);

    // running infer using imported model instead of IR
    assert_that().onInferModel().importedFrom(X)
            .inNotCompactMode().gna().propagate_forward().called().once();
}

TEST_F(GNAAOTTests, DISABLED_PoolingModel_canbe_export_imported) {
    // Disabled because of random conflicts with other tests: Issue-54220
#if GNA_LIB_VER == 1
    GTEST_SKIP();
#endif

    const std::string X = registerFileForRemove(generateFileName("unit_tests.bin"));

    // running export to a file
    export_network(maxpoolAfterRelu())
            .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f).as().gna().model().to(X);

    // running infer using imported model instead of IR
    assert_that().onInferModel().importedFrom(X)
            .inNotCompactMode().gna().propagate_forward().called().once();
}

TEST_F(GNAAOTTests, DISABLED_CanConvertFromAOTtoSueModel) {

    auto & nnet_type = storage<gna_nnet_type_t>();

    // saving pointer to nnet - todo probably deep copy required
    save_args().onInferModel(AffineWith2AffineOutputsModel())
        .inNotCompactMode().inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .from().gna().propagate_forward().to(&nnet_type);

    const std::string X = registerFileForRemove(generateFileName("unit_tests.bin"));

    // running export to a file
    export_network(AffineWith2AffineOutputsModel())
        .inNotCompactMode().inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f)
        .as().gna().model().to(X);

    // running infer using imported model instead of IR
    assert_that().onInferModel().importedFrom(X)
        .inNotCompactMode().withGNAConfig(GNA_CONFIG_KEY(SCALE_FACTOR), 1.0f).withGNAConfig(GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE), "sue.dump")
        .gna().dumpXNN().called();
}

