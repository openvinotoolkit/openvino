// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "subgraphs_builders.hpp"
#include "openvino/op/relu.hpp"

namespace {

class SerializeBaseTest : virtual public ov::test::SubgraphBaseStaticTest {
public:
    std::string cacheDirName = "cache_serialize";

    void run() override {
        std::stringstream model_stream;
        ov::AnyMap config = { ov::cache_dir(cacheDirName) };
        compiledModel = core->compile_model(function, targetDevice);
        compiledModel.export_model(model_stream);
        auto imported_model = core->import_model(model_stream, targetDevice);
    }

    void TearDown() override{
        ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
        ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
        ov::test::utils::removeDir(cacheDirName);
        ov::test::SubgraphBaseStaticTest::TearDown();
    }
};


class LSTMSequenceTest : virtual public SerializeBaseTest {
public:
    void SetUp() override {
        auto init_shape = ov::PartialShape({1, 30, 512});
        auto batch_size = static_cast<size_t>(init_shape[0].get_length());
        size_t hidden_size = 10;
        size_t sequence_axis = 1;
        targetDevice = "GPU";
        auto input_size = static_cast<size_t>(init_shape[init_shape.size()-1].get_length());
        function = tests::makeLSTMSequence(ov::element::f16, init_shape, batch_size, input_size, hidden_size, sequence_axis,
            ov::op::RecurrentSequenceDirection::FORWARD);
    }
};


class GRUSequenceTest : virtual public SerializeBaseTest {
public:
    void SetUp() {
        std::string cacheDirName = "cache_gru";
        auto init_shape = ov::PartialShape({1, 30, 512});
        auto batch_size = static_cast<size_t>(init_shape[0].get_length());
        size_t hidden_size = 10;
        size_t sequence_axis = 1;
        targetDevice = "GPU";
        auto input_size = static_cast<size_t>(init_shape[init_shape.size()-1].get_length());
        function = tests::makeLBRGRUSequence(ov::element::f16, init_shape, batch_size, input_size, hidden_size, sequence_axis,
            ov::op::RecurrentSequenceDirection::FORWARD);
    }
};

TEST_F(LSTMSequenceTest, smoke_serialize) {
    run();
}

TEST_F(GRUSequenceTest, smoke_serialize) {
    run();
}

class GpuCacheDirWithDotsParamTest : public ::testing::TestWithParam<std::string> {
protected:
    ov::Core core;
    std::string cacheDir;

    void SetUp() override {
        std::stringstream ss;
        ss << std::hex << std::hash<std::string>{}(std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));

        // Base (no trailing slash first)
        cacheDir = ss.str() + GetParam();

        // Clean previous
        ov::test::utils::removeFilesWithExt(cacheDir, "blob");
        ov::test::utils::removeFilesWithExt(cacheDir, "cl_cache");
        ov::test::utils::removeDir(cacheDir);

        core.set_property(ov::cache_dir(cacheDir));
    }

    void TearDown() override {
        ov::test::utils::removeFilesWithExt(cacheDir, "blob");
        ov::test::utils::removeFilesWithExt(cacheDir, "cl_cache");
        ov::test::utils::removeDir(cacheDir);
    }
};

TEST_P(GpuCacheDirWithDotsParamTest, smoke_PopulateAndReuseCache) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 8, 8});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "CacheDotsModel");
    core.compile_model(model, "GPU");
}

INSTANTIATE_TEST_SUITE_P(CacheDirDotVariants,
                         GpuCacheDirWithDotsParamTest,
                         ::testing::Values("/test_encoder/test_encoder.encrypted/", "/test_encoder/test_encoder.encrypted"));

}  // namespace
