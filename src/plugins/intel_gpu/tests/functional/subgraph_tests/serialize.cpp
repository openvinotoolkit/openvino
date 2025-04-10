// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "subgraphs_builders.hpp"

namespace {
class LSTMSequenceTest : virtual public ov::test::SubgraphBaseStaticTest {
public:
    void run() {
        std::string cacheDirName = "cache_lstm";
        auto init_shape = ov::PartialShape({1, 30, 512});
        auto batch_size = static_cast<size_t>(init_shape[0].get_length());
        size_t hidden_size = 10;
        size_t sequence_axis = 1;
        targetDevice = "GPU";
        auto input_size = static_cast<size_t>(init_shape[init_shape.size()-1].get_length());
        function = tests::makeLSTMSequence(ov::element::f16, init_shape, batch_size, input_size, hidden_size, sequence_axis,
            ov::op::RecurrentSequenceDirection::FORWARD);
        std::stringstream model_stream;
        ov::AnyMap config = { ov::cache_dir(cacheDirName) };
        compiledModel = core->compile_model(function, targetDevice);
        compiledModel.export_model(model_stream);
        auto imported_model = core->import_model(model_stream, targetDevice);
        ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
        ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
        ov::test::utils::removeDir(cacheDirName);
    }
};

TEST_F(LSTMSequenceTest, smoke_serialize) {
    run();
}
}  // namespace
