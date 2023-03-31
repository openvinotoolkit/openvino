// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_gna/properties.hpp"
#include "test_models.hpp"


#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace ov;
using std::vector;

class GnaFp32Tests : virtual public ov::test::SubgraphBaseTest {
public:
    void SetUp(std::shared_ptr<ov::Model> model) {
        targetDevice = "GNA";
        configuration = {intel_gna::execution_mode(intel_gna::ExecutionMode::SW_FP32)};
        function = model;
        vector<Shape> input_shapes;
        for (auto& input : function->inputs()) {
            input_shapes.push_back(input.get_shape());
        }
        targetStaticShapes.push_back(input_shapes);
        abs_threshold = std::numeric_limits<double>::min();
        rel_threshold = std::numeric_limits<double>::min();
    };
};

TEST_F(GnaFp32Tests, test_eltwise_add_model) {
    SetUp(eltwise_add_model());
    run();
}

TEST_F(GnaFp32Tests, test_lstm_cell_model) {
    SetUp(lstm_cell_only_model());
    run();
}

// TODO 
TEST_F(GnaFp32Tests, DISABLED_test_lstm_cell_unaligned_model) {
    SetUp(lstm_cell_only_model_unaligned());
    run();
}

TEST_F(GnaFp32Tests, test_fc_with_padding_after_split_model) {
    SetUp(fc_with_padding_after_split_model());
    run();
}

TEST_F(GnaFp32Tests, test_scaleshift_3d_model) {
    SetUp(scaleshift_3d_model());
    run();
}

TEST_F(GnaFp32Tests, test_input_split_concat_model) {
    SetUp(input_split_concat_model());
    run();
}

TEST_F(GnaFp32Tests, test_input_split_concat_unaligned_model) {
    SetUp(input_split_concat_unaligned_model());
    run();
}

TEST_F(GnaFp32Tests, test_power_with_scale_factor_model) {
    SetUp(power_with_scale_factor_model());
    run();
}

TEST_F(GnaFp32Tests, test_reshape_convolution_less_than_48_filters) {
    SetUp(reshape_convolution_less_than_48_filters());
    run();
}

TEST_F(GnaFp32Tests, multiple_inputs_model) {
    SetUp(two_inputs_to_affine_model());
    run();
}

// TODO subgraph test fails to properly initialize input of these models
TEST_F(GnaFp32Tests, DISABLED_test_slice_model_with_aligned_outputs) {
    SetUp(slice_model_with_aligned_outputs());
    run();
}

TEST_F(GnaFp32Tests, DISABLED_test_two_fc_with_padding_after_slice_model) {
    SetUp(two_fc_with_padding_after_slice_model());
    run();
}

TEST_F(GnaFp32Tests, DISABLED_test_two_outputs_model) {
    SetUp(two_outputs_model());
    run();
}

TEST_F(GnaFp32Tests, DISABLED_test_two_outputs_relu_model) {
    SetUp(two_outputs_relu_model());
    run();
}
