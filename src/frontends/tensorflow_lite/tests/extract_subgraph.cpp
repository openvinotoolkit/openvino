// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "convert_model.hpp"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::frontend;

// extract_subgraph() must reset the input index of the previously-registered input places (so a
// tensor that is no longer an input stops reporting is_input()). A copy-paste defect reset the
// outputs twice and never reset the old inputs, leaving a dropped input still reported as an input.
TEST(TensorFlowLiteExtractSubgraph, resets_old_input_places) {
    FrontEndManager fem;
    auto fe = fem.load_by_framework(TF_LITE_FE);
    ASSERT_NE(fe, nullptr);
    auto model_filename =
        FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_LITE_MODELS_DIRNAME) + "2in_2out/2in_2out.tflite");
    InputModel::Ptr input_model;
    OV_ASSERT_NO_THROW(input_model = fe->load(model_filename));
    ASSERT_NE(input_model, nullptr);

    auto dropped = input_model->get_place_by_tensor_name("inputX1");
    auto kept = input_model->get_place_by_tensor_name("inputX2");
    ASSERT_NE(dropped, nullptr);
    ASSERT_NE(kept, nullptr);
    EXPECT_TRUE(dropped->is_input());
    EXPECT_TRUE(kept->is_input());

    // Re-root the model on inputX2 alone; inputX1 is dropped from the inputs.
    OV_ASSERT_NO_THROW(input_model->extract_subgraph({kept}, input_model->get_outputs()));

    EXPECT_FALSE(dropped->is_input()) << "extract_subgraph did not reset the dropped input place";
    const auto inputs = input_model->get_inputs();
    for (const auto& in : inputs)
        EXPECT_NE(in, dropped) << "dropped input is still reported among the model inputs";
}
