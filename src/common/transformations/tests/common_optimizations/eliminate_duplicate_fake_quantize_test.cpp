// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/eliminate_duplicate_fake_quantize.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/opsets/opset5_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, EliminateDuplicateFakeQuantizeIdenticalRanges) {
    // Test case: Two cascaded FakeQuantize with identical ranges and same levels
    // Should be merged into single FakeQuantize
    Shape data_shape{1, 3, 224, 224};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        
        // FQ1: input_low=0, input_high=255, output_low=0, output_high=255
        auto fq1_il = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq1_ih = opset5::Constant::create(element::f32, Shape{1}, {255});
        auto fq1_ol = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq1_oh = opset5::Constant::create(element::f32, Shape{1}, {255});
        auto fq1 = std::make_shared<opset5::FakeQuantize>(data, fq1_il, fq1_ih, fq1_ol, fq1_oh, 256);
        
        // FQ2: input_low=0, input_high=255, output_low=0, output_high=127
        auto fq2_il = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq2_ih = opset5::Constant::create(element::f32, Shape{1}, {255});
        auto fq2_ol = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq2_oh = opset5::Constant::create(element::f32, Shape{1}, {127});
        auto fq2 = std::make_shared<opset5::FakeQuantize>(fq1, fq2_il, fq2_ih, fq2_ol, fq2_oh, 256);
        
        model = std::make_shared<Model>(OutputVector{fq2}, ParameterVector{data});
        manager.register_pass<ov::pass::EliminateDuplicateFakeQuantize>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        
        // Merged FQ: Uses FQ1's input range and FQ2's output range
        auto il = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto ih = opset5::Constant::create(element::f32, Shape{1}, {255});
        auto ol = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto oh = opset5::Constant::create(element::f32, Shape{1}, {127});
        auto fq = std::make_shared<opset5::FakeQuantize>(data, il, ih, ol, oh, 256);
        
        model_ref = std::make_shared<Model>(OutputVector{fq}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, EliminateDuplicateFakeQuantizeSubsetRange) {
    // Test case: FQ1 output range is within FQ2 input range (subset)
    // Should be merged
    Shape data_shape{1, 3, 224, 224};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        
        // FQ1: outputs [0, 100]
        auto fq1_il = opset5::Constant::create(element::f32, Shape{1}, {-10});
        auto fq1_ih = opset5::Constant::create(element::f32, Shape{1}, {500});
        auto fq1_ol = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq1_oh = opset5::Constant::create(element::f32, Shape{1}, {100});
        auto fq1 = std::make_shared<opset5::FakeQuantize>(data, fq1_il, fq1_ih, fq1_ol, fq1_oh, 256);
        
        // FQ2: expects input [0, 200] (FQ1's output is subset)
        auto fq2_il = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq2_ih = opset5::Constant::create(element::f32, Shape{1}, {200});
        auto fq2_ol = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq2_oh = opset5::Constant::create(element::f32, Shape{1}, {50});
        auto fq2 = std::make_shared<opset5::FakeQuantize>(fq1, fq2_il, fq2_ih, fq2_ol, fq2_oh, 256);
        
        model = std::make_shared<Model>(OutputVector{fq2}, ParameterVector{data});
        manager.register_pass<ov::pass::EliminateDuplicateFakeQuantize>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        
        // Merged FQ
        auto il = opset5::Constant::create(element::f32, Shape{1}, {-10});
        auto ih = opset5::Constant::create(element::f32, Shape{1}, {500});
        auto ol = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto oh = opset5::Constant::create(element::f32, Shape{1}, {50});
        auto fq = std::make_shared<opset5::FakeQuantize>(data, il, ih, ol, oh, 256);
        
        model_ref = std::make_shared<Model>(OutputVector{fq}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, EliminateDuplicateFakeQuantizeDifferentLevels) {
    // Test case: Different levels - should NOT be merged
    Shape data_shape{1, 3, 224, 224};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        
        auto fq1_il = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq1_ih = opset5::Constant::create(element::f32, Shape{1}, {255});
        auto fq1_ol = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq1_oh = opset5::Constant::create(element::f32, Shape{1}, {255});
        auto fq1 = std::make_shared<opset5::FakeQuantize>(data, fq1_il, fq1_ih, fq1_ol, fq1_oh, 256);
        
        auto fq2_il = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq2_ih = opset5::Constant::create(element::f32, Shape{1}, {255});
        auto fq2_ol = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq2_oh = opset5::Constant::create(element::f32, Shape{1}, {127});
        auto fq2 = std::make_shared<opset5::FakeQuantize>(fq1, fq2_il, fq2_ih, fq2_ol, fq2_oh, 16);  // Different levels!
        
        model = std::make_shared<Model>(OutputVector{fq2}, ParameterVector{data});
        manager.register_pass<ov::pass::EliminateDuplicateFakeQuantize>();
    }
    // Reference model should be identical (no transformation)
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        
        auto fq1_il = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq1_ih = opset5::Constant::create(element::f32, Shape{1}, {255});
        auto fq1_ol = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq1_oh = opset5::Constant::create(element::f32, Shape{1}, {255});
        auto fq1 = std::make_shared<opset5::FakeQuantize>(data, fq1_il, fq1_ih, fq1_ol, fq1_oh, 256);
        
        auto fq2_il = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq2_ih = opset5::Constant::create(element::f32, Shape{1}, {255});
        auto fq2_ol = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq2_oh = opset5::Constant::create(element::f32, Shape{1}, {127});
        auto fq2 = std::make_shared<opset5::FakeQuantize>(fq1, fq2_il, fq2_ih, fq2_ol, fq2_oh, 16);
        
        model_ref = std::make_shared<Model>(OutputVector{fq2}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, EliminateDuplicateFakeQuantizeLargeRangeMismatch) {
    // Test case: Large range mismatch (>5%) - should NOT be merged
    Shape data_shape{1, 3, 224, 224};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        
        // FQ1 outputs [0, 100]
        auto fq1_il = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq1_ih = opset5::Constant::create(element::f32, Shape{1}, {100});
        auto fq1_ol = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq1_oh = opset5::Constant::create(element::f32, Shape{1}, {100});
        auto fq1 = std::make_shared<opset5::FakeQuantize>(data, fq1_il, fq1_ih, fq1_ol, fq1_oh, 256);
        
        // FQ2 expects input [0, 50] - will clip significantly
        auto fq2_il = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq2_ih = opset5::Constant::create(element::f32, Shape{1}, {50});  // Large mismatch!
        auto fq2_ol = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq2_oh = opset5::Constant::create(element::f32, Shape{1}, {50});
        auto fq2 = std::make_shared<opset5::FakeQuantize>(fq1, fq2_il, fq2_ih, fq2_ol, fq2_oh, 256);
        
        model = std::make_shared<Model>(OutputVector{fq2}, ParameterVector{data});
        manager.register_pass<ov::pass::EliminateDuplicateFakeQuantize>();
    }
    // Should not be transformed
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        
        auto fq1_il = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq1_ih = opset5::Constant::create(element::f32, Shape{1}, {100});
        auto fq1_ol = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq1_oh = opset5::Constant::create(element::f32, Shape{1}, {100});
        auto fq1 = std::make_shared<opset5::FakeQuantize>(data, fq1_il, fq1_ih, fq1_ol, fq1_oh, 256);
        
        auto fq2_il = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq2_ih = opset5::Constant::create(element::f32, Shape{1}, {50});
        auto fq2_ol = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto fq2_oh = opset5::Constant::create(element::f32, Shape{1}, {50});
        auto fq2 = std::make_shared<opset5::FakeQuantize>(fq1, fq2_il, fq2_ih, fq2_ol, fq2_oh, 256);
        
        model_ref = std::make_shared<Model>(OutputVector{fq2}, ParameterVector{data});
    }
}
