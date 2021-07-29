// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

static void SpaceToBatchTest(const std::vector<float>& inputs,
                            const Shape inputs_shape,
                            const std::vector<int64_t>& block_shapes,
                            const Shape blocks_shape,
                            const std::vector<int64_t>& pads_begins,
                            const std::vector<int64_t>& pads_ends,
                            const Shape pads_shape,
                            const std::vector<float>& outputs,
                            const Shape outputs_shape)
{
    auto inputs_param = make_shared<op::Parameter>(element::f32, inputs_shape);
    auto block_shapes_param = make_shared<op::Constant>(element::i64, blocks_shape, block_shapes);
    auto pads_begins_param = make_shared<op::Constant>(element::i64, pads_shape, pads_begins);
    auto pads_ends_param = make_shared<op::Constant>(element::i64, pads_shape, pads_ends);

    auto space_to_batch = make_shared<op::v1::SpaceToBatch>(
        inputs_param, block_shapes_param, pads_begins_param, pads_ends_param);
    auto f = make_shared<Function>(space_to_batch, ParameterVector{inputs_param});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(inputs);
    test_case.add_expected_output<float>(outputs_shape, outputs);
    test_case.run();
}


NGRAPH_TEST(${BACKEND_NAME}, space_to_batch_4D)
{   
    const Shape inputs_shape{1, 1, 2, 2};
    const std::vector<float> inputs{1.0f, 1.0f, 
                                    1.0f, 1.0f};

    const Shape blocks_shape{4};
    const std::vector<int64_t> block_shapes{1, 1, 1, 1};

    const Shape pads_shape{4};
    const std::vector<int64_t> pads_begins{0, 0 ,0, 0};
    const std::vector<int64_t> pads_ends{0, 0, 0, 0};

    const Shape outputs_shape{1, 1, 2, 2};
    const std::vector<float> outputs{1.0f, 1.0f,
                                     1.0f, 1.0f};


    SpaceToBatchTest(inputs, inputs_shape, block_shapes, blocks_shape, pads_begins, 
                     pads_ends, pads_shape, outputs, outputs_shape);    
}

NGRAPH_TEST(${BACKEND_NAME}, space_to_batch_5D)
{   
    const Shape inputs_shape{1, 1, 3, 2, 1};
    const std::vector<float> inputs{1.0f, 1.0f, 1.0f,
                                    1.0f, 1.0f, 1.0f};

    const Shape blocks_shape{5};
    const std::vector<int64_t> block_shapes{1, 1, 3, 2, 2};

    const Shape pads_shape{5};
    const std::vector<int64_t> pads_begins{0, 0 ,1, 0, 3};
    const std::vector<int64_t> pads_ends{0, 0, 2, 0, 0};

    const Shape outputs_shape{12, 1, 2, 1, 2};
    const std::vector<float> outputs{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};


    SpaceToBatchTest(inputs, inputs_shape, block_shapes, blocks_shape, pads_begins, 
                     pads_ends, pads_shape, outputs, outputs_shape);    
}

NGRAPH_TEST(${BACKEND_NAME}, space_to_batch_4x4)
{   
    const Shape inputs_shape{1, 1, 4, 4};
    const std::vector<float> inputs{1.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 1.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 1.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 1.0f};

    const Shape blocks_shape{4};
    const std::vector<int64_t> block_shapes{1, 1, 1, 1};

    const Shape pads_shape{4};
    const std::vector<int64_t> pads_begins{0, 0, 1, 0};
    const std::vector<int64_t> pads_ends{0, 0, 0, 0};

    const Shape outputs_shape{1, 1, 5, 4};
    const std::vector<float> outputs{0.0f, 0.0f, 0.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 0.0f,
                                     0.0f, 1.0f, 0.0f, 0.0f,
                                     0.0f, 0.0f, 1.0f, 0.0f,
                                     0.0f, 0.0f, 0.0f, 1.0f};

    SpaceToBatchTest(inputs, inputs_shape, block_shapes, blocks_shape, pads_begins, 
                     pads_ends, pads_shape, outputs, outputs_shape);    
}
