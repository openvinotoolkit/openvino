// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <transformations/common_optimizations/matmul_horizontal_fusing.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

namespace {
using namespace testing;
using namespace ngraph;

struct ReshapeBuilder {
    std::vector<float> values;
    size_t num_of_fused_reshapes;
};

struct ReshapeHorizontalFusingTestValues {
    element::Type input_precision;
    PartialShape input_shape;
    std::int64_t split_axis;
    std::vector<ReshapeBuilder> reshapes_before;
    std::vector<ReshapeBuilder> reshapes_after;
};

std::shared_ptr<Function> get(
    const element::Type input_precision,
    const PartialShape& input_shape,
    const std::int64_t split_axis_value,
    const std::vector<ReshapeBuilder>& reshape_values) {
    auto input = std::make_shared<opset8::Parameter>(input_precision, input_shape);
    ParameterVector inputs{ input };

    OutputVector reshape_inputs(reshape_values.size());
    if (reshape_values.size() == 1 && reshape_values[0].num_of_fused_reshapes > 0) {
        reshape_inputs[0] = input->output(0);
    } else {
        const auto split_axis = ngraph::opset8::Constant::create(ngraph::element::i32, {}, { split_axis_value });
        const auto split = std::make_shared<ngraph::opset8::Split>(input, split_axis, reshape_inputs.size());
        for (size_t i = 0; i < reshape_inputs.size(); ++i) {
            reshape_inputs[i] = split->output(i);
        }
    }

    OutputVector output_nodes;
    for (size_t i = 0; i < reshape_values.size(); ++i) {
        std::shared_ptr<Node> second_input;
        const auto reshape_const = opset8::Constant::create(ngraph::element::i64, { reshape_values[i].values.size() }, reshape_values[i].values);
        const auto reshape = std::make_shared<ngraph::opset8::Reshape>(reshape_inputs[i], reshape_const, true);

        if (reshape_values[i].num_of_fused_reshapes == 0) {
            output_nodes.emplace_back(reshape);
        } else {
            const auto split_axis = opset8::Constant::create(element::i32, Shape{}, { split_axis_value });
            const auto split = std::make_shared<opset8::Split>(reshape, split_axis, reshape_values[i].num_of_fused_reshapes);
            const auto outputs = split->outputs();
            for (const auto& out : outputs) {
                output_nodes.emplace_back(out);
            }
        }
    }

    ResultVector results;
    for (const auto& node : output_nodes) {
        const auto result_node = std::make_shared<ngraph::opset8::Relu>(node);
        results.emplace_back(std::make_shared<ngraph::opset8::Result>(result_node));
    }

    return std::make_shared<Function>(results, inputs);
}

class ReshapeHorizontalFusing : public ::testing::Test, public testing::WithParamInterface<ReshapeHorizontalFusingTestValues> {
public:
    void SetUp() override {
        const auto vals = GetParam();

        f = get(vals.input_precision, vals.input_shape, vals.split_axis, vals.reshapes_before);

        pass::Manager manager;
        auto pass_config = manager.get_pass_config();
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ReshapeHorizontalFusing>();
        manager.run_passes(f);

        f_ref = get(vals.input_precision, vals.input_shape, vals.split_axis, vals.reshapes_after);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ReshapeHorizontalFusingTestValues> obj) {
        const auto vals = obj.param;

        std::ostringstream result;
        result << vals.input_shape << "_" << vals.input_precision << "_split_axis_"
               << vals.split_axis << "_reshapes_before_";
        for (const auto& elem : vals.reshapes_before) {
            result << vector_to_string(elem.values) << "_";
        }

        result << "reshapes_after_";
        for (const auto& elem : vals.reshapes_after) {
            result << vector_to_string(elem.values) << "_" << elem.num_of_fused_reshapes << "_splits}_";
        }
        return result.str();
    }

protected:
    std::shared_ptr<Function> f;
    std::shared_ptr<Function> f_ref;
};

TEST_P(ReshapeHorizontalFusing, CompareFunctions) {
    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

const std::vector<ReshapeHorizontalFusingTestValues> test_values = {
    {
        element::f32, PartialShape{ 1, 128, 768 * 3 }, 2,
        // actual
        {
            {{1, 128, 12, 64}},
            {{1, 128, 12, 64}},
            {{1, 128, 12, 64}},
        },
        // expected
        {
            {{1, 128, 36, 64}, 3},
        }
    },
    {
        element::f32, PartialShape{ 2, 64, 768 * 2 }, 2,
        {
            {{2, 64, 0}},
            {{2, 64, 0}},
        },
        {
            {{2, 64, 0}, 2},
        }
    },
    {
        element::f32, PartialShape{ 2, 64, 768 * 2 }, 1,
        {
            {{1, 0, 12, 64}},
            {{1, 0, 12, 64}},
        },
        {
            {{1, 0, 12, 64}, 2},
        }
    },
    {
        element::f32, PartialShape{ -1, -1, -1 }, 1,
        {
            {{1, 0, 12, 64}},
            {{1, 0, 12, 64}},
        },
        {
            {{1, 0, 12, 64}, 2},
        }
    },
    {
        element::f32, PartialShape{ 2, 128, 64, 2 }, 1,
        {
            {{1, 64, 128, 2}},
            {{1, 64, 128, 2}},
        },
        {
            {{1, 128, 128, 2}, 2},
        }
    },
    {
        element::f32, PartialShape{ 1, 128, 120 }, 2,
        {
            {{2, -1, 3, 20}},
            {{2, -1, 3, 20}},
        },
        {
            {{2, -1, 6, 20}, 2},
        }
    },
    {
        element::f32, PartialShape{ 1, 100, 128, 2 }, 2,
        {
            {{1, 100, 128}},
            {{1, 100, 128}},
        },
        {
            {{1, 100, 256}, 2},
        }
    },
    // 3D -> 4D: splitted dimension before split and 2 last dimensions after reshape doesn't match
    {
        element::f32, PartialShape{ 1, 128, 120 }, 2,
        {
            {{4, -1, 3, 10}},
            {{4, -1, 3, 10}},
        },
        {
            {{4, -1, 3, 10}},
            {{4, -1, 3, 10}},
        }
    },
    // 4D -> 3D: 2 last dimensions after reshape and splitted dimension after split doesn't match
    {
        element::f32, PartialShape{ 1, 100, 128, 2 }, 2,
        {
            {{1, 200, 64}},
            {{1, 200, 64}},
        },
        {
            {{1, 200, 64}},
            {{1, 200, 64}},
        }
    },
    // rank doesn't change and splitted dim changed
    {
        element::f32, PartialShape{ 1, 128, 64 }, 1,
        {
            {{1, 32, 128}},
            {{1, 32, 128}},
        },
        {
            {{1, 32, 128}},
            {{1, 32, 128}},
        }
    },
    // split axis < shape rank after reshape
    {
        element::f32, PartialShape{ 1, 128, 64 }, 2,
        {
            {{1, -1}},
            {{1, -1}},
        },
        {
            {{1, -1}},
            {{1, -1}},
        }
    },
    // dynamic shapes and undefined (non-zero in the pattern) splitted dimension
    {
        element::f32, PartialShape{ -1, -1, -1 }, 2,
        {
            {{2, -1, 3, 20}},
            {{2, -1, 3, 20}},
        },
        {
            {{2, -1, 3, 20}},
            {{2, -1, 3, 20}},
        }
    },
    // dynamic rank
    {
        element::f32, PartialShape::dynamic(), 2,
        {
            {{2, -1, 3, 20}},
            {{2, -1, 3, 20}},
        },
        {
            {{2, -1, 3, 20}},
            {{2, -1, 3, 20}},
        }
    },
    // not all reshape constant match
    {
        element::f32, PartialShape{ 1, 128, 768 * 3 }, 2,
        {
            {{1, 128, 12, 64}},
            {{1, 128, 12, 64}},
            {{1, 128, 64, 12}},
        },
        {
            {{1, 128, 12, 64}},
            {{1, 128, 12, 64}},
            {{1, 128, 64, 12}},
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    TransformationTests,
    ReshapeHorizontalFusing,
    ::testing::ValuesIn(test_values),
    ReshapeHorizontalFusing::getTestCaseName);
} // namespace
