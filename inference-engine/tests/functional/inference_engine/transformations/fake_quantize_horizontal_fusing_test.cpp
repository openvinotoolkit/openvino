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
struct FakeQuantizeBuilder {
    struct Interval {
        PartialShape shape;
        std::vector<float> values;
    };

    element::Type out_precision;
    size_t levels;
    Interval il;
    Interval ih;
    Interval ol;
    Interval oh;
    size_t num_of_fused_fq;
};

struct FakeQuantizeHorizontalFusingTestValues {
    element::Type input_precision;
    PartialShape input_shape;
    std::int64_t split_axis;
    std::vector<FakeQuantizeBuilder> fake_quantizes_before;
    std::vector<FakeQuantizeBuilder> fake_quantizes_after;
};

std::shared_ptr<Function> get(
    const element::Type input_precision,
    const PartialShape& input_shape,
    const std::int64_t split_axis_value,
    const std::vector<FakeQuantizeBuilder>& fq_values) {
    auto input = std::make_shared<opset8::Parameter>(input_precision, input_shape);
    ParameterVector inputs{ input };
    OutputVector matmul_inputs(fq_values.size());

    OutputVector fq_inputs(fq_values.size());
    if (fq_values.size() == 1 && fq_values[0].num_of_fused_fq > 0) {
        fq_inputs[0] = input->output(0);
    } else {
        const auto split_axis = ngraph::opset8::Constant::create(ngraph::element::i32, {}, { split_axis_value });
        const auto split = std::make_shared<ngraph::opset8::Split>(input, split_axis, fq_inputs.size());
        for (size_t i = 0; i < fq_inputs.size(); ++i) {
            fq_inputs[i] = split->output(i);
        }
    }
    OutputVector output_nodes;
    for (size_t i = 0; i < fq_values.size(); ++i) {
        std::shared_ptr<Node> second_input;
        const auto il = opset8::Constant::create(ngraph::element::f32, fq_values[i].il.shape.to_shape(), fq_values[i].il.values);
        const auto ih = opset8::Constant::create(ngraph::element::f32, fq_values[i].ih.shape.to_shape(), fq_values[i].ih.values);
        const auto ol = opset8::Constant::create(ngraph::element::f32, fq_values[i].ol.shape.to_shape(), fq_values[i].ol.values);
        const auto oh = opset8::Constant::create(ngraph::element::f32, fq_values[i].oh.shape.to_shape(), fq_values[i].oh.values);

        const auto fq = ngraph::opset1::FakeQuantize(fq_inputs[i], il, ih, ol, oh, fq_values[i].levels);
        const auto relaxed_fq = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::FakeQuantize>>(fq, fq_values[i].out_precision);

        if (fq_values[i].num_of_fused_fq == 0) {
            output_nodes.emplace_back(relaxed_fq);
        } else {
            const auto split_axis = opset8::Constant::create(element::i32, Shape{}, { split_axis_value });
            const auto split = std::make_shared<opset8::Split>(relaxed_fq, split_axis, fq_values[i].num_of_fused_fq);
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

class FakeQuantizeHorizontalFusing : public ::testing::Test, public testing::WithParamInterface<FakeQuantizeHorizontalFusingTestValues> {
public:
    void SetUp() override {
        const auto vals = GetParam();

        f = get(vals.input_precision, vals.input_shape, vals.split_axis, vals.fake_quantizes_before);

        pass::Manager manager;
        auto pass_config = manager.get_pass_config();
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::FakeQuantizeHorizontalFusing>();
        manager.run_passes(f);

        f_ref = get(vals.input_precision, vals.input_shape, vals.split_axis, vals.fake_quantizes_after);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FakeQuantizeHorizontalFusingTestValues> obj) {
        const auto vals = obj.param;

        std::ostringstream result;
        result << vals.input_shape << "_" << vals.input_precision << "_split_axis_"
               << vals.split_axis << "_" << "fq_before_";
        for (const auto& elem : vals.fake_quantizes_before) {
            result << "{levels_" << elem.levels << "_out_precision_" << elem.out_precision
                   << "_il_" << elem.il.shape << "_" << vector_to_string(elem.il.values)
                   << "_ih_" << elem.ih.shape << "_" << vector_to_string(elem.ih.values)
                   << "_ol_" << elem.ol.shape << "_" << vector_to_string(elem.ol.values)
                   << "_oh_" << elem.oh.shape << "_" << vector_to_string(elem.oh.values)
                   << "}";
        }

        result << "fq_after_";
        for (const auto& elem : vals.fake_quantizes_after) {
            result << "{levels_" << elem.levels << "_out_precision_" << elem.out_precision
                   << "_il_" << elem.il.shape << "_" << vector_to_string(elem.il.values)
                   << "_ih_" << elem.ih.shape << "_" << vector_to_string(elem.ih.values)
                   << "_ol_" << elem.ol.shape << "_" << vector_to_string(elem.ol.values)
                   << "_oh_" << elem.oh.shape << "_" << vector_to_string(elem.oh.values)
                   << "split_after_has_" << elem.num_of_fused_fq << "_splits}_";
        }
        return result.str();
    }

protected:
    std::shared_ptr<Function> f;
    std::shared_ptr<Function> f_ref;
};

TEST_P(FakeQuantizeHorizontalFusing, CompareFunctions) {
    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

const std::vector<FakeQuantizeHorizontalFusingTestValues> test_values = {
    {
        element::f32, PartialShape{ 1, 4, 8 }, 2,
        // actual
        {
            { element::f32, 256, {{}, {0.f}}, {{}, {25.5f}}, {{}, {0.f}}, {{}, {25.5f}} },
            { element::f32, 256, {{}, {0.f}}, {{}, {25.5f}}, {{}, {0.f}}, {{}, {25.5f}} },
        },
        // expected
        {
            { element::f32, 256, {{}, {0.f}}, {{}, {25.5f}}, {{}, {0.f}}, {{}, {25.5f}}, 2 },
        }
    },
    {
        element::f32, PartialShape{ 1, 4, 4 }, 2,
        // actual
        {
            { element::f32, 256, {{}, {-12.8f}}, {{}, {12.7f}}, {{}, {-12.8f}}, {{}, {12.7f}} },
            { element::f32, 256, {{}, {-1.28f}}, {{}, {1.27f}}, {{}, {-1.28f}}, {{}, {1.27f}} },
        },
        // expected
        {
            {
                element::f32, 256,
                {{1, 1, 4}, {-12.8f, -12.8f, -1.28f, -1.28f}},
                {{1, 1, 4}, {12.7f, 12.7f, 1.27f, 1.27f}},
                {{1, 1, 4}, {-12.8f, -12.8f, -1.28f, -1.28f}},
                {{1, 1, 4}, {12.7f, 12.7f, 1.27f, 1.27f}}, 2
            },
        }
    },
    {
        element::f32, PartialShape{ 1, 4, 8 }, 2,
        {
            { element::u8, 256, {{}, {0.f}}, {{}, {25.5f}}, {{}, {0.f}}, {{}, {255.f}} },
            { element::u8, 256, {{}, {0.f}}, {{}, {25.5f}}, {{}, {0.f}}, {{}, {255.f}} },
        },
        {
            { element::u8, 256, {{}, {0.f}}, {{}, {25.5f}}, {{}, {0.f}}, {{}, {255.f}}, 2 },
        }
    },
    {
        element::f32, PartialShape{ 1, 4, 6 }, 2,
        {
            { element::u8, 256, {{}, {0.125f}}, {{}, {0.255f}}, {{}, {0.f}}, {{}, {255.f}} },
            { element::u8, 256, {{}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
            { element::u8, 256, {{}, {12.5f}}, {{}, {25.5f}}, {{}, {0.f}}, {{}, {255.f}} },
        },
        {
            {
                element::u8, 256,
                {{1, 1, 6}, {0.125f, 0.125f, 1.25f, 1.25f, 12.5f, 12.5f}},
                {{1, 1, 6}, {0.255f, 0.255f, 2.55f, 2.55f, 25.5f, 25.5f}},
                {{}, {0.f}}, {{}, {255.f}}, 3
            },
        }
    },
    {
        element::f32, PartialShape{ 1, 4, 8 }, 1,
        {
            { element::u8, 256, {{1, 2, 1}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
            { element::u8, 256, {{1, 2, 1}, {12.5f}}, {{}, {25.5f}}, {{}, {0.f}}, {{}, {255.f}} },
        },
        {
            {
                element::u8, 256,
                {{1, 4, 1}, {1.25f, 1.25f, 12.5f, 12.5f}},
                {{1, 4, 1}, {2.55f, 2.55f, 25.5f, 25.5f}},
                {{}, {0.f}}, {{}, {255.f}}, 2
            },
        }
    },
    {
        element::f32, PartialShape{ 1, 4, 8 }, 1,
        {
            { element::u8, 256, {{}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
            { element::u8, 256, {{1, 2, 1}, {12.5f}}, {{}, {25.5f}}, {{}, {0.f}}, {{}, {255.f}} },
        },
        {
            {
                element::u8, 256,
                {{1, 4, 1}, {1.25f, 1.25f, 12.5f, 12.5f}},
                {{1, 4, 1}, {2.55f, 2.55f, 25.5f, 25.5f}},
                {{}, {0.f}}, {{}, {255.f}}, 2
            },
        }
    },
    {
        element::f32, PartialShape{ -1, 4, -1 }, 1,
        {
            { element::u8, 256, {{}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
            { element::u8, 256, {{1, 2, 1}, {12.5f}}, {{}, {25.5f}}, {{}, {0.f}}, {{}, {255.f}} },
        },
        {
            {
                element::u8, 256,
                {{1, 4, 1}, {1.25f, 1.25f, 12.5f, 12.5f}},
                {{1, 4, 1}, {2.55f, 2.55f, 25.5f, 25.5f}},
                {{}, {0.f}}, {{}, {255.f}}, 2
            },
        }
    },
    {
        element::f32, PartialShape{ -1, -1, -1 }, 1,
        {
            { element::f32, 256, {{1, 2, 1}, {1.25f}}, {{1, 2, 1}, {2.55f}}, {{1, 2, 1}, {1.25f}}, {{1, 2, 1}, {2.55f}} },
            { element::f32, 256, {{1, 2, 1}, {12.5f}}, {{1, 2, 1}, {25.5f}}, {{1, 2, 1}, {12.5f}}, {{1, 2, 1}, {25.5f}} },
        },
        {
            {
                element::f32, 256,
                {{1, 4, 1}, {1.25f, 1.25f, 12.5f, 12.5f}},
                {{1, 4, 1}, {2.55f, 2.55f, 25.5f, 25.5f}},
                {{1, 4, 1}, {1.25f, 1.25f, 12.5f, 12.5f}},
                {{1, 4, 1}, {2.55f, 2.55f, 25.5f, 25.5f}}, 2
            },
        }
    },
    // dynamic split dimension
    {
        element::f32, PartialShape{ -1, -1, -1 }, 1,
        {
            { element::u8, 256, {{}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
            { element::u8, 256, {{1, 2, 1}, {12.5f}}, {{}, {25.5f}}, {{}, {0.f}}, {{}, {255.f}} },
        },
        {
            { element::u8, 256, {{}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
            { element::u8, 256, {{1, 2, 1}, {12.5f}}, {{}, {25.5f}}, {{}, {0.f}}, {{}, {255.f}} },
        }
    },
    // dynamic rank
    {
        element::f32, PartialShape::dynamic(), 1,
        {
            { element::u8, 256, {{}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
            { element::u8, 256, {{1, 2, 1}, {12.5f}}, {{}, {25.5f}}, {{}, {0.f}}, {{}, {255.f}} },
        },
        {
            { element::u8, 256, {{}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
            { element::u8, 256, {{1, 2, 1}, {12.5f}}, {{}, {25.5f}}, {{}, {0.f}}, {{}, {255.f}} },
        }
    },
    // different fq precisions
    {
        element::f32, PartialShape{ 1, 4, 8 }, 1,
        {
            { element::u8, 256, {{}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
            { element::i8, 256, {{}, {-12.5f}}, {{}, {12.5f}}, {{}, {-128.f}}, {{}, {127.f}} },
        },
        {
            { element::u8, 256, {{}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
            { element::i8, 256, {{}, {-12.5f}}, {{}, {12.5f}}, {{}, {-128.f}}, {{}, {127.f}} },
        }
    },
    // different fq levels
    {
        element::f32, PartialShape{ 1, 4, 8 }, 1,
        {
            { element::u8, 256, {{}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
            { element::u8, 255, {{}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
        },
        {
            { element::u8, 256, {{}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
            { element::u8, 255, {{}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
        }
    },
    // different quantization axis
    {
        element::f32, PartialShape{ 1, 4, 4 }, 1,
        {
            { element::u8, 256, {{1, 2, 1}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
            { element::u8, 255, {{1, 1, 4}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
        },
        {
            { element::u8, 256, {{1, 2, 1}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
            { element::u8, 255, {{1, 1, 4}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
        }
    },
    // quantization axis and split axis mismatch
    {
        element::f32, PartialShape{ 1, 4, 8 }, 1,
        {
            { element::u8, 256, {{1, 1, 8}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
            { element::u8, 255, {{1, 1, 8}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
        },
        {
            { element::u8, 256, {{1, 1, 8}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
            { element::u8, 255, {{1, 1, 8}, {1.25f}}, {{}, {2.55f}}, {{}, {0.f}}, {{}, {255.f}} },
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    TransformationTests,
    FakeQuantizeHorizontalFusing,
    ::testing::ValuesIn(test_values),
    FakeQuantizeHorizontalFusing::getTestCaseName);
} // namespace
