// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/reduce.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"

namespace {
using ov::test::InputShape;

typedef struct {
    std::vector<InputShape> data_shape;
    std::vector<int>        axes;
} ReduceInput;

typedef std::tuple<
    ReduceInput,                      // input data (data shape, axes shape, axes values)
    ov::element::Type,                // presion of inputs
    ov::test::utils::ReductionType,   // reduction type
    bool,                             // keepDims
    std::string                       // device name
> ReduceLayerTestParamSet;

class ReduceLayerGPUTest : public testing::WithParamInterface<ReduceLayerTestParamSet>,
                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReduceLayerTestParamSet>& obj) {
        ReduceInput input_data;
        ov::element::Type netType;
        ov::test::utils::ReductionType reductionType;
        bool keepDims;
        std::string targetDevice;
        std::tie(input_data, netType, reductionType, keepDims, targetDevice) = obj.param;

        std::vector<InputShape> inshapes = input_data.data_shape;
        std::vector<int> axes = input_data.axes;

        std::ostringstream result;

        result << "IS=";
        for (const auto& shape : inshapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inshapes) {
            for (const auto& item : shape.second) {
                result << ov::test::utils::vec2str(item) << "_";
            }
        }
        result << "axes=";
        result << ov::test::utils::vec2str(axes) << "_";

        result << "Precision=" << netType << "_";
        result << "reductionType=" << reductionType << "_";
        result << "keepDims=" << keepDims << "_";
        result << "trgDev=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        ReduceInput input_data;
        ov::element::Type netPrecision;
        ov::test::utils::ReductionType reductionType;
        bool keepDims;
        std::tie(input_data, netPrecision, reductionType, keepDims, targetDevice) = this->GetParam();

        std::vector<InputShape> inputShapes = input_data.data_shape;
        std::vector<int> axes = input_data.axes;

        init_input_shapes(inputShapes);

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes)
            params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));

        std::vector<size_t> shapeAxes;
        shapeAxes.push_back(axes.size());

        auto reductionAxesNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape(shapeAxes), axes);

        const auto reduce = ov::test::utils::make_reduce(params[0], reductionAxesNode, keepDims, reductionType);

        auto makeFunction = [](ov::ParameterVector &params, const std::shared_ptr<ov::Node> &lastNode) {
            ov::ResultVector results;

            for (size_t i = 0; i < lastNode->get_output_size(); i++)
                results.push_back(std::make_shared<ov::op::v0::Result>(lastNode->output(i)));

            return std::make_shared<ov::Model>(results, params, "ReduceLayerGPUTest");
        };

        function = makeFunction(params, reduce);
    }
};

TEST_P(ReduceLayerGPUTest, Inference) {
   run();
}

const std::vector<bool> keepDims = {
    true,
    false,
};

const std::vector<ov::element::Type> float_types = {
    ov::element::f32,
    ov::element::f16,
};

const std::vector<ov::element::Type> float_int_types = {
    ov::element::f32,
    ov::element::f16,
    ov::element::i32,
};

const ReduceInput dyn1d = {
    {
        { {-1}, {{4}, {5}} }
    },
    {0}
};

const ReduceInput dyn2d = {
    {
        { {-1, -1}, {{100, 3}, {5, 6}} }
    },
    {1}
};

const ReduceInput dyn3d = {
    {
        { {-1, -1, -1}, {{4, 5, 6}, {5, 1, 6}} }
    },
    {0, -1}
};

const ReduceInput dyn4d = {
    {
        { {-1, -1, -1, -1}, {{2, 3, 4, 5}, {5, 4, 3, 1}} }
    },
    {1, -2}
};

const ReduceInput dyn5d = {
    {
        { {-1, -1, -1, -1, -1}, {{2, 3, 4, 5, 6}, {5, 6, 3, 1, 2}} }
    },
    {-3, 3}
};

const ReduceInput dyn6d = {
    {
        { {-1, -1, -1, -1, -1, -1}, {{2, 3, 4, 5, 6, 7}, {5, 4, 3, 1, 2, 6}} }
    },
    {1}
};


// ================== Reduction int32/float types (Sum, Min, Max, L1) ==================
const auto reduceSum = ::testing::Combine(
        ::testing::ValuesIn({dyn1d, dyn5d}),
        ::testing::ValuesIn(float_int_types),
        ::testing::Values(ov::test::utils::ReductionType::Sum),
        ::testing::ValuesIn(keepDims),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);
INSTANTIATE_TEST_SUITE_P(smoke_reduce_sum_compareWithRefs_dynamic, ReduceLayerGPUTest, reduceSum, ReduceLayerGPUTest::getTestCaseName);

const auto reduceMin = ::testing::Combine(
        ::testing::ValuesIn({dyn2d, dyn6d}),
        ::testing::ValuesIn(float_int_types),
        ::testing::Values(ov::test::utils::ReductionType::Min),
        ::testing::ValuesIn(keepDims),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);
INSTANTIATE_TEST_SUITE_P(smoke_reduce_min_compareWithRefs_dynamic, ReduceLayerGPUTest, reduceMin, ReduceLayerGPUTest::getTestCaseName);

const auto reduceMax = ::testing::Combine(
        ::testing::ValuesIn({dyn3d, dyn5d}),
        ::testing::ValuesIn(float_int_types),
        ::testing::Values(ov::test::utils::ReductionType::Max),
        ::testing::ValuesIn(keepDims),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);
INSTANTIATE_TEST_SUITE_P(smoke_reduce_max_compareWithRefs_dynamic, ReduceLayerGPUTest, reduceMax, ReduceLayerGPUTest::getTestCaseName);

const auto reduceL1 = ::testing::Combine(
        ::testing::ValuesIn({dyn4d, dyn6d}),
        ::testing::ValuesIn(float_int_types),
        ::testing::Values(ov::test::utils::ReductionType::L1),
        ::testing::ValuesIn(keepDims),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);
INSTANTIATE_TEST_SUITE_P(smoke_reduce_l1_compareWithRefs_dynamic, ReduceLayerGPUTest, reduceL1, ReduceLayerGPUTest::getTestCaseName);


// ================== Reduction float types (Mean, Prod, L2) ==================
const auto reduceMean = ::testing::Combine(
        ::testing::ValuesIn({dyn1d, dyn6d}),
        ::testing::ValuesIn(float_types),
        ::testing::Values(ov::test::utils::ReductionType::Mean),
        ::testing::ValuesIn(keepDims),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);
INSTANTIATE_TEST_SUITE_P(smoke_reduce_mean_compareWithRefs_dynamic, ReduceLayerGPUTest, reduceMean, ReduceLayerGPUTest::getTestCaseName);

const auto reduceProd = ::testing::Combine(
        ::testing::ValuesIn({dyn2d, dyn4d}),
        ::testing::ValuesIn({ov::element::f32}),
        ::testing::Values(ov::test::utils::ReductionType::Prod),
        ::testing::ValuesIn(keepDims),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);
INSTANTIATE_TEST_SUITE_P(smoke_reduce_prod_compareWithRefs_dynamic, ReduceLayerGPUTest, reduceProd, ReduceLayerGPUTest::getTestCaseName);

const auto reduceL2 = ::testing::Combine(
        ::testing::ValuesIn({dyn4d, dyn5d}),
        ::testing::ValuesIn(float_types),
        ::testing::Values(ov::test::utils::ReductionType::L2),
        ::testing::ValuesIn(keepDims),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);
INSTANTIATE_TEST_SUITE_P(smoke_reduce_l2_compareWithRefs_dynamic, ReduceLayerGPUTest, reduceL2, ReduceLayerGPUTest::getTestCaseName);


// ================== Reduction logical types (LogicalOr, LogicalAnd) ==================
const auto reduceLogicalOr = ::testing::Combine(
        ::testing::ValuesIn({dyn1d, dyn6d}),
        ::testing::Values(ov::element::boolean),
        ::testing::Values(ov::test::utils::ReductionType::LogicalOr),
        ::testing::ValuesIn(keepDims),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);
INSTANTIATE_TEST_SUITE_P(smoke_reduce_logicalor_compareWithRefs_dynamic, ReduceLayerGPUTest, reduceLogicalOr, ReduceLayerGPUTest::getTestCaseName);

const auto reduceLogicalAnd = ::testing::Combine(
        ::testing::ValuesIn({dyn3d, dyn5d}),
        ::testing::Values(ov::element::boolean),
        ::testing::Values(ov::test::utils::ReductionType::LogicalAnd),
        ::testing::ValuesIn(keepDims),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);
INSTANTIATE_TEST_SUITE_P(smoke_reduce_logicaland_compareWithRefs_dynamic, ReduceLayerGPUTest, reduceLogicalAnd, ReduceLayerGPUTest::getTestCaseName);


// ================== various reduce-axis ==================
const std::vector<ReduceInput> dynVariousAxisInputs = {
    // 4D
    {
        {
            { {-1, -1, -1, -1}, {{2, 3, 4, 5}, {5, 4, 3, 1}} }
        },
        {0}
    },
    {
        {
            { {-1, -1, -1, -1}, {{2, 3, 4, 5}, {5, 4, 3, 1}} }
        },
        {1, -1}
    },
    {
        {
            { {-1, -1, -1, -1}, {{2, 3, 4, 5}, {5, 3, 7, 1}} }
        },
        {2, 3}
    },
    {
        {
            { {-1, -1, -1, -1}, {{2, 3, 4, 5}, {1, 2, 3, 1}} }
        },
        {0, 2, -1}
    },
    // 5D
    {
        {
            { {-1, -1, -1, -1, -1}, {{2, 4, 3, 4, 5}, {5, 3, 2, 1, 2}} }
        },
        {1}
    },
    {
        {
            { {-1, -1, -1, -1, -1}, {{4, 3, 2, 5, 6}, {5, 3, 2, 1, 4}} }
        },
        {0, -3}
    },
    {
        {
            { {-1, -1, -1, -1, -1}, {{3, 4, 2, 6, 5}, {3, 5, 7, 1, 5}} }
        },
        {2, -2, 4}
    },
    {
        {
            { {-1, -1, -1, -1, -1}, {{4, 2, 5, 1, 9}, {5, 3, 7, 1, 2}} }
        },
        {0, 1, -2, 4}
    },
    // 6D
    {
        {
            { {-1, -1, -1, -1, -1, -1}, {{2, 3, 4, 5, 6, 7}, {5, 3, 4, 1, 7, 5}} }
        },
        {0}
    },
    {
        {
            { {-1, -1, -1, -1, -1, -1}, {{2, 3, 4, 5, 6, 7}, {5, 3, 5, 1, 2, 5}} }
        },
        {0, -3}
    },
    {
        {
            { {-1, -1, -1, -1, -1, -1}, {{2, 3, 4, 5, 6, 7}, {2, 5, 4, 1, 5, 3}} }
        },
        {2, 3, -2, 5}
    },
    {
        {
            { {-1, -1, -1, -1, -1, -1}, {{2, 3, 4, 5, 6, 7}, {3, 5, 4, 1, 8, 5}} }
        },
        {0, 2, -3, 4, 5}
    },
    {
        {
            { {-1, -1, -1, -1, -1, -1}, {{2, 3, 4, 5, 6, 7}, {7, 5, 3, 1, 6, 9}} }
        },
        {4}
    },
};

const auto reduceMaxWithVariousAxis = ::testing::Combine(
        ::testing::ValuesIn(dynVariousAxisInputs),
        ::testing::Values(ov::element::f32),
        ::testing::Values(ov::test::utils::ReductionType::Max),
        ::testing::ValuesIn(keepDims),
        ::testing::Values(ov::test::utils::DEVICE_GPU)
);
INSTANTIATE_TEST_SUITE_P(smoke_reduce_max_withVariousAxis_compareWithRefs_dynamic,
                        ReduceLayerGPUTest, reduceMaxWithVariousAxis, ReduceLayerGPUTest::getTestCaseName);
} // namespace
