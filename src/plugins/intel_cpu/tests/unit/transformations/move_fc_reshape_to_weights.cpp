// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/cpu_opset/common/pass/move_fc_reshape_to_weights.hpp>

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include "ov_ops/fully_connected.hpp"
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;

enum class ZeroPointType { NO_ZP, ZP_WEIGHTS_PRC, ZP_DECOMPRESSION_PRC };
inline std::ostream& operator<<(std::ostream& os, ZeroPointType type) {
    switch (type) {
        case ZeroPointType::NO_ZP:
            os << "NO_ZP";
            break;
        case ZeroPointType::ZP_WEIGHTS_PRC:
            os << "ZP_WEIGHTS_PRC";
            break;
        case ZeroPointType::ZP_DECOMPRESSION_PRC:
            os << "ZP_DECOMPRESSION_PRC";
            break;
        default:
            OPENVINO_THROW("Unknown ZeroPointType");
    }
    return os;
}

enum class ZeroPointShape { SCALAR, PER_CHANNEL };
inline std::ostream& operator<<(std::ostream& os, ZeroPointShape type) {
    switch (type) {
        case ZeroPointShape::SCALAR:
            os << "SCALAR";
            break;
        case ZeroPointShape::PER_CHANNEL:
            os << "PER_CHANNEL";
            break;
        default:
            OPENVINO_THROW("Unknown ZeroPointShape");
    }
    return os;
}

using MoveFCReshapeToWeightsParams = std::tuple<std::pair<ov::PartialShape, ov::Shape>,  // data_shape - weights_shape
                                                bool,                                    // add transpose
                                                ZeroPointType,
                                                ZeroPointShape>;

class MoveFCReshapeToWeightsTests : public TransformationTestsF, public WithParamInterface<MoveFCReshapeToWeightsParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MoveFCReshapeToWeightsParams> obj) {
        std::pair<ov::PartialShape, ov::Shape> input_shapes;
        bool add_transpose;
        ZeroPointType zp_type;
        ZeroPointShape zp_shape;
        std::tie(input_shapes, add_transpose, zp_type, zp_shape) = obj.param;

        std::ostringstream result;
        result << "Input_shape=(" << input_shapes.first << ")_Weights_shape=(" << input_shapes.second
               << ")_add_transpose=" << add_transpose << "_zp_type=" << zp_type << "_zp_shape=" << zp_shape;
        return result.str();
    }

    static std::shared_ptr<ov::Model> initModel(const ov::PartialShape& data_shape,
                                                const ov::Shape& weights_shape,
                                                const bool add_transpose,
                                                const ZeroPointType zp_type,
                                                ZeroPointShape zp_shape,
                                                const bool add_reshape) {
        const auto decompression_prc = ov::element::f32;
        const auto weights_prc = ov::element::u8;
        auto data = std::make_shared<ov::opset1::Parameter>(decompression_prc, data_shape);
        auto transposed_shape = weights_shape;
        if (add_transpose)
            std::swap(*(transposed_shape.rbegin() + 1), *transposed_shape.rbegin());
        std::shared_ptr<ov::Node> weights_path = ov::opset1::Constant::create(weights_prc, transposed_shape, {1});
        weights_path = std::make_shared<ov::opset1::Convert>(weights_path, decompression_prc);

        ov::Shape decompression_shape(weights_shape.size(), 1);
        const size_t n_idx = add_transpose ? transposed_shape.size() - 1 : transposed_shape.size() - 2;
        decompression_shape[n_idx] = transposed_shape[n_idx];
        ov::Shape subtract_shape = zp_shape != ZeroPointShape::SCALAR ? decompression_shape : ov::Shape(weights_shape.size(), 1);

        if (zp_type == ZeroPointType::ZP_DECOMPRESSION_PRC) {
            auto sub_const = ov::opset1::Constant::create(weights_prc, subtract_shape, {1});
            auto sub_convert = std::make_shared<ov::opset1::Convert>(sub_const, decompression_prc);
            weights_path = std::make_shared<ov::opset1::Subtract>(weights_path, sub_convert);
        } else if (zp_type == ZeroPointType::ZP_WEIGHTS_PRC) {
            auto sub_const = ov::opset1::Constant::create(decompression_prc, subtract_shape, {1});
            weights_path = std::make_shared<ov::opset1::Subtract>(weights_path, sub_const);
        }

        auto mul_const = ov::opset1::Constant::create(decompression_prc, decompression_shape, {1});
        weights_path = std::make_shared<ov::opset1::Multiply>(weights_path, mul_const);

        if (add_reshape) {
            auto target_shape = transposed_shape;
            target_shape.erase(target_shape.begin());
            auto reshape_const = ov::opset1::Constant::create(ov::element::i32, {2}, target_shape);
            weights_path = std::make_shared<ov::opset1::Reshape>(weights_path, reshape_const, false);
        }
        if (add_transpose) {
            auto transpose_const = ov::opset1::Constant::create(ov::element::i32, {2}, {1, 0});
            weights_path = std::make_shared<ov::opset1::Transpose>(weights_path, transpose_const);
        }

        auto fully_connected = std::make_shared<ov::op::internal::FullyConnected>(
            data,
            weights_path,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));

        return std::make_shared<ov::Model>(ov::NodeVector{fully_connected}, ov::ParameterVector{data});
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        std::pair<ov::PartialShape, ov::Shape> input_shapes;
        bool add_transpose;
        ZeroPointType zp_type;
        ZeroPointShape zp_shape;
        std::tie(input_shapes, add_transpose, zp_type, zp_shape) = this->GetParam();

        ov::Shape ref_weights_shape = input_shapes.second;
        ref_weights_shape.erase(ref_weights_shape.begin());
        model = initModel(input_shapes.first, input_shapes.second, add_transpose, zp_type, zp_shape, true);
        model_ref = initModel(input_shapes.first, ref_weights_shape, add_transpose, zp_type, zp_shape, false);
        manager.register_pass<MoveFCReshapeToWeights>();
    }
};

TEST_P(MoveFCReshapeToWeightsTests, CompareFunctions) {}

const std::vector<std::pair<ov::PartialShape, ov::Shape>> input_shapes_wo_transpose = {
    {{-1, -1, -1}, {1, 4, 3}}
};
const std::vector<bool> add_transpose = {false, true};
const std::vector<ZeroPointType> zp_types = {
    ZeroPointType::NO_ZP,
    ZeroPointType::ZP_DECOMPRESSION_PRC,
    ZeroPointType::ZP_WEIGHTS_PRC
};
const std::vector<ZeroPointShape> zp_shapes = {
    ZeroPointShape::SCALAR,
    ZeroPointShape::PER_CHANNEL
};

INSTANTIATE_TEST_SUITE_P(TransformationTests_wo_transpose, MoveFCReshapeToWeightsTests,
                        ::testing::Combine(
                                ::testing::ValuesIn(input_shapes_wo_transpose),
                                ::testing::ValuesIn(add_transpose),
                                ::testing::ValuesIn(zp_types),
                                ::testing::ValuesIn(zp_shapes)),
                            MoveFCReshapeToWeightsTests::getTestCaseName);
