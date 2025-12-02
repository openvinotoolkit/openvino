// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/lstm_cell.hpp"

namespace {
using ov::test::InputShape;
using ov::test::utils::InputLayerType;

typedef std::tuple<
    size_t,                            // batch
    size_t,                            // hidden size
    size_t,                            // input size
    ov::test::utils::InputLayerType,   // W input type (Constant or Parameter)
    ov::test::utils::InputLayerType,   // R input type (Constant or Parameter)
    ov::test::utils::InputLayerType,   // B input type (Constant or Parameter)
    ov::element::Type,                 // Network precision
    std::string                        // Device name
> LSTMCellLayerParamSet;

class LSTMCellLayerGPUTest : public testing::WithParamInterface<LSTMCellLayerParamSet>,
                                 virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LSTMCellLayerParamSet>& obj) {
        const auto& [batch, hidden_size, input_size, WType, RType, BType, model_type, targetDevice] = obj.param;
        std::ostringstream result;
        result << "batch=" << batch << "_";
        result << "hidden_size=" << hidden_size << "_";
        result << "input_size=" << input_size << "_";
        result << "WType=" << WType << "_";
        result << "RType=" << RType << "_";
        result << "BType=" << BType << "_";
        result << "modelType=" << model_type.get_type_name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [batch, hidden_size, input_size, WType, RType, BType, model_type, _targetDevice] = this->GetParam();
        targetDevice = _targetDevice;

        int32_t is = input_size;
        int32_t hs = hidden_size;

        std::vector<ov::PartialShape> init_dyn_shapes = {
            {-1, is},
            {-1, hs},
            {-1, hs},
            {4 * hs, is},
            {4 * hs, hs},
            {4 * hs}
        };

        std::vector<ov::Shape> init_shapes = {
            {batch, input_size},
            {batch, hidden_size},
            {batch, hidden_size},
            {4 * hidden_size, input_size},
            {4 * hidden_size, hidden_size},
            {4 * hidden_size}
        };

        std::vector<InputShape> input_shapes;
        for (size_t i = 0; i < init_dyn_shapes.size(); i++) {
            input_shapes.push_back({init_dyn_shapes[i], {init_shapes[i]}});
        }

        init_input_shapes(input_shapes);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]),
                                std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1]),
                                std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[2])};


        ov::NodeVector inputs{params[0], params[1], params[2]};
        if (WType == InputLayerType::PARAMETER) {
            auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[params.size()]);
            params.push_back(param);
            inputs.push_back(param);
        } else {
            auto tensor = ov::test::utils::create_and_fill_tensor(model_type, init_shapes[3]);
            auto constant = std::make_shared<ov::op::v0::Constant>(tensor);
            inputs.push_back(constant);
        }

        if (RType == InputLayerType::PARAMETER) {
            auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[params.size()]);
            params.push_back(param);
            inputs.push_back(param);
        } else {
            auto tensor = ov::test::utils::create_and_fill_tensor(model_type, init_shapes[4]);
            auto constant = std::make_shared<ov::op::v0::Constant>(tensor);
            inputs.push_back(constant);
        }

        if (BType == InputLayerType::PARAMETER) {
            auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[params.size()]);
            params.push_back(param);
            inputs.push_back(param);
        } else {
            auto tensor = ov::test::utils::create_and_fill_tensor(model_type, init_shapes[5]);
            auto constant = std::make_shared<ov::op::v0::Constant>(tensor);
            inputs.push_back(constant);
        }

        auto lstm_cell = std::make_shared<ov::op::v4::LSTMCell>(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
                                                                hidden_size);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lstm_cell->output(0)),
                                    std::make_shared<ov::op::v0::Result>(lstm_cell->output(1))};
        function = std::make_shared<ov::Model>(results, params, "lstm_cell");
    }
};

TEST_P(LSTMCellLayerGPUTest, Inference) {
    run();
}

std::vector<size_t> batch{1, 2, 5};
std::vector<size_t> hidden_size{8};
std::vector<size_t> input_size{10};
std::vector<ov::test::utils::InputLayerType> layer_types = {
    ov::test::utils::InputLayerType::PARAMETER
};
std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_LSTMCell_dynamic_batch, LSTMCellLayerGPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size),
                                ::testing::ValuesIn(input_size),
                                ::testing::ValuesIn(layer_types),
                                ::testing::ValuesIn(layer_types),
                                ::testing::ValuesIn(layer_types),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        LSTMCellLayerGPUTest::getTestCaseName);

} // namespace
