// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/opsets/opset9.hpp>

#include "shared_test_classes/base/layer_test_utils.hpp"
using namespace ov::opset9;
using namespace ov::element;
using namespace ov;

namespace {
struct FunctionConfig {
    std::vector<size_t> input_shape;
    std::vector<size_t> extra_arg;  // Broadcast target_shape, Tile repeats
    Type ngraph_precision;
    bool use_axes_mapping;  // used only by Broadcast
};

struct ModelWithExpect {
    std::shared_ptr<Model> function;
    std::vector<std::string> input_friendly_names;
    std::vector<std::string> ouput_friendly_names;
};

// Declarations Broadcast and Tile layer creators used to build test model
class LayerForEliminationCreator {
public:
    LayerForEliminationCreator(std::string&& layer_name, std::string&& extra_arg_name, bool contains_non_functional);
    virtual ~LayerForEliminationCreator() = default;

    virtual std::shared_ptr<Node> CreateLayer(const Output<Node>& input, const FunctionConfig& config) const = 0;

    bool IsContainingNonFunctional() const;
    virtual std::string GenerateLayerName(const FunctionConfig& config) const;

private:
    bool _contains_non_functional;
    std::string _layer_name;
    std::string _extra_arg_name;
};

class BroadcastCreator : public LayerForEliminationCreator {
public:
    BroadcastCreator(bool contains_non_functional = false);

    std::shared_ptr<Node> CreateLayer(const Output<Node>& input, const FunctionConfig& config) const override;
    std::string GenerateLayerName(const FunctionConfig& config) const override;
};

class TileCreator : public LayerForEliminationCreator {
public:
    TileCreator(bool contains_non_functional = false);
    std::shared_ptr<Node> CreateLayer(const Output<Node>& input, const FunctionConfig& config) const override;
};

// Full model creators
class ModelExpectCreator {
public:
    ModelExpectCreator(std::shared_ptr<LayerForEliminationCreator>&& layer_creator, std::string&& function_name);
    virtual ~ModelExpectCreator() = default;

    virtual std::string GenerateFunctionName(const FunctionConfig& config) const;
    virtual ModelWithExpect CreateFunctionWithExpects(const FunctionConfig& config) const = 0;

protected:
    std::shared_ptr<Node> CreateLayer(const Output<Node>& input, const FunctionConfig& config) const;

private:
    std::shared_ptr<LayerForEliminationCreator> _layer_creator;
    std::string _function_name;
};

class LayerAfterActivationCreator : public ModelExpectCreator {
public:
    LayerAfterActivationCreator(std::shared_ptr<LayerForEliminationCreator>&& layer_creator);

    ModelWithExpect CreateFunctionWithExpects(const FunctionConfig& config) const override;
};

class LayerBeforeActivationCreator : public ModelExpectCreator {
public:
    LayerBeforeActivationCreator(std::shared_ptr<LayerForEliminationCreator>&& layer_creator);

    ModelWithExpect CreateFunctionWithExpects(const FunctionConfig& config) const override;
};

class LayerTwoOutputsFunctionCreator : public ModelExpectCreator {
public:
    LayerTwoOutputsFunctionCreator(std::shared_ptr<LayerForEliminationCreator>&& layer_creator);

    ModelWithExpect CreateFunctionWithExpects(const FunctionConfig& config) const override;
};

using TestConfig = std::tuple<std::shared_ptr<ModelExpectCreator>,                  // variant of fuction
                              std::pair<std::vector<size_t>, std::vector<size_t>>,  // <input_shape, extra_arg/repeats>
                              InferenceEngine::Precision,                           // net precision
                              bool                                                  // axes mapping
                              >;

// Test class
class BroadcastTileIssue : public ::testing::WithParamInterface<TestConfig>,
                           virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TestConfig>& obj);

protected:
    void SetUp() override;
    void Validate() override;
    static const char* s_target_device_name;
    ModelWithExpect _function_with_expects;
};

// Implementations
LayerForEliminationCreator::LayerForEliminationCreator(std::string&& layer_name,
                                                       std::string&& extra_arg_name,
                                                       bool contains_non_functional)
    : _layer_name(layer_name),
      _contains_non_functional(contains_non_functional),
      _extra_arg_name(std::move(extra_arg_name)) {}

bool LayerForEliminationCreator::IsContainingNonFunctional() const {
    return _contains_non_functional;
}

std::string LayerForEliminationCreator::GenerateLayerName(const FunctionConfig& config) const {
    std::stringstream name;
    name << _layer_name;
    if (_contains_non_functional) {
        name << "_with_non_functional";
    }
    name << "_" << _extra_arg_name << "=" << ov::test::utils::vec2str(config.extra_arg);
    return name.str();
}

BroadcastCreator::BroadcastCreator(bool contains_non_functional)
    : LayerForEliminationCreator("Broadcast", "AxesMapping", contains_non_functional) {}

std::shared_ptr<Node> BroadcastCreator::CreateLayer(const Output<Node>& input, const FunctionConfig& config) const {
    Output<Node> broadcast_input = input;
    auto input_shape = config.input_shape;
    auto targtet_shape = config.extra_arg;

    if (IsContainingNonFunctional()) {
        auto shape_pattern = Constant::create(i32, Shape{input_shape.size()}, input_shape);
        broadcast_input = std::make_shared<Reshape>(input, shape_pattern, false);
    }

    auto target_shape_const = Constant::create(i32, Shape{targtet_shape.size()}, targtet_shape);

    if (config.use_axes_mapping) {
        std::vector<size_t> axes;
        for (int i = 1; i <= input_shape.size(); ++i) {
            axes.push_back(i);
        }
        auto axes_mapping = Constant::create(i32, Shape{axes.size()}, axes);

        return std::make_shared<Broadcast>(broadcast_input, target_shape_const, axes_mapping);
    }

    return std::make_shared<Broadcast>(broadcast_input, target_shape_const);
}

std::string BroadcastCreator::GenerateLayerName(const FunctionConfig& config) const {
    std::stringstream name;
    name << LayerForEliminationCreator::GenerateLayerName(config) << "_";
    name << "UseAxesMapping=" << (config.use_axes_mapping ? "ON" : "OFF");
    return name.str();
}

TileCreator::TileCreator(bool contains_non_functional)
    : LayerForEliminationCreator("Tile", "Repeats", contains_non_functional) {}

std::shared_ptr<Node> TileCreator::CreateLayer(const Output<Node>& input, const FunctionConfig& config) const {
    Output<Node> tile_input = input;
    auto input_shape = config.input_shape;

    auto repeats = config.extra_arg;
    auto repeats_const = Constant::create(i32, Shape{repeats.size()}, repeats);

    if (IsContainingNonFunctional()) {
        auto shape_pattern = Constant::create(i32, Shape{input_shape.size()}, input_shape);
        tile_input = std::make_shared<Reshape>(input, shape_pattern, false);
    }

    return std::make_shared<Tile>(input, repeats_const);
}

std::shared_ptr<Node> ModelExpectCreator::CreateLayer(const Output<Node>& input, const FunctionConfig& config) const {
    return _layer_creator->CreateLayer(input, config);
}

ModelExpectCreator::ModelExpectCreator(std::shared_ptr<LayerForEliminationCreator>&& layer_creator,
                                       std::string&& function_name)
    : _layer_creator(std::move(layer_creator)),
      _function_name(std::move(function_name)) {}

std::string ModelExpectCreator::GenerateFunctionName(const FunctionConfig& config) const {
    std::stringstream name;
    name << _function_name << "_";
    name << _layer_creator->GenerateLayerName(config);
    return name.str();
}

LayerAfterActivationCreator::LayerAfterActivationCreator(std::shared_ptr<LayerForEliminationCreator>&& layer_creator)
    : ModelExpectCreator(std::move(layer_creator), "AfterActivation") {}

ModelWithExpect LayerAfterActivationCreator::CreateFunctionWithExpects(const FunctionConfig& config) const {
    const std::string input_friendly_name = "input_1";
    const auto& input_shape = config.input_shape;

    auto input_param = std::make_shared<Parameter>(config.ngraph_precision, Shape{input_shape});
    input_param->set_friendly_name(input_friendly_name);

    auto length = std::accumulate(input_shape.begin(), input_shape.end(), (size_t)1, std::multiplies<size_t>());
    std::vector<float> vector_data(length, 1.0);
    auto constant = std::make_shared<Constant>(config.ngraph_precision, Shape{input_shape}, vector_data);
    auto add = std::make_shared<Add>(input_param, constant);
    auto activation = std::make_shared<Sigmoid>(add);

    auto layer = CreateLayer(activation, config);

    const std::string output_friendly_name = "ouput_1";
    layer->set_friendly_name(output_friendly_name);
    auto result = std::make_shared<Result>(layer);

    auto model =
        std::make_shared<Model>(ResultVector{result}, ParameterVector{input_param}, GenerateFunctionName(config));
    return {model, {input_friendly_name}, {output_friendly_name}};
}

LayerBeforeActivationCreator::LayerBeforeActivationCreator(std::shared_ptr<LayerForEliminationCreator>&& layer_creator)
    : ModelExpectCreator(std::move(layer_creator), "LayerBeforeActivation") {}

ModelWithExpect LayerBeforeActivationCreator::CreateFunctionWithExpects(const FunctionConfig& config) const {
    const std::string input_friendly_name = "input_1";
    const auto& input_shape = config.input_shape;
    auto input_param = std::make_shared<Parameter>(config.ngraph_precision, Shape{input_shape});
    input_param->set_friendly_name(input_friendly_name);

    auto layer = CreateLayer(input_param, config);

    auto activation = std::make_shared<Sigmoid>(layer);
    const std::string output_friendly_name = "ouput_1";
    activation->set_friendly_name(output_friendly_name);
    auto result = std::make_shared<Result>(activation);

    auto model =
        std::make_shared<Model>(ResultVector{result}, ParameterVector{input_param}, GenerateFunctionName(config));
    return {model, {input_friendly_name}, {output_friendly_name}};
}

LayerTwoOutputsFunctionCreator::LayerTwoOutputsFunctionCreator(
    std::shared_ptr<LayerForEliminationCreator>&& layer_creator)
    : ModelExpectCreator(std::move(layer_creator), "LayerTwoOutputsFunction") {}

ModelWithExpect LayerTwoOutputsFunctionCreator::CreateFunctionWithExpects(const FunctionConfig& config) const {
    const std::string input_friendly_name = "input_1";
    const auto& input_shape = config.input_shape;
    auto input_param = std::make_shared<Parameter>(config.ngraph_precision, Shape{input_shape});
    input_param->set_friendly_name(input_friendly_name);

    auto layer = CreateLayer(input_param, config);

    const std::string output_friendly_name_1 = "ouput_1";
    layer->set_friendly_name(output_friendly_name_1);
    auto result_1 = std::make_shared<Result>(layer);
    auto length = std::accumulate(input_shape.begin(), input_shape.end(), (size_t)1, std::multiplies<size_t>());
    std::vector<float> vector_data(length, 1.0);
    auto constant = std::make_shared<Constant>(config.ngraph_precision, Shape{input_shape}, vector_data);
    auto add = std::make_shared<Add>(input_param, constant);
    auto activation = std::make_shared<Sigmoid>(add);
    const std::string output_friendly_name_2 = "ouput_2";
    activation->set_friendly_name(output_friendly_name_2);
    auto result_2 = std::make_shared<Result>(activation);

    ResultVector results = {result_1, result_2};
    auto model = std::make_shared<Model>(results, ParameterVector{input_param}, GenerateFunctionName(config));
    return {model, {input_friendly_name}, {output_friendly_name_1, output_friendly_name_2}};
}

const char* BroadcastTileIssue::s_target_device_name = ov::test::utils::DEVICE_GNA;

std::string BroadcastTileIssue::getTestCaseName(const testing::TestParamInfo<TestConfig>& obj) {
    std::shared_ptr<ModelExpectCreator> function_creator;
    InferenceEngine::Precision net_precision;
    std::pair<std::vector<size_t>, std::vector<size_t>> shapes;
    bool axes_mapping;
    std::tie(function_creator, shapes, net_precision, axes_mapping) = obj.param;

    auto precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(net_precision);
    FunctionConfig func_config = {shapes.first, shapes.second, precision, axes_mapping};

    std::stringstream test_name;

    test_name << "IS=" << ov::test::utils::vec2str(shapes.first) << "_";
    test_name << "netPRC=" << precision.get_type_name() << "_";
    test_name << "targetDevice=" << s_target_device_name << "_";
    test_name << "FunctionVariant=" << function_creator->GenerateFunctionName(func_config) << "_";
    return test_name.str();
}

void BroadcastTileIssue::SetUp() {
    targetDevice = s_target_device_name;

    std::shared_ptr<ModelExpectCreator> function_creator;
    std::pair<std::vector<size_t>, std::vector<size_t>> shapes;
    bool axes_mapping;
    InferenceEngine::Precision net_precision;

    std::tie(function_creator, shapes, net_precision, axes_mapping) = GetParam();
    std::vector<size_t> input_shape = shapes.first;
    std::vector<size_t> extra_arg = shapes.second;

    auto ngraph_precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(net_precision);

    FunctionConfig func_config = {input_shape, extra_arg, ngraph_precision, axes_mapping};

    _function_with_expects = function_creator->CreateFunctionWithExpects(func_config);
    function = _function_with_expects.function;
}

void BroadcastTileIssue::Validate() {
    LayerTestsCommon::Validate();
    auto inputs = executableNetwork.GetInputsInfo();
    ASSERT_EQ(_function_with_expects.input_friendly_names.size(), inputs.size());
    for (const auto& name : _function_with_expects.input_friendly_names) {
        ASSERT_TRUE(inputs.end() != inputs.find(name));
    }

    auto outputs = executableNetwork.GetOutputsInfo();
    ASSERT_EQ(_function_with_expects.ouput_friendly_names.size(), outputs.size());
    for (const auto& name : _function_with_expects.ouput_friendly_names) {
        ASSERT_TRUE(outputs.end() != outputs.find(name));
    }
}

}  // namespace

TEST_P(BroadcastTileIssue, CompareWithRefs) {
    Run();
}

static const std::pair<std::vector<size_t>, std::vector<size_t>> broadcast_shape_the_same = {{1, 1, 590}, {1, 1, 590}};
static const std::pair<std::vector<size_t>, std::vector<size_t>> broadcast_shape_output_higher_rank = {{1, 1, 590},
                                                                                                       {1, 1, 1, 590}};

static const std::pair<std::vector<size_t>, std::vector<size_t>> tile_shape_the_same = {{1, 1, 590}, {1, 1, 1}};

static const InferenceEngine::Precision precision = InferenceEngine::Precision::FP32;

std::vector<TestConfig> configs = {
    {std::make_shared<LayerAfterActivationCreator>(std::make_shared<BroadcastCreator>()),
     broadcast_shape_the_same,
     precision,
     false},
    {std::make_shared<LayerAfterActivationCreator>(std::make_shared<BroadcastCreator>(true)),
     broadcast_shape_the_same,
     precision,
     false},
    {std::make_shared<LayerBeforeActivationCreator>(std::make_shared<BroadcastCreator>()),
     broadcast_shape_the_same,
     precision,
     false},
    {std::make_shared<LayerBeforeActivationCreator>(std::make_shared<BroadcastCreator>(true)),
     broadcast_shape_the_same,
     precision,
     false},
    {std::make_shared<LayerBeforeActivationCreator>(std::make_shared<BroadcastCreator>()),
     broadcast_shape_output_higher_rank,
     precision,
     false},
    {std::make_shared<LayerBeforeActivationCreator>(std::make_shared<BroadcastCreator>()),
     broadcast_shape_output_higher_rank,
     precision,
     true},
    {std::make_shared<LayerTwoOutputsFunctionCreator>(std::make_shared<BroadcastCreator>()),
     broadcast_shape_the_same,
     precision,
     false},
    {std::make_shared<LayerTwoOutputsFunctionCreator>(std::make_shared<BroadcastCreator>(true)),
     broadcast_shape_the_same,
     precision,
     false},
    {std::make_shared<LayerTwoOutputsFunctionCreator>(std::make_shared<BroadcastCreator>()),
     broadcast_shape_output_higher_rank,
     precision,
     false},
    {std::make_shared<LayerTwoOutputsFunctionCreator>(std::make_shared<BroadcastCreator>()),
     broadcast_shape_output_higher_rank,
     precision,
     true},
    {std::make_shared<LayerAfterActivationCreator>(std::make_shared<TileCreator>()),
     tile_shape_the_same,
     precision,
     false},
    {std::make_shared<LayerAfterActivationCreator>(std::make_shared<TileCreator>(true)),
     tile_shape_the_same,
     precision,
     false},
    {std::make_shared<LayerBeforeActivationCreator>(std::make_shared<TileCreator>()),
     tile_shape_the_same,
     precision,
     false},
    {std::make_shared<LayerBeforeActivationCreator>(std::make_shared<TileCreator>(true)),
     tile_shape_the_same,
     precision,
     false},
    {std::make_shared<LayerTwoOutputsFunctionCreator>(std::make_shared<TileCreator>()),
     tile_shape_the_same,
     precision,
     false},
    {std::make_shared<LayerTwoOutputsFunctionCreator>(std::make_shared<TileCreator>(true)),
     tile_shape_the_same,
     precision,
     false},
};

INSTANTIATE_TEST_SUITE_P(broadcast_tile_issue,
                         BroadcastTileIssue,
                         ::testing::ValuesIn(configs),
                         BroadcastTileIssue::getTestCaseName);
