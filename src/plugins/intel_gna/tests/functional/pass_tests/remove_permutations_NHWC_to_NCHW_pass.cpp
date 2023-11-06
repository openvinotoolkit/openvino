// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/pass/convert_prc.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "transformations/common_optimizations/transpose_to_reshape.hpp"

typedef std::tuple<InferenceEngine::Precision,          // Network Precision
                   std::string,                         // Target Device
                   std::map<std::string, std::string>,  // Configuration
                   std::vector<size_t>                  // Input shape
                   >
    removePermutationsPassParams;

typedef std::tuple<InferenceEngine::Precision,          // Network Precision
                   std::string,                         // Target Device
                   std::map<std::string, std::string>,  // Configuration
                   std::vector<size_t>,                 // Input shape
                   bool,                                // additional bool parameter
                   bool                                 // transpose to reshape
                   >
    removePermutationsAddParamPassParams;

typedef std::tuple<InferenceEngine::Precision,          // Network Precision
                   std::string,                         // Target Device
                   std::map<std::string, std::string>,  // Configuration
                   std::vector<size_t>,                 // Input shape
                   size_t                               // Splits number
                   >
    removeSharedPermutationPassParams;

namespace LayerTestsDefinitions {

std::vector<size_t> GetKernelShape(std::vector<size_t> input_shape, size_t kernel_size, bool output_1d = false) {
    if (input_shape.size() == 3) {
        return output_1d ? std::vector<size_t>{input_shape.back()} : std::vector<size_t>{kernel_size};
    }

    if (output_1d) {
        return std::vector<size_t>{input_shape[1], input_shape[2]};
    }

    return (input_shape[1] == 1 ? std::vector<size_t>{1, kernel_size}
                                : (input_shape[2] == 1 ? std::vector<size_t>{kernel_size, 1}
                                                       : std::vector<size_t>{kernel_size, kernel_size}));
}

std::shared_ptr<ngraph::Node> CreateTranspose(std::shared_ptr<ngraph::Node> input,
                                              size_t shape_size,
                                              bool before_conv) {
    std::vector<int32_t> permute_order;
    if (before_conv) {
        permute_order = shape_size == 4 ? std::vector<int32_t>{0, 3, 1, 2} : std::vector<int32_t>{0, 2, 1};
    } else {
        permute_order = shape_size == 4 ? std::vector<int32_t>{0, 2, 3, 1} : std::vector<int32_t>{0, 2, 1};
    }
    return std::make_shared<ngraph::opset1::Transpose>(
        input,
        ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{shape_size}, permute_order));
}

ngraph::Shape GetLayerTransposedOutputShape(std::shared_ptr<ngraph::Node> layer) {
    auto out_shape = layer->get_output_shape(0);
    auto perm = out_shape.size() == 4 ? std::vector<size_t>{0, 2, 3, 1} : std::vector<size_t>{0, 2, 1};
    std::vector<size_t> nhwc_out_shape;
    std::transform(std::begin(perm), std::end(perm), std::back_inserter(nhwc_out_shape), [out_shape](int i) {
        return out_shape[i];
    });
    return nhwc_out_shape;
}

std::shared_ptr<ngraph::Node> CreateConvolution(const ngraph::Output<ngraph::Node>& input,
                                                const ov::element::Type& ngPrc,
                                                const std::vector<size_t>& input_shape,
                                                bool output1D = false,
                                                bool withPool = false,
                                                bool withActivation = false) {
    const size_t num_out_channels = 12;
    auto kernel_shape = GetKernelShape(input_shape, 8, output1D);
    size_t filter_total_size =
        num_out_channels * input_shape.back() *
        std::accumulate(std::begin(kernel_shape), std::end(kernel_shape), 1, std::multiplies<size_t>());
    const std::vector<float> filter_weights = ov::test::utils::generate_float_numbers(filter_total_size, -0.01f, 0.01f);
    const auto shape_size = input_shape.size();
    auto conv = ngraph::builder::makeConvolution(input,
                                                 ngPrc,
                                                 kernel_shape,
                                                 std::vector<size_t>(shape_size - 2, 1),
                                                 std::vector<ptrdiff_t>(shape_size - 2, 0l),
                                                 std::vector<ptrdiff_t>(shape_size - 2, 0l),
                                                 std::vector<size_t>(shape_size - 2, 1),
                                                 ngraph::op::PadType::VALID,
                                                 num_out_channels,
                                                 false,
                                                 filter_weights);

    if (!withPool) {
        return conv;
    }

    auto pool_kernal_shape = GetKernelShape(GetLayerTransposedOutputShape(conv), 2, false);
    auto pool = ngraph::builder::makePooling(conv,
                                             pool_kernal_shape,
                                             std::vector<size_t>(shape_size - 2, 0),
                                             std::vector<size_t>(shape_size - 2, 0),
                                             pool_kernal_shape,
                                             ngraph::op::RoundingType::FLOOR,
                                             ngraph::op::PadType::VALID,
                                             false,
                                             ngraph::helpers::PoolingTypes::MAX);
    return withActivation ? std::make_shared<ngraph::opset3::Relu>(pool) : pool;
}

class RemovePermutationsNHWCToNCHWPassTest : public testing::WithParamInterface<removePermutationsAddParamPassParams>,
                                             public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<removePermutationsAddParamPassParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> inputShape;
        bool output1D;
        bool transpose_to_reshape;
        std::tie(netPrecision, targetDevice, configuration, inputShape, output1D, transpose_to_reshape) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_IS=" << ov::test::utils::vec2str(inputShape);
        result << "_1d_out=" << output1D;
        result << "_transpose2reshape=" << transpose_to_reshape;
        return result.str();
    }

protected:
    void SetUp() override {
        //      Reshape
        //          |
        //      Permute (order: [0, 3, 1, 2])
        //          |
        //      Convolution
        //          |
        //      Permute (order: [0, 2, 3, 1])
        //          |
        //      Reshape
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        bool output1D;
        bool transpose_to_reshape;
        std::tie(netPrecision, targetDevice, configuration, inputShape, output1D, transpose_to_reshape) =
            this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        size_t shape_size = inputShape.size();
        ASSERT_GT(shape_size, 2);
        ASSERT_LT(shape_size, 5);
        size_t in_total_dims_size =
            std::accumulate(std::begin(inputShape), std::end(inputShape), 1, std::multiplies<double>());
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, in_total_dims_size})};

        auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64,
                                                                   ngraph::Shape{shape_size},
                                                                   inputShape);
        auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);

        auto permute1 = CreateTranspose(reshape1, shape_size, true);
        auto conv = CreateConvolution(permute1, ngPrc, inputShape, output1D);
        auto permute2 = CreateTranspose(conv, shape_size, false);

        auto conv_out_size = std::accumulate(std::begin(conv->get_output_shape(0)),
                                             std::end(conv->get_output_shape(0)),
                                             size_t(1),
                                             std::multiplies<size_t>());
        auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64,
                                                                   ngraph::Shape{2},
                                                                   ngraph::Shape{1, conv_out_size});
        auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(permute2, pattern2, false);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(reshape2)};
        function = std::make_shared<ngraph::Function>(results, params, "RemovePermutationsTest");
        if (transpose_to_reshape) {
            ngraph::pass::Manager manager;
            manager.register_pass<ov::pass::TransposeToReshape>();
            manager.run_passes(function);
        }
    }
};

class RemovePermutationsNHWCToNCHWPassNoReshapesTest : public testing::WithParamInterface<removePermutationsPassParams>,
                                                       public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<removePermutationsPassParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> inputShape;
        std::tie(netPrecision, targetDevice, configuration, inputShape) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_IS=" << ov::test::utils::vec2str(inputShape);
        return result.str();
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        std::tie(netPrecision, targetDevice, configuration, inputShape) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        size_t shape_size = inputShape.size();
        ASSERT_GT(shape_size, 2);
        ASSERT_LT(shape_size, 5);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
        auto permute1 = CreateTranspose(params[0], shape_size, true);
        auto conv = CreateConvolution(permute1, ngPrc, inputShape);
        auto permute2 = CreateTranspose(conv, shape_size, false);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(permute2)};

        function = std::make_shared<ngraph::Function>(results, params, "RemovePermutationPassNoReshapes");
    }
};

class RemovePermutationsWithPoolAndActTest : public testing::WithParamInterface<removePermutationsAddParamPassParams>,
                                             public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<removePermutationsAddParamPassParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> inputShape;
        bool withActivation;
        bool transpose_to_reshape;
        std::tie(netPrecision, targetDevice, configuration, inputShape, withActivation, transpose_to_reshape) =
            obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_IS=" << ov::test::utils::vec2str(inputShape);
        result << "_withActivation=" << withActivation;
        result << "_transpose2reshape=" << transpose_to_reshape;
        return result.str();
    }

protected:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        std::vector<float> values = ov::test::utils::generate_float_numbers(blob->size(), -0.2f, 0.2f);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }

    void SetUp() override {
        //      Reshape
        //          |
        //      Permute (order: [0, 3, 1, 2])
        //          |
        //      Convolution
        //          |
        //       Pooling
        //          |
        //        Relu
        //          |
        //      Permute (order: [0, 2, 3, 1])
        //          |
        //      Reshape
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        bool withActivation;
        bool transpose_to_reshape;
        std::tie(netPrecision, targetDevice, configuration, inputShape, withActivation, transpose_to_reshape) =
            this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        size_t shape_size = inputShape.size();
        ASSERT_GT(shape_size, 2);
        ASSERT_LT(shape_size, 5);

        size_t in_total_dims_size =
            std::accumulate(std::begin(inputShape), std::end(inputShape), 1, std::multiplies<double>());
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, in_total_dims_size})};

        auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64,
                                                                   ngraph::Shape{shape_size},
                                                                   inputShape);
        auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);

        auto permute1 = CreateTranspose(reshape1, shape_size, true);
        auto conv = CreateConvolution(permute1, ngPrc, inputShape, false, true, true);
        auto permute2 = CreateTranspose(conv, shape_size, false);

        auto conv_out_size = std::accumulate(std::begin(conv->get_output_shape(0)),
                                             std::end(conv->get_output_shape(0)),
                                             size_t(1),
                                             std::multiplies<size_t>());
        auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64,
                                                                   ngraph::Shape{2},
                                                                   ngraph::Shape{1, conv_out_size});
        auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(permute2, pattern2, false);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(reshape2)};
        function = std::make_shared<ngraph::Function>(results, params, "RemovePermutationsWithPoolAndActTest");

        if (transpose_to_reshape) {
            ngraph::pass::Manager manager;
            manager.register_pass<ov::pass::TransposeToReshape>();
            manager.run_passes(function);
        }
    }
};

class RemovePermutationsWithTwoConvTest : public testing::WithParamInterface<removePermutationsPassParams>,
                                          public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<removePermutationsPassParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> inputShape;
        std::tie(netPrecision, targetDevice, configuration, inputShape) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_IS=" << ov::test::utils::vec2str(inputShape);
        return result.str();
    }

protected:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        std::vector<float> values = ov::test::utils::generate_float_numbers(blob->size(), 0.0f, 0.5f);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }

    void SetUp() override {
        //      Reshape
        //          |
        //      Permute (order: [0, 3, 1, 2])
        //          |
        //      Convolution
        //          |
        //      Convolution
        //          |
        //      Permute (order: [0, 2, 3, 1])
        //          |
        //      Reshape
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        std::tie(netPrecision, targetDevice, configuration, inputShape) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        size_t shape_size = inputShape.size();
        ASSERT_GT(shape_size, 2);
        ASSERT_LT(shape_size, 5);

        size_t in_total_dims_size =
            std::accumulate(std::begin(inputShape), std::end(inputShape), 1, std::multiplies<double>());
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, in_total_dims_size})};

        auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64,
                                                                   ngraph::Shape{shape_size},
                                                                   inputShape);
        auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);

        auto permute1 = CreateTranspose(reshape1, shape_size, true);
        auto conv1 = CreateConvolution(permute1, ngPrc, inputShape);
        auto conv2 = CreateConvolution(conv1, ngPrc, GetLayerTransposedOutputShape(conv1));
        auto permute2 = CreateTranspose(conv2, shape_size, false);

        auto conv_out_size = std::accumulate(std::begin(conv2->get_output_shape(0)),
                                             std::end(conv2->get_output_shape(0)),
                                             size_t(1),
                                             std::multiplies<size_t>());
        auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64,
                                                                   ngraph::Shape{2},
                                                                   ngraph::Shape{1, conv_out_size});
        auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(permute2, pattern2, false);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(reshape2)};
        function = std::make_shared<ngraph::Function>(results, params, "RemovePermutationPass");
    }
};

class RemovePermutationsWithEltwiseTest : public testing::WithParamInterface<removePermutationsPassParams>,
                                          public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<removePermutationsPassParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> inputShape;
        std::tie(netPrecision, targetDevice, configuration, inputShape) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_IS=" << ov::test::utils::vec2str(inputShape);
        return result.str();
    }

protected:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        std::vector<float> values = ov::test::utils::generate_float_numbers(blob->size(), -0.2f, 0.2f);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }

    void SetUp() override {
        //      Reshape                                 Reshape
        //          |                                      |
        //      Permute (order: [0, 3, 1, 2])          Permute (order: [0, 3, 1, 2])
        //          |                                      |
        //      Convolution                            Convolution
        //          |______________________________________|
        //                              |
        //                             Add
        //                              |
        //                  Permute (order: [0, 2, 3, 1])
        //                              |
        //                            Reshape
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        std::tie(netPrecision, targetDevice, configuration, inputShape) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        size_t shape_size = inputShape.size();
        ASSERT_GT(shape_size, 2);
        ASSERT_LT(shape_size, 5);

        size_t in_total_dims_size =
            std::accumulate(std::begin(inputShape), std::end(inputShape), 1, std::multiplies<double>());
        ov::ParameterVector params{
            std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, 2 * in_total_dims_size})};
        auto split = ngraph::builder::makeSplit(params[0], ngPrc, 2, 1);

        auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64,
                                                                   ngraph::Shape{shape_size},
                                                                   inputShape);
        auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(split->output(0), pattern1, false);

        auto permute1 = CreateTranspose(reshape1, shape_size, true);
        auto conv1 = CreateConvolution(permute1, ngPrc, inputShape);

        auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64,
                                                                   ngraph::Shape{shape_size},
                                                                   inputShape);
        auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(split->output(1), pattern2, false);

        auto permute2 = CreateTranspose(reshape2, shape_size, true);
        auto conv2 = CreateConvolution(permute2, ngPrc, inputShape);

        auto add = std::make_shared<ngraph::opset1::Add>(conv1, conv2);
        auto permute3 = CreateTranspose(add, add->get_output_shape(0).size(), false);

        auto conv_out_size = std::accumulate(std::begin(add->get_output_shape(0)),
                                             std::end(add->get_output_shape(0)),
                                             size_t(1),
                                             std::multiplies<size_t>());
        auto pattern3 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64,
                                                                   ngraph::Shape{2},
                                                                   ngraph::Shape{1, conv_out_size});
        auto reshape3 = std::make_shared<ngraph::opset1::Reshape>(permute3, pattern3, false);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(reshape3)};
        function = std::make_shared<ngraph::Function>(results, params, "RemovePermutationPass");
    }
};

class RemoveSharedPermutationTest : public testing::WithParamInterface<removeSharedPermutationPassParams>,
                                    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<removeSharedPermutationPassParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> inputShape;
        size_t splits_num;
        std::tie(netPrecision, targetDevice, configuration, inputShape, splits_num) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_IS=" << ov::test::utils::vec2str(inputShape);
        result << "_splits=" << splits_num;
        return result.str();
    }

protected:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        std::vector<float> values = ov::test::utils::generate_float_numbers(blob->size(), -0.2f, 0.2f);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }

    void SetUp() override {
        //                       Reshape
        //                          |
        //            Permute (order: [0, 3, 1, 2])
        //                          |
        //          ______________Split____________________
        //          |                                      |
        //      Convolution                            Convolution
        //          |                                      |
        //  Permute (order: [0, 2, 3, 1])      Permute (order: [0, 2, 3, 1])
        //          |                                      |
        //       Reshape                                 Reshape
        //          |______________________________________|
        //                         Concat
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        size_t splits_num;
        std::tie(netPrecision, targetDevice, configuration, inputShape, splits_num) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        size_t shape_size = inputShape.size();
        ASSERT_GT(shape_size, 2);
        ASSERT_LT(shape_size, 5);

        size_t in_total_dims_size =
            std::accumulate(std::begin(inputShape), std::end(inputShape), 1, std::multiplies<double>());
        ov::ParameterVector params{
            std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, splits_num * in_total_dims_size})};

        auto multipleInputShape = inputShape;
        size_t mul_dim = inputShape.size() == 4 && inputShape[1] > 1 ? 1 : (inputShape.size() - 2);
        multipleInputShape[mul_dim] *= splits_num;
        auto pattern = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64,
                                                                  ngraph::Shape{multipleInputShape.size()},
                                                                  multipleInputShape);
        auto reshape = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern, false);
        auto permute = CreateTranspose(reshape, shape_size, true);
        auto split = ngraph::builder::makeSplit(
            permute,
            ngPrc,
            splits_num,
            inputShape.size() == 4 && inputShape[1] > 1 ? inputShape.size() - 2 : inputShape.size() - 1);

        auto conv1 = CreateConvolution(split->output(0), ngPrc, inputShape);
        auto permute1 = CreateTranspose(conv1, conv1->get_output_shape(0).size(), false);
        auto conv1_out_size = std::accumulate(std::begin(conv1->get_output_shape(0)),
                                              std::end(conv1->get_output_shape(0)),
                                              size_t(1),
                                              std::multiplies<size_t>());
        auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64,
                                                                   ngraph::Shape{2},
                                                                   ngraph::Shape{1, conv1_out_size});
        auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(permute1, pattern1, false);

        auto conv2 = CreateConvolution(split->output(1), ngPrc, inputShape);
        auto permute2 = CreateTranspose(conv2, conv2->get_output_shape(0).size(), false);
        auto conv2_out_size = std::accumulate(std::begin(conv2->get_output_shape(0)),
                                              std::end(conv2->get_output_shape(0)),
                                              size_t(1),
                                              std::multiplies<size_t>());
        auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64,
                                                                   ngraph::Shape{2},
                                                                   ngraph::Shape{1, conv2_out_size});
        auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(permute2, pattern2, false);

        auto concat = ngraph::builder::makeConcat({reshape1, reshape2}, 1);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(concat)};
        function = std::make_shared<ngraph::Function>(results, params, "RemoveSharedPermutationTest");
    }
};

TEST_P(RemovePermutationsNHWCToNCHWPassTest, CompareWithRefImpl) {
    Run();
};

TEST_P(RemovePermutationsNHWCToNCHWPassNoReshapesTest, CompareWithRefImpl) {
    Run();
};

TEST_P(RemovePermutationsWithPoolAndActTest, CompareWithRefImpl) {
    Run();
};

TEST_P(RemovePermutationsWithTwoConvTest, CompareWithRefImpl) {
    Run();
};

TEST_P(RemovePermutationsWithEltwiseTest, CompareWithRefImpl) {
    Run();
};

TEST_P(RemoveSharedPermutationTest, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "327.67"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

const std::vector<std::vector<size_t>> inputShapes{
    {1, 1, 168, 1}, {1, 1, 168, 2}, {1, 1, 168, 4}, {1, 1, 32, 1},  {1, 1, 32, 2}, {1, 1, 32, 8},
    {1, 1, 32, 9},  {1, 168, 1, 1}, {1, 168, 1, 2}, {1, 168, 1, 4}, {1, 32, 1, 1}, {1, 32, 1, 2},
    {1, 32, 1, 8},  {1, 32, 1, 9},  {1, 16, 8, 1},  {1, 168, 1},    {1, 168, 2},   {1, 168, 4},
    {1, 32, 1},     {1, 32, 2},     {1, 32, 8},     {1, 32, 9}};

const std::vector<std::vector<size_t>> inputShapesSplit{
    {1, 1, 160, 1}, {1, 1, 160, 2}, {1, 1, 168, 4}, {1, 1, 32, 1},  {1, 1, 32, 2}, {1, 1, 32, 8},
    {1, 1, 32, 9},  {1, 160, 1, 1}, {1, 160, 1, 2}, {1, 168, 1, 4}, {1, 32, 1, 1}, {1, 32, 1, 2},
    {1, 32, 1, 8},  {1, 32, 1, 9},  {1, 16, 8, 1},  {1, 168, 1},    {1, 168, 2},   {1, 168, 4},
    {1, 32, 1},     {1, 32, 2},     {1, 32, 8},     {1, 32, 9}};

const std::vector<size_t> splitsNum = {2, 4, 8};

INSTANTIATE_TEST_SUITE_P(
    smoke_PermutationPass,
    RemovePermutationsNHWCToNCHWPassTest,
    ::testing::Combine(::testing::ValuesIn(netPrecisions),
                       ::testing::Values(ov::test::utils::DEVICE_GNA),
                       ::testing::ValuesIn(configs),
                       ::testing::ValuesIn(inputShapes),
                       ::testing::ValuesIn(std::vector<bool>{false, true}),   // with 1d output of convolution
                       ::testing::ValuesIn(std::vector<bool>{false, true})),  // transpose to reshape
    RemovePermutationsNHWCToNCHWPassTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PermutationPass,
                         RemovePermutationsNHWCToNCHWPassNoReshapesTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(inputShapes)),
                         RemovePermutationsNHWCToNCHWPassNoReshapesTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PermutationPass,
                         RemovePermutationsWithPoolAndActTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(std::vector<bool>{false, true}),  // with activation
                                            ::testing::ValuesIn(std::vector<bool>{false,
                                                                                  true})),  // transpose to reshape
                         RemovePermutationsWithPoolAndActTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PermutationPass,
                         RemovePermutationsWithTwoConvTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(inputShapes)),
                         RemovePermutationsWithTwoConvTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PermutationPass,
                         RemovePermutationsWithEltwiseTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(inputShapes)),
                         RemovePermutationsWithEltwiseTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PermutationPass,
                         RemoveSharedPermutationTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(inputShapesSplit),
                                            ::testing::ValuesIn(splitsNum)),
                         RemoveSharedPermutationTest::getTestCaseName);

}  // namespace LayerTestsDefinitions
