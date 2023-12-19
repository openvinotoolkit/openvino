// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base/behavior_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace behavior {

using PreprocessingPrecisionConvertParams =
    std::tuple<ov::element::Type,  // Input type
               std::size_t,        // channels number
               bool,               // Use normal (i.e. set_tensor() or unusal i.e. get_tensor()) input method
               std::string,        // Device name
               ov::AnyMap          // Config
               >;

class PreprocessingPrecisionConvertTests : public testing::WithParamInterface<PreprocessingPrecisionConvertParams>,
                                           virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PreprocessingPrecisionConvertParams> obj) {
        ov::element::Type inPrc;
        bool useSetInput;
        std::size_t channels;
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(inPrc, channels, useSetInput, targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "inPRC=" << inPrc.get_type_name() << "_";
        result << channels << "Ch"
               << "_";
        result << (useSetInput ? "set_tensor" : "get_tensor") << "_";
        result << "targetDevice=" << targetDevice;
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_" << configItem.second.as<std::string>() << "_";
            }
        }
        return result.str();
    }

    void infer() override {
        inferRequest = compiledModel.create_infer_request();

        for (const auto& input : inputs) {
            if (!use_set_input) {
                auto tmp_tensor = inferRequest.get_tensor(input.first);
                std::memcpy(static_cast<char*>(tmp_tensor.data()),
                            static_cast<char*>(input.second.data()),
                            tmp_tensor.get_byte_size());
            } else {
                inferRequest.set_tensor(input.first, input.second);
            }
        }
        inferRequest.infer();
    }

    void SetUp() override {
        // This test:
        // - Strive to test the plugin internal preprocessing (precision conversion) only.
        //   Thus (logically) no-op graph is used.
        // - Reference code mimic the preprocessing via extra ngraph Convert operation.
        // - Create/uses two (different) graphs here : one to feed the the plugin and one calculate the reference
        // result.

        std::tie(inType, channels, use_set_input, targetDevice, configuration) = this->GetParam();
        outType = inType;

        bool specialZero = true;

        std::vector<size_t> inputShape(channels, 4);

        input_shapes = {inputShape};

        auto make_ngraph = [&](bool with_extra_conv) {
            auto in_prec = with_extra_conv ? inType : decltype(inType)(ov::element::f32);
            ov::ParameterVector paramsIn{std::make_shared<ov::op::v0::Parameter>(in_prec, ov::Shape(inputShape))};

            auto toF32 = std::make_shared<ov::op::v0::Convert>(paramsIn[0], ov::element::Type_t::f32);

            auto constNode = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64,
                                                                    ov::Shape{inputShape.size()},
                                                                    inputShape);
            std::shared_ptr<ov::Node> reshape_input = with_extra_conv ? toF32->shared_from_this() : paramsIn[0];
            auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(
                std::make_shared<ov::op::v1::Reshape>(reshape_input, constNode, specialZero));
            ov::ResultVector results{std::make_shared<ov::op::v0::Result>(reshape)};
            return std::make_shared<ov::Model>(results, paramsIn, "Reshape");
        };

        function = make_ngraph(false);
    }

    void validate() override {
        // w/a: copy of original function is required to provide correct op coverage report (overflow of convert counter
        // issue)
        auto copyOriginalFunction = function;
        // force the reference implementation to use graph with extra Convert operation
        SubgraphBaseTest::validate();
        function = copyOriginalFunction;
    }

    void run() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        functionRefs = ov::clone_model(*function);
        try {
            compile_model();
            generate_inputs(input_shapes);
            infer();
            validate();
        } catch (const std::runtime_error& re) {
            GTEST_FATAL_FAILURE_(re.what());
        } catch (const std::exception& ex) {
            GTEST_FATAL_FAILURE_(ex.what());
        } catch (...) {
            GTEST_FATAL_FAILURE_("Unknown failure occurred.");
        }
    }

public:
    bool use_set_input = true;
    std::size_t channels = 0;
    std::vector<ov::Shape> input_shapes;
};

TEST_P(PreprocessingPrecisionConvertTests, InternalPluginPrecisionConvert) {
    run();
}
}  // namespace behavior
}  // namespace test
}  // namespace ov
