// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/convolution.hpp"
#include "openvino/openvino.hpp"
#include "utils/convolution_params.hpp"
#include "utils/cpu_test_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

class EdgeWithSameNameInTwoModels : public ::testing::Test, public CPUTestsBase {};

TEST_F(EdgeWithSameNameInTwoModels, smoke_CompareWithRef) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    const std::string targetDevice = ov::test::utils::DEVICE_CPU;
    const ov::element::Type type(ov::element::Type_t::f32);
    const std::string convName("conv_name"), weightName("weight_name");
    const std::vector<size_t> kernel{3, 3};
    const std::vector<size_t> strides{1, 1};
    const std::vector<ptrdiff_t> padsBegin{0, 0};
    const std::vector<ptrdiff_t> padsEnd{0, 0};
    const std::vector<size_t> dilations{1, 1};
    const ov::op::PadType autoPad(ov::op::PadType::EXPLICIT);

    if (ov::with_cpu_x86_avx512f()) {
        std::tie(inFmts, outFmts, priority, selectedType) = conv_avx512_2D;
    } else if (ov::with_cpu_x86_avx2()) {
        std::tie(inFmts, outFmts, priority, selectedType) = conv_avx2_2D;
    } else if (ov::with_cpu_x86_sse42()) {
        std::tie(inFmts, outFmts, priority, selectedType) = conv_sse42_2D;
    }

    // first model
    const std::vector<ov::Shape> shapes1{{1, 16, 720, 1280}};
    ov::ParameterVector params1;
    for (auto&& shape : shapes1) {
        params1.push_back(std::make_shared<ov::op::v0::Parameter>(type, shape));
    }
    const size_t convOutCh1 = 32;
    auto conv1 = ov::test::utils::make_convolution(params1.front(),
                                                   type,
                                                   kernel,
                                                   strides,
                                                   padsBegin,
                                                   padsEnd,
                                                   dilations,
                                                   autoPad,
                                                   convOutCh1);
    conv1->set_friendly_name(convName);
    conv1->get_input_node_shared_ptr(1)->set_friendly_name(weightName);
    auto model1 = makeNgraphFunction(type, params1, conv1, "Model1");

    // second model
    const std::vector<ov::Shape> shapes2{{1, 32, 24, 24}};
    ov::ParameterVector params2;
    for (auto&& shape : shapes2) {
        params2.push_back(std::make_shared<ov::op::v0::Parameter>(type, shape));
    }
    const size_t convOutCh2 = 16;
    auto conv2 = ov::test::utils::make_convolution(params2.front(),
                                                   type,
                                                   kernel,
                                                   strides,
                                                   padsBegin,
                                                   padsEnd,
                                                   dilations,
                                                   autoPad,
                                                   convOutCh2);
    conv2->set_friendly_name(convName);
    conv2->get_input_node_shared_ptr(1)->set_friendly_name(weightName);
    auto model2 = makeNgraphFunction(type, params2, conv2, "Model2");

    // model compilation
    std::map<std::string, ov::AnyMap> config;
    auto& device_config = config[targetDevice];
    device_config[ov::num_streams.name()] = 4;

    ov::Core core;
    for (auto&& item : config) {
        core.set_property(item.first, item.second);
    }

    auto compiledModel1 = core.compile_model(model1, targetDevice);
    auto compiledModel2 = core.compile_model(model2, targetDevice);

    auto inferReq1 = compiledModel1.create_infer_request();
    auto inferReq2 = compiledModel2.create_infer_request();

    inferReq1.infer();
    inferReq2.infer();
}

}  // namespace test
}  // namespace ov
