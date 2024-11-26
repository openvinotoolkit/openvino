// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/node_builders/activation.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "transformations/utils/utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/subgraph_builders/split_multi_conv_concat.hpp"
#include "common_test_utils/subgraph_builders/read_concat_split_assign.hpp"

namespace {
typedef std::tuple<
        ov::element::Type,   // Input/Output type
        ov::Shape,           // Input Shape
        std::string> newtworkParams;

class InferRequestIOPrecision : public testing::WithParamInterface<newtworkParams>,
                                virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<newtworkParams> &obj);

protected:
    void SetUp() override;
};

std::string InferRequestIOPrecision::getTestCaseName(const testing::TestParamInfo<newtworkParams> &obj) {
    ov::element::Type model_type;
    ov::Shape shape;
    std::string targetDevice;
    std::tie(model_type, shape, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << "netPRC=" << model_type.get_type_name() << separator;
    result << "trgDev=" << targetDevice;
    return result.str();
}

void InferRequestIOPrecision::SetUp() {
    ov::element::Type model_type;
    ov::Shape shape;
    std::tie(model_type, shape, targetDevice) = GetParam();

    float clamp_min = model_type.is_signed() ? -5.f : 0.0f;
    float clamp_max = 5.0f;

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(shape))};
    params[0]->set_friendly_name("Input");

    auto activation = ov::test::utils::make_activation(params[0],
                                                       model_type,
                                                       ov::test::utils::ActivationTypes::Clamp,
                                                       {},
                                                       {clamp_min, clamp_max});

    function = std::make_shared<ov::Model>(ov::NodeVector{activation}, params);
}

TEST_P(InferRequestIOPrecision, Inference) {
    run();
}

const std::vector<ov::element::Type> input_types = {
        ov::element::i16,
        ov::element::u16,
        ov::element::f32,
        ov::element::f16,
        ov::element::u8,
        ov::element::i8,
        ov::element::i32,
        ov::element::u32,
        ov::element::u64,
        ov::element::i64,
        // Interpreter backend doesn't implement evaluate method for OP
        // ov::element::f64,
};

INSTANTIATE_TEST_SUITE_P(smoke_GPU_BehaviorTests, InferRequestIOPrecision,
                         ::testing::Combine(
                                 ::testing::ValuesIn(input_types),
                                 ::testing::Values(ov::Shape{1, 50}),
                                 ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         InferRequestIOPrecision::getTestCaseName);

TEST(TensorTest, smoke_canSetShapeForPreallocatedTensor) {
    auto core = ov::Core();
    using namespace ov::preprocess;
    auto p = PrePostProcessor(ov::test::utils::make_split_multi_conv_concat());
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);

    auto function = p.build();
    auto exec_net = core.compile_model(function, ov::test::utils::DEVICE_GPU);
    auto inf_req = exec_net.create_infer_request();

    // Check set_shape call for pre-allocated input/output tensors
    auto input_tensor = inf_req.get_input_tensor(0);
    OV_ASSERT_NO_THROW(input_tensor.set_shape({1, 4, 20, 20}));
    OV_ASSERT_NO_THROW(input_tensor.set_shape({1, 3, 20, 20}));
    OV_ASSERT_NO_THROW(input_tensor.set_shape({2, 3, 20, 20}));
    auto output_tensor = inf_req.get_output_tensor(0);
    OV_ASSERT_NO_THROW(output_tensor.set_shape({1, 10, 12, 12}));
    OV_ASSERT_NO_THROW(output_tensor.set_shape({1, 10, 10, 10}));
    OV_ASSERT_NO_THROW(output_tensor.set_shape({2, 10, 20, 20}));
}

TEST(TensorTest, smoke_canSetScalarTensor) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f64, ov::Shape{})};
    params.front()->set_friendly_name("Scalar_1");
    params.front()->output(0).get_tensor().set_names({"scalar1"});

    std::vector<size_t> const_shape = {1};
    auto const1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, const_shape);
    const1->set_friendly_name("Const_1");
    const1->output(0).get_tensor().set_names({"const1"});
    const1->fill_data(ov::element::i64, 0);

    auto unsqueeze1 = std::make_shared<ov::op::v0::Unsqueeze>(params.front(), const1);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(unsqueeze1)};
    auto model = std::make_shared<ov::Model>(results, params);

    auto core = ov::Core();
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto request = compiled_model.create_infer_request();
    double real_data = 1.0;
    ov::Tensor input_data(ov::element::f64, {}, &real_data);
    request.set_tensor("scalar1", input_data);
    OV_ASSERT_NO_THROW(request.infer());
}

TEST(TensorTest, smoke_canSetTensorForDynamicInput) {
    auto core = ov::Core();
    using namespace ov::preprocess;
    auto p = PrePostProcessor(ov::test::utils::make_split_multi_conv_concat());
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);

    auto function = p.build();
    std::map<size_t, ov::PartialShape> shapes = { {0, ov::PartialShape{-1, -1, -1, -1}} };
    function->reshape(shapes);
    auto exec_net = core.compile_model(function, ov::test::utils::DEVICE_GPU);
    auto inf_req = exec_net.create_infer_request();

    ov::Tensor t1(ov::element::i8, {1, 4, 20, 20});
    ov::Tensor t2(ov::element::i8, {1, 4, 30, 30});
    ov::Tensor t3(ov::element::i8, {1, 4, 40, 40});

    // Check set_shape call for pre-allocated input/output tensors
    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t1));
    OV_ASSERT_NO_THROW(inf_req.infer());

    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t2));
    OV_ASSERT_NO_THROW(inf_req.infer());

    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t3));
    OV_ASSERT_NO_THROW(inf_req.infer());

    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t3));
    OV_ASSERT_NO_THROW(inf_req.infer());

    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t1));
    OV_ASSERT_NO_THROW(inf_req.infer());

    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t2));
    OV_ASSERT_NO_THROW(inf_req.infer());
}

TEST(TensorTest, smoke_canSetTensorForDynamicOutput) {
    auto core = ov::Core();
    using namespace ov::preprocess;
    auto p = PrePostProcessor(ov::test::utils::make_split_multi_conv_concat());
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);

    auto function = p.build();
    std::map<size_t, ov::PartialShape> shapes = { {0, ov::PartialShape{-1, -1, -1, -1}} };
    function->reshape(shapes);
    auto exec_net = core.compile_model(function, ov::test::utils::DEVICE_GPU);
    auto inf_req = exec_net.create_infer_request();

    ov::Tensor t1(ov::element::i8, {1, 4, 20, 20});
    auto out_tensor = inf_req.get_output_tensor();
    ov::Tensor t2(out_tensor.get_element_type(), out_tensor.get_shape());
    ASSERT_EQ(t2.get_byte_size(), 0);
    // Check set_shape call for pre-allocated input/output tensors
    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t1));
    OV_ASSERT_NO_THROW(inf_req.set_output_tensor(t2));
    OV_ASSERT_NO_THROW(inf_req.infer());
    ASSERT_NE(t2.get_byte_size(), 0);
}

TEST(TensorTest, smoke_canReallocateDeviceInputForHostTensor) {
    auto ov = ov::Core();
    using namespace ov::preprocess;
    auto p = PrePostProcessor(ov::test::utils::make_split_multi_conv_concat());
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);
    auto function = p.build();

    auto compiled_model = ov.compile_model(function, ov::test::utils::DEVICE_GPU);
    auto inf_req = compiled_model.create_infer_request();

    auto input = function->input();
    ov::Tensor host_tensor(input.get_element_type(), input.get_shape());

    // Infer with pre-allocated input tensor
    OV_ASSERT_NO_THROW(inf_req.infer());

    // Infer with host_tensor
    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(host_tensor));
    OV_ASSERT_NO_THROW(inf_req.infer());
}

TEST(VariablesTest, smoke_canSetStateTensor) {
    auto ov = ov::Core();
    const ov::Shape virable_shape = {1, 3, 2, 4};
    const ov::Shape input_shape = {1, 3, 2, 4};
    const ov::element::Type et = ov::element::f16;
    auto model = ov::test::utils::make_read_concat_split_assign(input_shape, et);
    auto compiled_model = ov.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto request = compiled_model.create_infer_request();

    ov::Tensor variable_tensor(et, virable_shape);
    ov::Tensor input_tensor(et, input_shape);

    auto variables = request.query_state();
    ASSERT_EQ(variables.size(), 1);
    auto variable = variables.front();
    ASSERT_EQ(variable.get_name(), "v0");
    auto default_state_tensor = variable.get_state();
    ASSERT_EQ(default_state_tensor.get_shape(), virable_shape);

    OV_ASSERT_NO_THROW(request.infer());
}

TEST(VariablesTest, smoke_set_get_state_with_convert) {
    auto build_model = [](ov::element::Type type, const ov::PartialShape& shape) {
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        const ov::op::util::VariableInfo variable_info { shape, type, "v0" };
        auto variable = std::make_shared<ov::op::util::Variable>(variable_info);
        auto read_value = std::make_shared<ov::op::v6::ReadValue>(param, variable);
        auto add = std::make_shared<ov::op::v1::Add>(read_value, param);
        auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
        auto res = std::make_shared<ov::op::v0::Result>(add);
        return std::make_shared<ov::Model>(ov::ResultVector { res }, ov::SinkVector { assign }, ov::ParameterVector{param}, "StateTestModel");
    };

    auto ov = ov::Core();
    const ov::Shape virable_shape = {1, 3, 2, 4};
    const ov::Shape input_shape = {1, 3, 2, 4};
    const ov::element::Type et = ov::element::f32;
    auto model = build_model(et, input_shape);
    auto compiled_model = ov.compile_model(model, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f16));
    auto request = compiled_model.create_infer_request();

    auto variables = request.query_state();
    ASSERT_EQ(variables.size(), 1);
    auto variable = variables.front();
    ASSERT_EQ(variable.get_name(), "v0");
    auto state_tensor = variable.get_state();
    ASSERT_EQ(state_tensor.get_shape(), virable_shape);
    ASSERT_EQ(state_tensor.get_element_type(), et);

    auto tensor_to_set = ov::test::utils::create_and_fill_tensor(et, state_tensor.get_shape());
    variable.set_state(tensor_to_set);
    state_tensor = variable.get_state();

    ov::test::utils::compare(tensor_to_set, state_tensor, 1e-5f, 1e-5f);
}

TEST(VariablesTest, smoke_padded_tensor_set_get_state_with_convert) {
    auto build_model = [](ov::element::Type type, const ov::PartialShape& shape) {
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        const ov::op::util::VariableInfo variable_info { shape, type, "v0" };
        auto variable = std::make_shared<ov::op::util::Variable>(variable_info);
        auto read_value = std::make_shared<ov::op::v6::ReadValue>(param, variable);
        auto add = std::make_shared<ov::op::v1::Add>(read_value, param);
        auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
        auto res = std::make_shared<ov::op::v0::Result>(add);
        return std::make_shared<ov::Model>(ov::ResultVector { res }, ov::SinkVector { assign }, ov::ParameterVector{param}, "StateTestModel");
    };

    auto ov = ov::Core();
    const ov::Shape virable_shape_padded = {1, 3, 4, 4};
    const ov::Shape virable_shape = {1, 3, 2, 4};
    const ov::Shape input_shape = {1, 3, 2, 4};
    const ov::element::Type et = ov::element::f32;
    auto model = build_model(et, input_shape);
    auto compiled_model = ov.compile_model(model, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f16));
    auto request = compiled_model.create_infer_request();

    auto variables = request.query_state();
    ASSERT_EQ(variables.size(), 1);
    auto variable = variables.front();
    ASSERT_EQ(variable.get_name(), "v0");
    auto state_tensor = variable.get_state();
    ASSERT_EQ(state_tensor.get_shape(), virable_shape);
    ASSERT_EQ(state_tensor.get_element_type(), et);

    auto tensor_to_set_padded = ov::test::utils::create_and_fill_tensor(et, virable_shape_padded);

    // trim original tensor
    auto tensor_to_set =
        ov::Tensor(tensor_to_set_padded, ov::Coordinate{0, 0, 0, 0}, ov::Coordinate(virable_shape));

    variable.set_state(tensor_to_set);
    state_tensor = variable.get_state();

    auto res_tensor_ptr = static_cast<float*>(state_tensor.data());
    auto ref_tensor_ptr = static_cast<float*>(tensor_to_set.data());
    auto ref_stride = tensor_to_set.get_strides();
    auto res_stride = state_tensor.get_strides();
    for (size_t i = 0; i < ref_stride.size(); ++i) {
        ref_stride[i] /= (tensor_to_set.get_element_type().bitwidth()/8);
        res_stride[i] /= (state_tensor.get_element_type().bitwidth()/8);
    }
    // ref stride: [48, 16, 4, 1]
    // res stride: [24, 8, 4, 1]
    // compare actual tensor w/o pad
    for (size_t b = 0; b < virable_shape[0]; ++b) {
        for (size_t f = 0; f < virable_shape[1]; ++f) {
            for (size_t y = 0; y < virable_shape[2]; ++y) {
                for (size_t x = 0; x < virable_shape[3]; ++x) {
                    auto ref_idx = b * ref_stride[0] + f * ref_stride[1] + y * ref_stride[2] + x * ref_stride[3];
                    auto res_idx = b * res_stride[0] + f * res_stride[1] + y * res_stride[2] + x * res_stride[3];
                    ASSERT_EQ(res_tensor_ptr[res_idx], ref_tensor_ptr[ref_idx]);
                }
            }
        }
    }
}

TEST(TensorTest, smoke_outputTensorShapesForDynamicInput) {
    auto core = ov::Core();
    using namespace ov::preprocess;
    auto p = PrePostProcessor(ov::test::utils::make_split_multi_conv_concat());
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);

    auto function = p.build();
    std::map<size_t, ov::PartialShape> shapes = { {0, ov::PartialShape{-1, -1, -1, -1}} };
    function->reshape(shapes);
    auto exec_net = core.compile_model(function, ov::test::utils::DEVICE_GPU);
    auto inf_req = exec_net.create_infer_request();

    ov::Tensor t1(ov::element::i8, {1, 4, 20, 40});
    ov::Tensor t2(ov::element::i8, {1, 4, 40, 20});
    ov::Tensor t3(ov::element::i8, {1, 4, 20, 40});
    const ov::Shape output1_shape = {1, 10, 12, 32};
    const ov::Shape output2_shape = {1, 10, 32, 12};
    const ov::Shape output3_shape = {1, 10, 12, 32};

    // Check output shape of output tensor is correct
    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t1));
    OV_ASSERT_NO_THROW(inf_req.infer());
    ASSERT_EQ(inf_req.get_output_tensor().get_shape(), output1_shape);

    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t2));
    OV_ASSERT_NO_THROW(inf_req.infer());
    ASSERT_EQ(inf_req.get_output_tensor().get_shape(), output2_shape);

    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t3));
    OV_ASSERT_NO_THROW(inf_req.infer());
    ASSERT_EQ(inf_req.get_output_tensor().get_shape(), output3_shape);
}

TEST(TensorTest, smoke_canShareTensorIfModelsFromDifferentCores) {
    auto core1 = ov::Core();
    auto core2 = ov::Core();

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4, 8});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    auto result = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    auto compiled_model1 = core1.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto compiled_model2 = core2.compile_model(model, ov::test::utils::DEVICE_GPU);

    auto request1 = compiled_model1.create_infer_request();
    auto request2 = compiled_model2.create_infer_request();

    request2.set_input_tensor(request1.get_output_tensor());
    request2.set_output_tensor(request1.get_input_tensor());

    OV_ASSERT_NO_THROW(request1.infer());
    OV_ASSERT_NO_THROW(request2.infer());
}
} // namespace
