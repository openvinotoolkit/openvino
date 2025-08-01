// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/op/constant.hpp"

#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

using namespace ::testing;

namespace ov {
namespace test {
namespace intel_gpu {

class CustomAddOp : public ov::op::Op {
private:
    float m_alpha;
    float m_beta;

public:
    OPENVINO_OP("CustomAddOp", "gpu_opset");

    CustomAddOp() = default;

    CustomAddOp(const ov::Output<ov::Node>& input, float alpha, float beta) : Op({input}), m_alpha(alpha), m_beta(beta) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("alpha", m_alpha);
        visitor.on_attribute("beta", m_beta);
        return true;
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");
        return std::make_shared<CustomAddOp>(new_args[0], m_alpha, m_beta);
    }

    bool has_evaluate() const override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        auto in = inputs[0];
        auto out = outputs[0];
        out.set_shape(in.get_shape());
        for (size_t i = 0; i < out.get_size(); i++) {
            out.data<float>()[i] = in.data<float>()[i] * m_alpha + m_beta;
        }
        return true;
    }
};

using CustomOpDynamicTestParams = std::tuple<std::vector<ov::Shape>,            // input shape
                                             std::vector<std::vector<float>>>;  // input data
class CustomOpDynamic : public ov::test::TestsCommon, public testing::WithParamInterface<CustomOpDynamicTestParams> {
    void SetUp() override {
        generate_config_files();
    };

    void TearDown() override {
        ov::test::utils::removeFile(config_cl);
        ov::test::utils::removeFile(config_xml);
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<CustomOpDynamicTestParams>& obj) {
        std::vector<ov::Shape> input_shapes;
        std::vector<std::vector<float>> input_datas;
        std::tie(input_shapes, input_datas) = obj.param;

        std::ostringstream result;
        result << "input_shape=";
        for (auto shape : input_shapes) {
            result << shape;
        }
        return result.str();
    }

    static const size_t dim1 = 1;
    void run() {
        std::vector<ov::Shape> input_shapes;
        std::vector<std::vector<float>> input_datas;
        std::tie(input_shapes, input_datas) = GetParam();
        ASSERT_TRUE(input_shapes.size() == input_datas.size());

        ov::Core core;
        float alpha = 1.0, beta = 0.1;
        auto model = generate_model_with_custom_add_op(alpha, beta, ov::PartialShape{-1, dim1, -1});

        ov::AnyMap config = {ov::hint::inference_precision(ov::element::f32), {"CONFIG_FILE", config_xml}};
        auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config);

        auto runtime_graph = compiled_model.get_runtime_model();
        auto ops = runtime_graph->get_ordered_ops();

        bool found_custom_op = false;
        for (auto op : ops) {
            if (op->get_rt_info()[ov::exec_model_info::LAYER_TYPE].as<std::string>() == "CustomGPUPrimitive") {
                found_custom_op = true;
                break;
            }
        }
        ASSERT_TRUE(found_custom_op);

        auto ireq = compiled_model.create_infer_request();
        for (size_t i = 0; i < input_datas.size(); i++) {
            auto input = ov::Tensor({ov::element::f32}, input_shapes[i], input_datas[i].data());
            ireq.set_input_tensor(0, input);
            ireq.infer();
            auto output = ireq.get_output_tensor(0);
            std::vector<float> actual(output.data<float>(), output.data<float>() + output.get_size());

            ASSERT_EQ(output.get_element_type(), element::f32);

            float* inp_data = input.data<float>();
            for (size_t i = 0; i < output.get_size(); i++) {
                ASSERT_FLOAT_EQ(actual[i], inp_data[i] * alpha + beta);
            }
        }
    }

private:
    std::string config_cl;
    std::string config_xml;

    void generate_config_files() {
        config_cl = ov::test::utils::generateTestFilePrefix() + "_custom_op_dynamic.cl";
        config_xml = ov::test::utils::generateTestFilePrefix() + "_custom_op_dynamic.xml";

        std::string content_cl = R"(
        __kernel void custom_add_kernel(
            __global const INPUT0_TYPE* inp0,
            __global OUTPUT0_TYPE* outp) {
            const uint b = (uint)get_global_id(0);
            const uint f = (uint)get_global_id(1);
            const uint y = (uint)get_global_id(2);
            #if INPUT0_DIMS_SIZE == 4
                const uint x = 0;
            #endif
    
            const unsigned src_index = b*INPUT0_DIMS[1]*INPUT0_DIMS[2]*INPUT0_DIMS[3] + f*INPUT0_DIMS[2]*INPUT0_DIMS[3] + y*INPUT0_DIMS[3] + x;
            const unsigned dst_index = src_index;
    
            outp[dst_index] = inp0[src_index] * alpha + beta;
        })";

        std::string content_xml = R"(
            <CustomLayer name="CustomAddOp" type="SimpleGPU" version="1">
                <Kernel entry="custom_add_kernel">
                    <Source filename=")" + config_cl + R"("/>
                    <Define name="alpha" type="float" param="alpha" default="1.0"/>
                    <Define name="beta" type="float" param="beta" default="0.1"/>
                </Kernel>
                <Buffers>
                    <Tensor arg-index="0" type="input" port-index="0" format="BFYX"/>
                    <Tensor arg-index="1" type="output" port-index="0" format="BFYX"/>
                </Buffers>
                <CompilerOptions options="-cl-mad-enable"/>
                <WorkSizes global="B,F,Y"/>
            </CustomLayer>)";

        ov::test::utils::createFile(config_cl, content_cl);
        ov::test::utils::createFile(config_xml, content_xml);
    }

    std::shared_ptr<ov::Model> generate_model_with_custom_add_op(float alpha, float beta, ov::PartialShape inp_shape) {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inp_shape);
        auto op = std::make_shared<CustomAddOp>(input, alpha, beta);
        auto result = std::make_shared<ov::op::v0::Result>(op);
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "model_with_custom_op_dynamic");
    }
};

TEST_P(CustomOpDynamic, Accuracy) {
    run();
}

const std::vector<ov::Shape> input_shapes{{1, CustomOpDynamic::dim1, 2}, {2, CustomOpDynamic::dim1, 3}};
const std::vector<std::vector<float>> input_datas{{0.2, 0.4}, {0.2, 0.4, 0.3, 0.5, 0.7, 0.9}};

INSTANTIATE_TEST_SUITE_P(smoke_GPU_Accuracy, CustomOpDynamic,
    ::testing::Combine(::testing::Values(input_shapes),
                       ::testing::Values(input_datas)),
    CustomOpDynamic::getTestCaseName);

} // namespace intel_gpu
} // namespace test
} // namespace ov
