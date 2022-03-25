// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"

#include "transformations/hash.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/util/variable.hpp"

using namespace testing;
using namespace std;
using namespace ov;

class TestHashAttribute : public RuntimeAttribute {
public:
    TestHashAttribute() = default;

    bool visit_attributes(AttributeVisitor &visitor) override {
        visitor.on_attribute(string_name, s);
        visitor.on_attribute("int64", i64);
        visitor.on_attribute("double", d);
        visitor.on_attribute("bool", b);
        visitor.on_attribute("vector_int64", vi64);
        visitor.on_attribute("vector_uint64", vu64);
        visitor.on_attribute("vector_int", vi);
        visitor.on_attribute("vector_float", vf);
        visitor.on_attribute("vector_string", vs);
        visitor.on_attribute("set_strings", string_set);
        if (rt_model) {
            visitor.on_attribute("model", rt_model);
        }
        return true;
    }

    std::set<std::string> string_set = {"1", "2", "3"};
    std::string s = "1";
    std::string string_name = "string";
    int64_t i64 = 1;
    std::vector<int64_t> vi64 = {1, 2, 3};
    std::vector<uint64_t> vu64 = {1, 2, 3};
    std::vector<float> vf = {1.f, 2.f, 3.f};
    std::vector<std::string> vs = {"1", "2", "3"};
    std::vector<int> vi = {1, 2, 3};
    bool b = true;
    double d = 3.14;
    std::shared_ptr<ov::Model> rt_model = {}; // not supported
};

template <typename T>
void check_rt_one(const T& value1, const T& value2, std::shared_ptr<Node>& data, std::shared_ptr<Model>& model) {
    uint64_t hash0 = 0, hash1 = 0;
    data->get_rt_info()["0"] = value1;
    pass::FasterHash(hash0).run_on_model(model);
    pass::FasterHash(hash1).run_on_model(model);
    data->get_rt_info()["0"] = value2;
    EXPECT_EQ(hash1, hash0) << "Reverting rt value didn't restored hash";
    pass::FasterHash(hash1).run_on_model(model);
    EXPECT_NE(hash1, hash0) << "Changing rt value doesn't change hash";
    data->get_rt_info()["0"] = value1;
    pass::FasterHash(hash1).run_on_model(model);
    EXPECT_EQ(hash1, hash0) << "Reverting rt value didn't restored hash";
}

TEST(TransformationTests, FasterHash_rt_info) {
    auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{3, 1, 2});
    param->set_friendly_name("Parameter");
    param->get_output_tensor(0).set_names({"parameter"});
    std::shared_ptr<Node> data = std::make_shared<opset8::Relu>(param);
    auto result = std::make_shared<opset8::Result>(data);
    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{param});
    check_rt_one(true, false, data, model);
    check_rt_one<std::string>("1", "2", data, model);
    check_rt_one<int64_t>(1, 2, data, model);
    check_rt_one<uint64_t>(1, 2, data, model);
    check_rt_one<float>(1, 2, data, model);
    check_rt_one<double>(1, 2, data, model);
    check_rt_one<std::vector<std::string>>({"a", "b", "c"}, {"c", "b", "a"}, data, model);
    check_rt_one<std::vector<int>>({1, 2, 3}, {3, 2, 1}, data, model);
    check_rt_one<std::vector<double>>({1, 2, 3}, {3, 2, 1}, data, model);
    check_rt_one<std::vector<float>>({1, 2, 3}, {3, 2, 1}, data, model);
    check_rt_one<std::vector<int64_t>>({1, 2, 3}, {3, 2, 1}, data, model);
    check_rt_one<std::vector<uint64_t>>({1, 2, 3}, {3, 2, 1}, data, model);
    TestHashAttribute attr1, attr2;
    {
        attr2 = attr1;
        attr2.b = !attr1.b;
        check_rt_one<TestHashAttribute>(attr1, attr2, data, model);
    }
    {
        attr2 = attr1;
        attr2.d = attr1.d + 1;
        check_rt_one<TestHashAttribute>(attr1, attr2, data, model);
    }
    {
        attr2 = attr1;
        attr2.i64 = attr1.i64 + 1;
        check_rt_one<TestHashAttribute>(attr1, attr2, data, model);
    }
    {
        attr2 = attr1;
        attr2.s = attr1.s + "1";
        check_rt_one<TestHashAttribute>(attr1, attr2, data, model);
    }
    {
        attr2 = attr1;
        attr2.string_name = attr1.string_name + "1";
        check_rt_one<TestHashAttribute>(attr1, attr2, data, model);
    }
    {
        attr2 = attr1;
        attr2.vf[0] = attr2.vf[0] + 1;
        check_rt_one<TestHashAttribute>(attr1, attr2, data, model);
    }
    {
        attr2 = attr1;
        attr2.vi64[0] = attr2.vi64[0] + 1;
        check_rt_one<TestHashAttribute>(attr1, attr2, data, model);
    }
    {
        attr2 = attr1;
        attr2.vi[0] = attr2.vi[0] + 1;
        check_rt_one<TestHashAttribute>(attr1, attr2, data, model);
    }
    {
        attr2 = attr1;
        attr2.vs[0] = attr2.vs[0] + "1";
        check_rt_one<TestHashAttribute>(attr1, attr2, data, model);
    }
    {
        attr2 = attr1;
        attr2.vu64[0] = attr2.vu64[0] + 1;
        check_rt_one<TestHashAttribute>(attr1, attr2, data, model);
    }
    {
        attr2 = attr1;
        attr2.string_set = std::set<std::string>{"4", "5", "6"};
        check_rt_one<TestHashAttribute>(attr1, attr2, data, model);
    }
}

TEST(TransformationTests, FasterHash_weights) {
    auto create_fun = [](size_t size, const std::vector<float>& buffer) {
        auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{size});
        auto const1 = opset8::Constant::create(element::f32, Shape{size}, buffer);
        std::shared_ptr<Node> data = std::make_shared<opset8::Add>(param, const1);
        auto result = std::make_shared<opset8::Result>(data);
        return std::make_shared<Model>(ResultVector{result}, ParameterVector{param});
    };
    std::vector<size_t> sizes = {1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256, 257};
    for (const auto& size : sizes) {
        std::vector<float> orig(size, 0);
        uint64_t hash1, hash2;
        auto model_orig = create_fun(size, orig);
        pass::FasterHash(hash1).run_on_model(model_orig);
        for (size_t j = 0; j < size; j++) {
            std::vector<float> modified(size, 0);
            modified[j] = 1;
            auto model_mod = create_fun(size, modified);
            pass::FasterHash(hash2).run_on_model(model_mod);
            ASSERT_NE(hash1, hash2) << "Modifying " << j << " item of " << size << " didn't change hash";
            modified[j] = 0; // restore
            model_mod = create_fun(size, modified);
            pass::FasterHash(hash2).run_on_model(model_mod);
            ASSERT_EQ(hash1, hash2) << "Restoring weights didn't restore hash";
        }
    }
}

template <typename T>
void check_changed_attribute(T& value1, const T& value2, std::shared_ptr<Model>& model, const std::string& name = "") {
    uint64_t hash0, hash1;
    pass::FasterHash(hash0).run_on_model(model);
    T value_copy = value1;
    value1 = value2;
    pass::FasterHash(hash1).run_on_model(model);
    EXPECT_NE(hash1, hash0) << "Changing attribute value doesn't change hash:" << name;
    value1 = value_copy;
    pass::FasterHash(hash1).run_on_model(model);
    EXPECT_EQ(hash1, hash0) << "Reverting attribute value didn't restored hash: " << name;
}

TEST(TransformationTests, FasterHash_attributes) {
    class HashCustomOp : public op::Op {
    public:
        HashCustomOp(const Output<Node>& arg0): op::Op({arg0}) {
            set_output_size(1);
            mergedInputDescription = std::make_shared<op::util::MultiSubGraphOp::MergedInputDescription>(0, 0, 0);
            sliceInputDescription = std::make_shared<op::util::MultiSubGraphOp::SliceInputDescription>(0, 0, 0, 0, 0, 0, 0);
            invariantInputDescription = std::make_shared<op::util::MultiSubGraphOp::InvariantInputDescription>(0, 0);
            concatOutputDescription = std::make_shared<op::util::MultiSubGraphOp::ConcatOutputDescription>(0, 0, 0, 0, 0, 0, 0);
            model = create_model(Shape{1});
            frameworkNodeAttrs.set_opset_name("HashCustomOpset");
            frameworkNodeAttrs.set_type_name("HashTypeName");
            frameworkNodeAttrs["someKey"] = "someValue";
            variableInfo.data_shape = PartialShape {1, 1, 1};
            variableInfo.data_type = element::f32;
            variableInfo.variable_id = "variable_id";
        }

        static std::shared_ptr<Model> create_model(const Shape& shape) {
            auto param = std::make_shared<opset8::Parameter>(element::f32, shape);
            auto data = std::make_shared<opset8::Relu>(param);
            auto result = std::make_shared<opset8::Result>(data);
            auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{param});
            model->set_friendly_name("Inner model");
            model->get_rt_info()["version"] = int64_t(11);
            return model;
        }

        std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
            return {};
        }

        bool visit_attributes(AttributeVisitor &visitor) override {
            visitor.on_attribute("string", s);
            visitor.on_attribute("int64", i64);
            visitor.on_attribute("double", d);
            visitor.on_attribute("bool", b);
            visitor.on_attribute("vector_int64", vi64);
            visitor.on_attribute("vector_uint64", vu64);
            visitor.on_attribute("vector_int", vi);
            visitor.on_attribute("vector_float", vf);
            visitor.on_attribute("vector_string", vs);

            std::vector<std::shared_ptr<op::util::MultiSubGraphOp::InputDescription>> inp_descs {
                    mergedInputDescription, sliceInputDescription, invariantInputDescription
            };
            visitor.on_attribute("input_desc", inp_descs);

            std::vector<std::shared_ptr<op::util::MultiSubGraphOp::OutputDescription>> out_descs {
                    concatOutputDescription
            };
            visitor.on_attribute("output_desc", out_descs);

            visitor.on_attribute("inner_model", model);
            visitor.on_attribute("special_ports", specialBodyPorts);
            visitor.on_attribute("dimension", dimension);
            visitor.on_attribute("pShape", partialShape);
            visitor.on_attribute("typeVector", typeVector);
            visitor.on_attribute("frameworkNodeAttrs", frameworkNodeAttrs);
            auto var = std::make_shared<op::util::Variable>(variableInfo);
            visitor.on_attribute("variable", var);
            return true;
        }

        std::string s = "1";
        int64_t i64 = 1;
        std::vector<int64_t> vi64 = {1, 2, 3};
        std::vector<uint64_t> vu64 = {1, 2, 3};
        std::vector<float> vf = {1.f, 2.f, 3.f};
        std::vector<std::string> vs = {"1", "2", "3"};
        std::vector<int> vi = {1, 2, 3};
        bool b = true;
        double d = 3.14;
        std::shared_ptr<op::util::MultiSubGraphOp::MergedInputDescription> mergedInputDescription;
        std::shared_ptr<op::util::MultiSubGraphOp::SliceInputDescription> sliceInputDescription;
        std::shared_ptr<op::util::MultiSubGraphOp::InvariantInputDescription> invariantInputDescription;

        std::shared_ptr<op::util::MultiSubGraphOp::ConcatOutputDescription> concatOutputDescription;
        std::shared_ptr<Model> model;
        op::v5::Loop::SpecialBodyPorts specialBodyPorts;
        PartialShape partialShape = {1, 2, 3, 3};
        Dimension dimension = {1};
        element::TypeVector typeVector = {element::f32, element::u8};
        op::util::FrameworkNodeAttrs frameworkNodeAttrs;
        op::util::VariableInfo variableInfo;
    };
    uint64_t hash0;
    auto param0 = std::make_shared<opset8::Parameter>(element::f32, Shape{1});
    auto data0 = std::make_shared<HashCustomOp>(param0);
    auto result0 = std::make_shared<opset8::Result>(data0);
    auto model0 = std::make_shared<Model>(ResultVector{result0}, ParameterVector{param0});
    pass::FasterHash(hash0).run_on_model(model0);

    check_changed_attribute(data0->b, !data0->b, model0);
    check_changed_attribute(data0->d, data0->d + 1, model0);
    check_changed_attribute(data0->s, data0->s+ "1", model0);
    check_changed_attribute(data0->i64, data0->i64 + 1, model0);
    check_changed_attribute(
            data0->specialBodyPorts.current_iteration_input_idx, data0->specialBodyPorts.current_iteration_input_idx + 1, model0);
    check_changed_attribute(
            data0->specialBodyPorts.body_condition_output_idx, data0->specialBodyPorts.body_condition_output_idx + 1, model0);
    check_changed_attribute(data0->dimension, Dimension(10, 20), model0, "dimension");
    check_changed_attribute(data0->partialShape, PartialShape{3, 2, 1, 1}, model0, "partialShape");
    check_changed_attribute(data0->typeVector, element::TypeVector{element::i64}, model0, "typeVector");
    check_changed_attribute(data0->variableInfo.variable_id, std::string("newId"), model0, "variableId");
    check_changed_attribute(data0->variableInfo.data_type, element::u8, model0, "variableDataType");
    check_changed_attribute(data0->variableInfo.data_shape, PartialShape{2}, model0, "variableShape");
    {
        auto newVf = data0->vf;
        newVf[0] += 1;
        check_changed_attribute(data0->vf, newVf, model0);
    }
    {
        auto newVs = data0->vs;
        newVs[0] += "1";
        check_changed_attribute(data0->vs, newVs, model0);
    }
    {
        auto newvi64 = data0->vi64;
        newvi64[0] += 1;
        check_changed_attribute(data0->vi64, newvi64, model0);
    }
    {
        auto newvu64 = data0->vu64;
        newvu64[0] += 1;
        check_changed_attribute(data0->vu64, newvu64, model0);
    }
    check_changed_attribute(
            data0->invariantInputDescription->m_input_index, data0->invariantInputDescription->m_input_index + 1, model0, "01");
    check_changed_attribute(
            data0->invariantInputDescription->m_body_parameter_index, data0->invariantInputDescription->m_body_parameter_index + 1, model0, "02");

    check_changed_attribute(
            data0->sliceInputDescription->m_input_index, data0->sliceInputDescription->m_input_index + 1, model0, "03");
    check_changed_attribute(
            data0->sliceInputDescription->m_body_parameter_index, data0->sliceInputDescription->m_body_parameter_index + 1, model0, "04");
    check_changed_attribute(
            data0->sliceInputDescription->m_start, data0->sliceInputDescription->m_start + 1, model0, "05");
    check_changed_attribute(
            data0->sliceInputDescription->m_stride, data0->sliceInputDescription->m_stride + 1, model0, "06");
    check_changed_attribute(
            data0->sliceInputDescription->m_part_size, data0->sliceInputDescription->m_part_size + 1, model0, "07");
    check_changed_attribute(
            data0->sliceInputDescription->m_end, data0->sliceInputDescription->m_end + 1, model0, "08");
    check_changed_attribute(
            data0->sliceInputDescription->m_axis, data0->sliceInputDescription->m_axis + 1, model0, "09");

    check_changed_attribute(
            data0->mergedInputDescription->m_input_index, data0->mergedInputDescription->m_input_index + 1, model0, "10");
    check_changed_attribute(
            data0->mergedInputDescription->m_body_parameter_index, data0->mergedInputDescription->m_body_parameter_index + 1, model0, "11");
    check_changed_attribute(
            data0->mergedInputDescription->m_body_value_index, data0->mergedInputDescription->m_body_value_index + 1, model0, "12");

    check_changed_attribute(
            data0->concatOutputDescription->m_output_index, data0->concatOutputDescription->m_output_index + 1, model0, "13");
    check_changed_attribute(
            data0->concatOutputDescription->m_body_value_index, data0->concatOutputDescription->m_body_value_index + 1, model0, "14");
    check_changed_attribute(
            data0->concatOutputDescription->m_start, data0->concatOutputDescription->m_start + 1, model0, "15");
    check_changed_attribute(
            data0->concatOutputDescription->m_stride, data0->concatOutputDescription->m_stride + 1, model0, "16");
    check_changed_attribute(
            data0->concatOutputDescription->m_part_size, data0->concatOutputDescription->m_part_size + 1, model0, "17");
    check_changed_attribute(
            data0->concatOutputDescription->m_end, data0->concatOutputDescription->m_end + 1, model0, "18");
    check_changed_attribute(
            data0->concatOutputDescription->m_axis, data0->concatOutputDescription->m_axis + 1, model0, "19");
    {
        auto model2 = HashCustomOp::create_model(Shape{2});
        check_changed_attribute(data0->model, model2, model0, "20");
    }
    {
        auto newVersionRt = data0->model->get_rt_info();
        newVersionRt["version"] = int64_t(12);
        check_changed_attribute(data0->model->get_rt_info(), newVersionRt, model0, "21");
    }
    {
        auto model2 = HashCustomOp::create_model(Shape{1});
        model2->set_friendly_name("new friendly name");
        check_changed_attribute(data0->model, model2, model0, "22");
    }
    {
        auto fwAttrs = data0->frameworkNodeAttrs;
        fwAttrs.set_opset_name("OtherOpset");
        check_changed_attribute(data0->frameworkNodeAttrs, fwAttrs, model0, "fwAttrs_opset");
    }
    {
        auto fwAttrs = data0->frameworkNodeAttrs;
        fwAttrs.set_type_name("NewType");
        check_changed_attribute(data0->frameworkNodeAttrs, fwAttrs, model0, "fwAttrs_type");
    }
    {
        auto fwAttrs = data0->frameworkNodeAttrs;
        fwAttrs["someKey"] = "newValue";
        check_changed_attribute(data0->frameworkNodeAttrs, fwAttrs, model0, "fwAttrs_value");
    }
    {
        auto fwAttrs = data0->frameworkNodeAttrs;
        fwAttrs["someNewKey"] = "someNewValue";
        check_changed_attribute(data0->frameworkNodeAttrs, fwAttrs, model0, "fwAttrs_key");
    }
}

TEST(TransformationTests, FasterHash_LSTMv0_special_case) {
    uint64_t hash0, hash1;
    auto create_model = []() -> std::tuple<std::shared_ptr<ov::Model>, std::shared_ptr<op::v0::LSTMCell>> {
        auto p0 = std::make_shared<opset8::Parameter>(element::undefined, PartialShape{-1, -1});
        auto p1 = std::make_shared<opset8::Parameter>(element::undefined, PartialShape{-1, -1});
        auto p2 = std::make_shared<opset8::Parameter>(element::undefined, PartialShape{-1, -1});
        auto p3 = std::make_shared<opset8::Parameter>(element::undefined, PartialShape{-1, -1});
        auto p4 = std::make_shared<opset8::Parameter>(element::undefined, PartialShape{-1, -1});
        auto p5 = std::make_shared<opset8::Parameter>(element::undefined, PartialShape{-1});
        auto p6 = std::make_shared<opset8::Parameter>(element::undefined, PartialShape{-1});
        auto data0 = std::make_shared<op::v0::LSTMCell>(p0, p1, p2, p3, p4, p5, p6, 10);
        auto result0 = std::make_shared<opset8::Result>(data0);
        return {std::make_shared<Model>(ResultVector{result0}, ParameterVector{p0, p1, p2, p3, p4, p5, p6}),
                data0};
    };
    std::shared_ptr<ov::Model> model0;
    std::shared_ptr<op::v0::LSTMCell> cell0;
    std::tie(model0, cell0) = create_model();
    pass::FasterHash(hash0).run_on_model(model0);

    cell0->input(6).get_rt_info()["someNewKey"] = "someNewValue";
    pass::FasterHash(hash1).run_on_model(model0);
    EXPECT_EQ(hash1, hash0) << "LSTMv0 peephole input shall not be taken into account for hashing";
}

TEST(TransformationTests, FasterHash_error_rt_info_ov_model) {
    auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{3, 1, 2});
    param->set_friendly_name("Parameter");
    param->get_output_tensor(0).set_names({"parameter"});
    auto data = std::make_shared<opset8::Relu>(param);
    auto result = std::make_shared<opset8::Result>(data);
    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{param});

    auto param2 = std::make_shared<opset8::Parameter>(element::f32, Shape{3, 1, 2});
    auto data2 = std::make_shared<opset8::Relu>(param2);
    auto result2 = std::make_shared<opset8::Result>(data2);
    auto model2 = std::make_shared<Model>(ResultVector{result2}, ParameterVector{param2});

    uint64_t hash0 = 0;
    TestHashAttribute attr;
    attr.rt_model = model2;
    data->get_rt_info()["otherModel"] = attr;
    EXPECT_ANY_THROW(pass::FasterHash(hash0).run_on_model(model));
}

TEST(TransformationTests, FasterHash_error_rt_info_special_names) {
    auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{3, 1, 2});
    auto data = std::make_shared<opset8::Relu>(param);
    auto result = std::make_shared<opset8::Result>(data);
    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{param});

    uint64_t hash0 = 0;
    TestHashAttribute attr;
    attr.string_name = "name";
    data->get_rt_info()["someAttribute"] = attr;
    EXPECT_ANY_THROW(pass::FasterHash(hash0).run_on_model(model));

    attr.string_name = "version";
    data->get_rt_info()["someAttribute"] = attr;
    EXPECT_ANY_THROW(pass::FasterHash(hash0).run_on_model(model));
}
