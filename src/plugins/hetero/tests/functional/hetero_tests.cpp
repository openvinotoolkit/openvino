// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hetero_tests.hpp"

#include <memory>
#include <string>

#include "common_test_utils/file_utils.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

namespace {

std::string get_mock_engine_path() {
    std::string mock_engine_name("mock_engine");
    return ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                              mock_engine_name + OV_BUILD_POSTFIX);
}

template <class T>
std::function<T> make_std_function(const std::shared_ptr<void> so, const std::string& functionName) {
    std::function<T> ptr(reinterpret_cast<T*>(ov::util::get_symbol(so, functionName.c_str())));
    return ptr;
}

bool support_model(const std::shared_ptr<const ov::Model>& model, const ov::SupportedOpsMap& supported_ops) {
    for (const auto& op : model->get_ops()) {
        if (supported_ops.find(op->get_friendly_name()) == supported_ops.end())
            return false;
    }
    return true;
}

ov::PropertyName RO_property(const std::string& propertyName) {
    return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
};

ov::PropertyName RW_property(const std::string& propertyName) {
    return ov::PropertyName(propertyName, ov::PropertyMutability::RW);
};

}  // namespace

ov::Tensor ov::hetero::tests::HeteroTests::create_and_fill_tensor(const ov::element::Type& type,
                                                                  const ov::Shape& shape) {
    switch (type) {
    case ov::element::Type_t::i64:
        return create_tensor<ov::element_type_traits<ov::element::Type_t::i64>::value_type>(type, shape);
    default:
        break;
    }
    OPENVINO_THROW("Cannot generate tensor. Unsupported element type.");
}

std::shared_ptr<ov::Model> ov::hetero::tests::HeteroTests::create_model_with_subtract(bool dynamic) {
    int64_t bs = dynamic ? -1 : 1;
    auto param = std::make_shared<ov::opset11::Parameter>(ov::element::i64, ov::PartialShape{bs, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::opset11::Add>(param, const_value);
    add->set_friendly_name("add");
    auto subtract = std::make_shared<ov::opset11::Subtract>(add, const_value);
    subtract->set_friendly_name("sub");
    auto result = std::make_shared<ov::opset11::Result>(subtract);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

std::shared_ptr<ov::Model> ov::hetero::tests::HeteroTests::create_model_with_subtract_reshape(bool dynamic) {
    int64_t bs = dynamic ? -1 : 1;
    auto param = std::make_shared<ov::opset11::Parameter>(ov::element::i64, ov::PartialShape{bs, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::opset11::Add>(param, const_value);
    add->set_friendly_name("add");
    auto subtract = std::make_shared<ov::opset11::Subtract>(add, const_value);
    subtract->set_friendly_name("sub");
    auto reshape_val = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    reshape_val->set_friendly_name("reshape_val");
    auto reshape = std::make_shared<ov::opset11::Reshape>(subtract, reshape_val, true);
    reshape->set_friendly_name("reshape");
    auto result = std::make_shared<ov::opset11::Result>(reshape);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

std::shared_ptr<ov::Model> ov::hetero::tests::HeteroTests::create_model_with_subtract_reshape_relu(bool dynamic) {
    int64_t bs = dynamic ? -1 : 1;
    auto param = std::make_shared<ov::opset11::Parameter>(ov::element::i64, ov::PartialShape{bs, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::opset11::Add>(param, const_value);
    add->set_friendly_name("add");
    auto subtract = std::make_shared<ov::opset11::Subtract>(add, const_value);
    subtract->set_friendly_name("sub");
    auto reshape_val = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    reshape_val->set_friendly_name("reshape_val");
    auto reshape = std::make_shared<ov::opset11::Reshape>(subtract, reshape_val, true);
    reshape->set_friendly_name("reshape");
    auto relu = std::make_shared<ov::opset11::Relu>(reshape);
    relu->set_friendly_name("relu");
    auto result = std::make_shared<ov::opset11::Result>(relu);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

std::shared_ptr<ov::Model> ov::hetero::tests::HeteroTests::create_model_with_reshape(bool dynamic) {
    int64_t bs = dynamic ? -1 : 1;
    auto param = std::make_shared<ov::opset11::Parameter>(ov::element::i64, ov::PartialShape{bs, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::opset11::Add>(param, const_value);
    add->set_friendly_name("add");
    auto reshape_val = ov::opset11::Constant::create(ov::element::i64, ov::Shape{3}, {1, 3, 4});
    reshape_val->set_friendly_name("reshape_val");
    auto reshape = std::make_shared<ov::opset11::Reshape>(add, reshape_val, true);
    reshape->set_friendly_name("reshape");
    auto result = std::make_shared<ov::opset11::Result>(reshape);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

std::shared_ptr<ov::Model> ov::hetero::tests::HeteroTests::create_model_with_subtract_shapeof_reshape(bool dynamic) {
    int64_t bs = dynamic ? -1 : 1;
    auto param = std::make_shared<ov::opset11::Parameter>(ov::element::i64, ov::PartialShape{bs, 3, 2, 2});
    param->set_friendly_name("input");
    auto reshape_val0 = ov::opset11::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {bs, 12});
    reshape_val0->set_friendly_name("reshape_val0");
    auto reshape0 = std::make_shared<ov::opset11::Reshape>(param, reshape_val0, true);
    reshape0->set_friendly_name("reshape0");
    auto const_value = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto subtract = std::make_shared<ov::opset11::Subtract>(reshape0, const_value);
    subtract->set_friendly_name("sub");
    auto shape_of = std::make_shared<ov::opset11::ShapeOf>(param);
    shape_of->set_friendly_name("shape_of");
    auto reshape1 = std::make_shared<ov::opset11::Reshape>(subtract, shape_of, true);
    reshape1->set_friendly_name("reshape1");
    auto result = std::make_shared<ov::opset11::Result>(reshape1);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

std::shared_ptr<ov::Model> ov::hetero::tests::HeteroTests::create_model_with_independent_parameter(bool dynamic) {
    int64_t bs = dynamic ? -1 : 1;
    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{bs, 3, 2, 2});
    param1->set_friendly_name("input1");
    auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
    param2->set_friendly_name("input2");
    auto const_value = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::op::v1::Add>(param1, const_value);
    add->set_friendly_name("add");
    auto result = std::make_shared<ov::op::v0::Result>(add);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param1, param2});
}

std::shared_ptr<ov::Model> ov::hetero::tests::HeteroTests::create_model_with_multi_add() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 1, 1});
    param->set_friendly_name("input");
    auto const_value1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 1, 1}, {1});
    const_value1->set_friendly_name("const_val1");
    auto add1 = std::make_shared<ov::op::v1::Add>(param, const_value1);
    add1->set_friendly_name("add1");
    auto const_value2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 1, 1}, {1});
    const_value2->set_friendly_name("const_val2");
    auto add2 = std::make_shared<ov::op::v1::Add>(add1, const_value2);
    add2->set_friendly_name("add2");
    auto const_value3 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 1, 1}, {1});
    const_value3->set_friendly_name("const_val3");
    auto add3 = std::make_shared<ov::op::v1::Add>(add2, const_value3);
    add3->set_friendly_name("add3");
    auto const_value4 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 1, 1}, {1});
    const_value4->set_friendly_name("const_val4");
    auto add4 = std::make_shared<ov::op::v1::Add>(add3, const_value4);
    add4->set_friendly_name("add4");
    auto result = std::make_shared<ov::op::v0::Result>(add4);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}
// Mock plugins

class MockCompiledModel : public ov::ICompiledModel {
public:
    MockCompiledModel(const std::shared_ptr<const ov::Model>& model,
                      const std::shared_ptr<const ov::IPlugin>& plugin,
                      const ov::AnyMap& config)
        : ov::ICompiledModel(model, plugin),
          m_config(config),
          m_model(model),
          m_has_context(false) {}

    MockCompiledModel(const std::shared_ptr<const ov::Model>& model,
                      const std::shared_ptr<const ov::IPlugin>& plugin,
                      const ov::AnyMap& config,
                      const ov::SoPtr<ov::IRemoteContext>& context)
        : ov::ICompiledModel(model, plugin),
          m_config(config),
          m_model(model),
          m_has_context(true),
          m_context(context) {}

    // Methods from a base class ov::ICompiledModel
    void export_model(std::ostream& model) const override {
        ov::pass::StreamSerialize(model, std::function<void(std::ostream&)>())
            .run_on_model(std::const_pointer_cast<ov::Model>(m_model));
    }

    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        auto model = m_model->clone();
        // Add execution information into the model
        size_t exec_order = 0;
        for (const auto& op : model->get_ordered_ops()) {
            auto& info = op->get_rt_info();
            info[ov::exec_model_info::EXECUTION_ORDER] = std::to_string(exec_order++);
            info[ov::exec_model_info::IMPL_TYPE] = get_plugin()->get_device_name() + "_ " + op->get_type_info().name;
            auto perf_count_enabled = get_property(ov::enable_profiling.name()).as<bool>();
            info[ov::exec_model_info::PERF_COUNTER] = perf_count_enabled ? "0" : "not_executed";
            std::string original_names = ov::getFusedNames(op);
            if (original_names.empty()) {
                original_names = op->get_friendly_name();
            } else if (original_names.find(op->get_friendly_name()) == std::string::npos) {
                original_names = op->get_friendly_name() + "," + original_names;
            }
            info[ov::exec_model_info::ORIGINAL_NAMES] = original_names;
            if (op->inputs().size() > 0)
                info[ov::exec_model_info::RUNTIME_PRECISION] = op->get_input_element_type(0);
            else
                info[ov::exec_model_info::RUNTIME_PRECISION] = op->get_output_element_type(0);

            std::stringstream precisions_ss;
            for (size_t i = 0; i < op->get_output_size(); i++) {
                if (i > 0)
                    precisions_ss << ",";
                precisions_ss << op->get_output_element_type(i);
            }
            info[ov::exec_model_info::OUTPUT_PRECISIONS] = precisions_ss.str();
        }
        return model;
    }

    void set_property(const ov::AnyMap& properties) override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    ov::Any get_property(const std::string& name) const override {
        if (name == ov::supported_properties) {
            const std::vector<ov::PropertyName> supported_properties = {ov::num_streams.name(),
                                                                        ov::enable_profiling.name()};
            return decltype(ov::supported_properties)::value_type(supported_properties);
        } else if (name == ov::num_streams) {
            if (m_config.count(ov::internal::exclusive_async_requests.name())) {
                auto exclusive_async_requests = m_config.at(ov::internal::exclusive_async_requests.name()).as<bool>();
                if (exclusive_async_requests)
                    return ov::streams::Num(1);
            }
            return m_config.count(ov::num_streams.name()) ? m_config.at(ov::num_streams.name()) : ov::streams::Num(1);
        } else if (name == ov::enable_profiling) {
            return m_config.count(ov::enable_profiling.name()) ? m_config.at(ov::enable_profiling.name()) : false;
        } else {
            OPENVINO_THROW("get property: " + name);
        }
    }

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    const std::shared_ptr<const ov::Model>& get_model() const {
        return m_model;
    }

    ov::SoPtr<ov::IRemoteContext> get_context() const {
        return m_context;
    }

    bool has_context() const {
        return m_has_context;
    }

private:
    ov::AnyMap m_config;
    std::shared_ptr<const ov::Model> m_model;
    bool m_has_context;
    ov::SoPtr<ov::IRemoteContext> m_context;
};

class MockInferRequest : public ov::ISyncInferRequest {
public:
    MockInferRequest(const std::shared_ptr<const MockCompiledModel>& compiled_model)
        : ov::ISyncInferRequest(compiled_model) {
        OPENVINO_ASSERT(compiled_model);
        m_model = compiled_model->get_model();
        // Allocate input/output tensors
        for (const auto& input : get_inputs()) {
            allocate_tensor(input, [this, input, compiled_model](ov::SoPtr<ov::ITensor>& tensor) {
                // Can add a check to avoid double work in case of shared tensors
                allocate_tensor_impl(tensor,
                                     input.get_element_type(),
                                     input.get_partial_shape().is_dynamic() ? ov::Shape{0} : input.get_shape(),
                                     compiled_model->has_context(),
                                     compiled_model->get_context());
            });
        }
        for (const auto& output : get_outputs()) {
            allocate_tensor(output, [this, output, compiled_model](ov::SoPtr<ov::ITensor>& tensor) {
                // Can add a check to avoid double work in case of shared tensors
                allocate_tensor_impl(tensor,
                                     output.get_element_type(),
                                     output.get_partial_shape().is_dynamic() ? ov::Shape{0} : output.get_shape(),
                                     compiled_model->has_context(),
                                     compiled_model->get_context());
            });
        }
    }
    ~MockInferRequest() = default;

    void infer() override {
        ov::TensorVector input_tensors;
        for (const auto& input : get_inputs()) {
            input_tensors.emplace_back(ov::make_tensor(get_tensor(input)));
        }
        ov::TensorVector output_tensors;
        for (const auto& output : get_outputs()) {
            output_tensors.emplace_back(ov::make_tensor(get_tensor(output)));
        }
        m_model->evaluate(output_tensors, input_tensors);
    }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

private:
    void allocate_tensor_impl(ov::SoPtr<ov::ITensor>& tensor,
                              const ov::element::Type& element_type,
                              const ov::Shape& shape,
                              bool has_context,
                              ov::SoPtr<ov::IRemoteContext> context) {
        if (!tensor || tensor->get_element_type() != element_type) {
            if (has_context) {
                tensor = context->create_tensor(element_type, shape, {});
            } else {
                tensor = ov::SoPtr<ov::ITensor>(ov::make_tensor(element_type, shape), nullptr);
            }
        } else {
            tensor->set_shape(shape);
        }
    }
    std::shared_ptr<const ov::Model> m_model;
};

std::shared_ptr<ov::ISyncInferRequest> MockCompiledModel::create_sync_infer_request() const {
    return std::make_shared<MockInferRequest>(std::dynamic_pointer_cast<const MockCompiledModel>(shared_from_this()));
}

class MockRemoteTensor : public ov::IRemoteTensor {
    ov::AnyMap m_properties;
    std::string m_dev_name;

public:
    MockRemoteTensor(const std::string& name, const ov::AnyMap& props) : m_properties(props), m_dev_name(name) {}
    const ov::AnyMap& get_properties() const override {
        return m_properties;
    }
    const std::string& get_device_name() const override {
        return m_dev_name;
    }
    void set_shape(ov::Shape shape) override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    const ov::element::Type& get_element_type() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    const ov::Shape& get_shape() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    const ov::Strides& get_strides() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
};

class MockRemoteContext : public ov::IRemoteContext {
    ov::AnyMap m_property = {{"IS_DEFAULT", true}};
    std::string m_dev_name;

public:
    MockRemoteContext(const std::string& dev_name) : m_dev_name(dev_name) {}
    const std::string& get_device_name() const override {
        return m_dev_name;
    }

    const ov::AnyMap& get_property() const override {
        return m_property;
    }

    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type,
                                               const ov::Shape& shape,
                                               const ov::AnyMap& params = {}) override {
        auto remote_tensor = std::make_shared<MockRemoteTensor>(m_dev_name, m_property);
        return {remote_tensor, nullptr};
    }
};

class MockCustomRemoteContext : public ov::IRemoteContext {
    ov::AnyMap m_property = {{"IS_DEFAULT", false}};
    std::string m_dev_name;

public:
    MockCustomRemoteContext(const std::string& dev_name) : m_dev_name(dev_name) {}
    const std::string& get_device_name() const override {
        return m_dev_name;
    }

    const ov::AnyMap& get_property() const override {
        return m_property;
    }

    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type,
                                               const ov::Shape& shape,
                                               const ov::AnyMap& params = {}) override {
        auto remote_tensor = std::make_shared<MockRemoteTensor>(m_dev_name, m_property);
        return {remote_tensor, nullptr};
    }
};

class MockPluginBase : public ov::IPlugin {
public:
    MockPluginBase(const std::string& name,
                   const std::unordered_set<std::string>& supported_ops,
                   bool dynamism_supported = false)
        : m_supported_ops(supported_ops),
          m_dynamism_supported(dynamism_supported) {
        set_device_name(name);
    }

    virtual const ov::Version& get_const_version() = 0;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override {
        OPENVINO_ASSERT(model);
        if (!support_model(model, query_model(model, properties)))
            OPENVINO_THROW("Unsupported model");

        return std::make_shared<MockCompiledModel>(model, shared_from_this(), properties);
    }

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::string& model_path,
                                                      const ov::AnyMap& properties) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override {
        if (!support_model(model, query_model(model, properties)))
            OPENVINO_THROW("Unsupported model");

        return std::make_shared<MockCompiledModel>(model, shared_from_this(), properties, context);
    }

    void set_property(const ov::AnyMap& properties) override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override {
        if (remote_properties.find("CUSTOM_CTX") == remote_properties.end())
            return std::make_shared<MockRemoteContext>(get_device_name());
        return std::make_shared<MockCustomRemoteContext>(get_device_name());
    }

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override {
        return std::make_shared<MockRemoteContext>(get_device_name());
    }

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override {
        std::string xmlString, xmlInOutString;
        ov::Tensor weights;

        ov::pass::StreamSerialize::DataHeader hdr = {};
        model.read(reinterpret_cast<char*>(&hdr), sizeof hdr);

        model.seekg(hdr.custom_data_offset);
        xmlInOutString.resize(hdr.custom_data_size);
        model.read(const_cast<char*>(xmlInOutString.c_str()), hdr.custom_data_size);

        // read blob content
        model.seekg(hdr.consts_offset);
        if (hdr.consts_size) {
            weights = ov::Tensor(ov::element::i8, ov::Shape{hdr.consts_size});
            char* data = static_cast<char*>(weights.data());
            model.read(data, hdr.consts_size);
        }

        // read XML content
        model.seekg(hdr.model_offset);
        xmlString.resize(hdr.model_size);
        model.read(const_cast<char*>(xmlString.c_str()), hdr.model_size);

        ov::Core core;
        auto ov_model = core.read_model(xmlString, weights);
        return compile_model(ov_model, properties);
    }

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override {
        std::string xmlString, xmlInOutString;
        ov::Tensor weights;

        ov::pass::StreamSerialize::DataHeader hdr = {};
        model.read(reinterpret_cast<char*>(&hdr), sizeof hdr);

        model.seekg(hdr.custom_data_offset);
        xmlInOutString.resize(hdr.custom_data_size);
        model.read(const_cast<char*>(xmlInOutString.c_str()), hdr.custom_data_size);

        // read blob content
        model.seekg(hdr.consts_offset);
        if (hdr.consts_size) {
            weights = ov::Tensor(ov::element::i8, ov::Shape{hdr.consts_size});
            char* data = static_cast<char*>(weights.data());
            model.read(data, hdr.consts_size);
        }

        // read XML content
        model.seekg(hdr.model_offset);
        xmlString.resize(hdr.model_size);
        model.read(const_cast<char*>(xmlString.c_str()), hdr.model_size);

        ov::Core core;
        auto ov_model = core.read_model(xmlString, weights);
        return compile_model(ov_model, properties, context);
    }

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override {
        OPENVINO_ASSERT(model);
        ov::SupportedOpsMap res;
        auto device_id = properties.count(ov::device::id.name())
                             ? properties.at(ov::device::id.name()).as<std::string>()
                             : m_default_device_id;
        float query_model_ratio = properties.count(ov::internal::query_model_ratio.name())
                                      ? properties.at(ov::internal::query_model_ratio.name()).as<float>()
                                      : 1.0f;
        auto supported = ov::get_supported_nodes(
            model,
            [&](std::shared_ptr<ov::Model>& model) {
                ov::pass::Manager manager;
                manager.register_pass<ov::pass::InitNodeInfo>();
                manager.register_pass<ov::pass::ConstantFolding>();
                manager.run_passes(model);
            },
            [&](const std::shared_ptr<ov::Node>& op) {
                if (op->is_dynamic() && !m_dynamism_supported)
                    return false;
                if (m_supported_ops.find(op->get_type_info().name) == m_supported_ops.end())
                    return false;
                return true;
            },
            query_model_ratio);
        for (auto&& op_name : supported) {
            res.emplace(op_name, get_device_name() + "." + device_id);
        }
        return res;
    }

protected:
    std::string m_default_device_id = "0";
    std::unordered_set<std::string> m_supported_ops;
    bool m_dynamism_supported = false;
    bool m_profiling = false;
    bool m_loaded_from_cache{false};
};

class MockPluginReshape : public MockPluginBase {
public:
    MockPluginReshape(const std::string& name)
        : MockPluginBase(name, {"Parameter", "Result", "Add", "Constant", "Reshape"}, true) {}

    const ov::Version& get_const_version() override {
        static const ov::Version version = {CI_BUILD_NUMBER, "openvino_mock_reshape_plugin"};
        return version;
    }
    void set_property(const ov::AnyMap& properties) override {
        for (const auto& it : properties) {
            if (it.first == ov::num_streams.name())
                num_streams = it.second.as<int32_t>();
            else if (it.first == ov::enable_profiling.name())
                m_profiling = it.second.as<bool>();
            else if (it.first == ov::internal::exclusive_async_requests.name())
                exclusive_async_requests = it.second.as<bool>();
            else if (it.first == ov::device::id.name())
                continue;
            else
                OPENVINO_THROW(get_device_name(), " set config: " + it.first);
        }
    }

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override {
        const static std::vector<std::string> device_ids = {"0", "1", "2"};
        const static std::vector<ov::PropertyName> roProperties{
            RO_property(ov::supported_properties.name()),
            RO_property(ov::available_devices.name()),
            RO_property(ov::loaded_from_cache.name()),
            RO_property(ov::device::uuid.name()),
        };
        // the whole config is RW before network is loaded.
        const static std::vector<ov::PropertyName> rwProperties{
            RW_property(ov::num_streams.name()),
            RW_property(ov::enable_profiling.name()),
        };

        std::string device_id;
        if (arguments.find(ov::device::id.name()) != arguments.end()) {
            device_id = arguments.find(ov::device::id.name())->second.as<std::string>();
        }
        if (name == ov::supported_properties) {
            std::vector<ov::PropertyName> supportedProperties;
            supportedProperties.reserve(roProperties.size() + rwProperties.size());
            supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
            supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());

            return decltype(ov::supported_properties)::value_type(supportedProperties);
        } else if (name == ov::internal::supported_properties) {
            return decltype(ov::internal::supported_properties)::value_type(
                {ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO},
                 ov::PropertyName{ov::internal::exclusive_async_requests.name(), ov::PropertyMutability::RW}});
        } else if (name == ov::internal::exclusive_async_requests) {
            return decltype(ov::internal::exclusive_async_requests)::value_type{exclusive_async_requests};
        } else if (name == ov::device::uuid) {
            ov::device::UUID uuid;
            for (size_t i = 0; i < uuid.MAX_UUID_SIZE; i++) {
                if (device_id == device_ids[0])
                    uuid.uuid[i] = static_cast<uint8_t>(i);
                else if (device_id == device_ids[1])
                    uuid.uuid[i] = static_cast<uint8_t>(i * 2);
                else if (device_id == device_ids[2])
                    uuid.uuid[i] = static_cast<uint8_t>(i * 3);
            }
            return decltype(ov::device::uuid)::value_type{uuid};
        } else if (name == ov::available_devices) {
            return decltype(ov::available_devices)::value_type(device_ids);
        } else if (name == ov::device::capabilities) {
            std::vector<std::string> capabilities;
            capabilities.push_back(ov::device::capability::EXPORT_IMPORT);
            return decltype(ov::device::capabilities)::value_type(capabilities);
        } else if (ov::internal::caching_properties == name) {
            std::vector<ov::PropertyName> caching_properties = {ov::device::uuid};
            return decltype(ov::internal::caching_properties)::value_type(caching_properties);
        } else if (name == ov::loaded_from_cache.name()) {
            return m_loaded_from_cache;
        } else if (name == ov::enable_profiling.name()) {
            return decltype(ov::enable_profiling)::value_type{m_profiling};
        } else if (name == ov::streams::num.name()) {
            return decltype(ov::streams::num)::value_type{num_streams};
        }
        OPENVINO_THROW("Unsupported property: ", name);
    }

private:
    int32_t num_streams{0};
    bool exclusive_async_requests = false;
};

class MockPluginSubtract : public MockPluginBase {
public:
    MockPluginSubtract(const std::string& name)
        : MockPluginBase(name, {"Parameter", "Result", "Add", "Constant", "Subtract"}) {}

    const ov::Version& get_const_version() override {
        static const ov::Version version = {CI_BUILD_NUMBER, "openvino_mock_subtract_plugin"};
        return version;
    }

    void set_property(const ov::AnyMap& properties) override {
        for (const auto& it : properties) {
            if (it.first == ov::enable_profiling.name())
                m_profiling = it.second.as<bool>();
            else if (it.first == ov::device::id.name())
                continue;
            else
                OPENVINO_THROW(get_device_name(), " set config: " + it.first);
        }
    }

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override {
        const static std::vector<std::string> device_ids = {"0", "1"};
        const static std::vector<ov::PropertyName> roProperties{
            RO_property(ov::supported_properties.name()),
            RO_property(ov::available_devices.name()),
            RO_property(ov::loaded_from_cache.name()),
            RO_property(ov::device::uuid.name()),
        };
        // the whole config is RW before network is loaded.
        const static std::vector<ov::PropertyName> rwProperties{
            RW_property(ov::num_streams.name()),
            RW_property(ov::enable_profiling.name()),
        };
        std::string device_id;
        if (arguments.find(ov::device::id.name()) != arguments.end()) {
            device_id = arguments.find(ov::device::id.name())->second.as<std::string>();
        }
        if (name == ov::supported_properties) {
            std::vector<ov::PropertyName> supportedProperties;
            supportedProperties.reserve(roProperties.size() + rwProperties.size());
            supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
            supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());

            return decltype(ov::supported_properties)::value_type(supportedProperties);
        } else if (name == ov::internal::supported_properties) {
            return decltype(ov::internal::supported_properties)::value_type(
                {ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO}});
        } else if (name == ov::device::uuid) {
            ov::device::UUID uuid;
            for (size_t i = 0; i < uuid.MAX_UUID_SIZE; i++) {
                if (device_id == device_ids[0])
                    uuid.uuid[i] = static_cast<uint8_t>(i * 2);
                else if (device_id == device_ids[1])
                    uuid.uuid[i] = static_cast<uint8_t>(i * 4);
                else if (device_id == device_ids[2])
                    uuid.uuid[i] = static_cast<uint8_t>(i * 5);
            }
            return decltype(ov::device::uuid)::value_type{uuid};
        } else if (name == ov::available_devices) {
            return decltype(ov::available_devices)::value_type(device_ids);
        } else if (name == ov::device::capabilities) {
            std::vector<std::string> capabilities;
            capabilities.push_back(ov::device::capability::EXPORT_IMPORT);
            return decltype(ov::device::capabilities)::value_type(capabilities);
        } else if (name == ov::loaded_from_cache.name()) {
            return m_loaded_from_cache;
        } else if (name == ov::enable_profiling.name()) {
            return decltype(ov::enable_profiling)::value_type{m_profiling};
        } else if (ov::internal::caching_properties == name) {
            std::vector<ov::PropertyName> caching_properties = {ov::device::uuid};
            return decltype(ov::internal::caching_properties)::value_type(caching_properties);
        }
        OPENVINO_THROW("Unsupported property: ", name);
    }
};

class MockPluginGPU : public MockPluginBase {
public:
    MockPluginGPU(const std::string& name)
        : MockPluginBase(name, {"Parameter", "Result", "Add", "Constant", "Reshape"}, true) {}

    const ov::Version& get_const_version() override {
        static const ov::Version version = {CI_BUILD_NUMBER, "openvino_mock_reshape_plugin"};
        return version;
    }
    void set_property(const ov::AnyMap& properties) override {
        for (const auto& it : properties) {
            if (it.first == ov::num_streams.name())
                num_streams = it.second.as<int32_t>();
            else if (it.first == ov::enable_profiling.name())
                m_profiling = it.second.as<bool>();
            else if (it.first == ov::internal::exclusive_async_requests.name())
                exclusive_async_requests = it.second.as<bool>();
            else if (it.first == ov::device::id.name())
                continue;
            else
                OPENVINO_THROW(get_device_name(), " set config: " + it.first);
        }
    }

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override {
        const static std::vector<std::string> device_ids = {"0", "1", "2"};
        const std::vector<ov::PropertyName> roProperties{RO_property(ov::supported_properties.name()),
                                                         RO_property(ov::optimal_batch_size.name()),
                                                         RO_property(ov::device::capabilities.name()),
                                                         RO_property(ov::device::type.name()),
                                                         RO_property(ov::device::uuid.name()),
                                                         RO_property(ov::device::id.name()),
                                                         RO_property(ov::intel_gpu::memory_statistics.name()),
                                                         RO_property(ov::intel_gpu::device_total_mem_size.name())};
        // the whole config is RW before network is loaded.
        const std::vector<ov::PropertyName> rwProperties{RW_property(ov::num_streams.name()),
                                                         RW_property(ov::enable_profiling.name()),
                                                         RW_property(ov::compilation_num_threads.name()),
                                                         RW_property(ov::hint::performance_mode.name()),
                                                         RW_property(ov::hint::num_requests.name())};
        std::string device_id;
        if (arguments.find(ov::device::id.name()) != arguments.end()) {
            device_id = arguments.find(ov::device::id.name())->second.as<std::string>();
        }
        if (name == ov::supported_properties) {
            std::vector<ov::PropertyName> supportedProperties;
            supportedProperties.reserve(roProperties.size() + rwProperties.size());
            supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
            supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());

            return decltype(ov::supported_properties)::value_type(supportedProperties);
        } else if (name == ov::internal::supported_properties) {
            return decltype(ov::internal::supported_properties)::value_type(
                {ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO},
                 ov::PropertyName{ov::internal::exclusive_async_requests.name(), ov::PropertyMutability::RW},
                 ov::PropertyName{ov::internal::query_model_ratio.name(), ov::PropertyMutability::RW}});
        } else if (name == ov::internal::exclusive_async_requests) {
            return decltype(ov::internal::exclusive_async_requests)::value_type{exclusive_async_requests};
        } else if (name == ov::device::uuid) {
            ov::device::UUID uuid;
            for (size_t i = 0; i < uuid.MAX_UUID_SIZE; i++) {
                if (device_id == device_ids[0])
                    uuid.uuid[i] = static_cast<uint8_t>(i);
                else if (device_id == device_ids[1])
                    uuid.uuid[i] = static_cast<uint8_t>(i * 2);
                else if (device_id == device_ids[2])
                    uuid.uuid[i] = static_cast<uint8_t>(i * 3);
            }
            return decltype(ov::device::uuid)::value_type{uuid};
        } else if (name == ov::available_devices) {
            return decltype(ov::available_devices)::value_type(device_ids);
        } else if (name == ov::device::capabilities) {
            std::vector<std::string> capabilities;
            capabilities.push_back(ov::device::capability::EXPORT_IMPORT);
            return decltype(ov::device::capabilities)::value_type(capabilities);
        } else if (ov::internal::caching_properties == name) {
            std::vector<ov::PropertyName> caching_properties = {ov::device::uuid};
            return decltype(ov::internal::caching_properties)::value_type(caching_properties);
        } else if (name == ov::loaded_from_cache.name()) {
            return m_loaded_from_cache;
        } else if (name == ov::enable_profiling.name()) {
            return decltype(ov::enable_profiling)::value_type{m_profiling};
        } else if (name == ov::streams::num.name()) {
            return decltype(ov::streams::num)::value_type{num_streams};
        } else if (name == ov::intel_gpu::device_total_mem_size.name()) {
            size_t mem_size = 0;
            if (device_id == "0")
                mem_size = 64;
            else if (device_id == "1")
                mem_size = 16;
            else if (device_id == "2")
                mem_size = 32;
            return decltype(ov::intel_gpu::device_total_mem_size)::value_type{mem_size};
        } else if (name == ov::device::type.name()) {
            ov::device::Type device_type = ov::device::Type::INTEGRATED;
            if (device_id == "0")
                device_type = ov::device::Type::INTEGRATED;
            else if (device_id == "1")
                device_type = ov::device::Type::DISCRETE;
            else if (device_id == "2")
                device_type = ov::device::Type::DISCRETE;
            return decltype(ov::device::type)::value_type(device_type);
        }
        OPENVINO_THROW("Unsupported property: ", name);
    }

private:
    int32_t num_streams{0};
    bool exclusive_async_requests = false;
};

void ov::hetero::tests::HeteroTests::reg_plugin(std::shared_ptr<ov::IPlugin>& plugin) {
    std::string library_path = get_mock_engine_path();
    if (!m_so)
        m_so = ov::util::load_shared_object(library_path.c_str());
    if (auto mock_plugin = std::dynamic_pointer_cast<MockPluginBase>(plugin))
        mock_plugin->set_version(mock_plugin->get_const_version());
    std::function<void(ov::IPlugin*)> injectProxyEngine = make_std_function<void(ov::IPlugin*)>(m_so, "InjectPlugin");

    injectProxyEngine(plugin.get());
    core.register_plugin(library_path, plugin->get_device_name());
    m_mock_plugins.emplace_back(plugin);
}

template <typename T>
void ov::hetero::tests::HeteroTests::reg_plugin_type(const std::string& device_name) {
    auto plugin = std::dynamic_pointer_cast<ov::IPlugin>(std::make_shared<T>(device_name));
    reg_plugin(plugin);
}

void ov::hetero::tests::HeteroTests::SetUp() {
    if (m_mock_plugins.empty()) {
        reg_plugin_type<MockPluginReshape>("MOCK0");
        reg_plugin_type<MockPluginSubtract>("MOCK1");
        reg_plugin_type<MockPluginGPU>("MOCKGPU");
    }
}