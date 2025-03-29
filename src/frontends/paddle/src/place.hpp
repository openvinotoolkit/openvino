// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "input_model.hpp"
#include "openvino/frontend/manager.hpp"

namespace paddle {
namespace framework {
namespace proto {
class OpDesc;
class VarDesc;

}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace ov {
namespace frontend {
namespace paddle {

class TensorPlace;
class OpPlace;

class Place : public ov::frontend::Place {
public:
    Place(const ov::frontend::InputModel& input_model, const std::vector<std::string>& names)
        : m_input_model(input_model),
          m_names(names) {}

    explicit Place(const ov::frontend::InputModel& input_model) : Place(input_model, std::vector<std::string>{}) {}

    ~Place() override = default;

    bool is_input() const override;
    bool is_output() const override;
    bool is_equal(const Ptr& another) const override {
        return this == another.get();
    }

    std::vector<std::string> get_names() const override {
        return m_names;
    }

    int64_t get_version() const {
        return dynamic_cast<const ov::frontend::paddle::InputModel&>(m_input_model).get_version();
    }

private:
    const ov::frontend::InputModel& m_input_model;
    std::vector<std::string> m_names;
};

class InPortPlace : public Place {
public:
    explicit InPortPlace(const ov::frontend::InputModel& input_model) : Place(input_model) {}

    void set_op(const std::weak_ptr<OpPlace>& op) {
        m_op = op;
    }
    void set_source_tensor(const std::weak_ptr<TensorPlace>& source_tensor);

    // Internal usage
    std::shared_ptr<TensorPlace> get_source_tensor_paddle() const;
    std::shared_ptr<OpPlace> get_op();

    // External usage
    std::vector<Ptr> get_consuming_operations() const override;
    Ptr get_producing_operation() const override;
    Place::Ptr get_source_tensor() const override;
    Ptr get_producing_port() const override;

    bool is_equal_data(const Ptr& another) const override;

private:
    std::weak_ptr<TensorPlace> m_source_tensor;
    std::weak_ptr<OpPlace> m_op;
};

class OutPortPlace : public Place {
public:
    explicit OutPortPlace(const ov::frontend::InputModel& input_model) : Place(input_model) {}

    void set_op(const std::weak_ptr<OpPlace>& op) {
        m_op = op;
    }
    void set_target_tensor(const std::weak_ptr<TensorPlace>& target_tensor);

    std::shared_ptr<TensorPlace> get_target_tensor_paddle() const;

    // External usage
    std::vector<Ptr> get_consuming_operations() const override;
    Place::Ptr get_producing_operation() const override;
    std::vector<Place::Ptr> get_consuming_ports() const override;
    Ptr get_target_tensor() const override;
    bool is_equal_data(const Ptr& another) const override;

private:
    std::weak_ptr<OpPlace> m_op;
    std::weak_ptr<TensorPlace> m_target_tensor;
};

class OpPlace : public Place {
public:
    OpPlace(const ov::frontend::InputModel& input_model,
            const ::paddle::framework::proto::OpDesc& op_desc,
            const std::vector<std::string>& names);

    OpPlace(const ov::frontend::InputModel& input_model, const ::paddle::framework::proto::OpDesc& op_desc);

    void add_in_port(const std::shared_ptr<InPortPlace>& input, const std::string& name);
    void add_out_port(const std::shared_ptr<OutPortPlace>& output, const std::string& name);

    // Internal usage
    const std::map<std::string, std::vector<std::shared_ptr<OutPortPlace>>>& get_output_ports() const;
    const std::map<std::string, std::vector<std::shared_ptr<InPortPlace>>>& get_input_ports() const;
    std::shared_ptr<OutPortPlace> get_output_port_paddle(const std::string& outputName, int outputPortIndex) const;
    std::shared_ptr<InPortPlace> get_input_port_paddle(const std::string& inputName, int inputPortIndex) const;
    const ::paddle::framework::proto::OpDesc& get_desc() const;
    const std::shared_ptr<DecoderBase> get_decoder() const;
    void set_decoder(const std::shared_ptr<DecoderBase> op_decoder);

    // External API methods
    std::vector<Place::Ptr> get_consuming_ports() const override;

    Ptr get_output_port() const override;
    Ptr get_output_port(int outputPortIndex) const override;
    Ptr get_output_port(const std::string& outputPortName) const override;
    Ptr get_output_port(const std::string& outputPortName, int outputPortIndex) const override;

    Ptr get_input_port() const override;
    Ptr get_input_port(int inputPortIndex) const override;
    Ptr get_input_port(const std::string& inputName) const override;
    Ptr get_input_port(const std::string& inputName, int inputPortIndex) const override;

    std::vector<Ptr> get_consuming_operations() const override;
    std::vector<Ptr> get_consuming_operations(int outputPortIndex) const override;
    std::vector<Ptr> get_consuming_operations(const std::string& outputPortName) const override;
    std::vector<Ptr> get_consuming_operations(const std::string& outputPortName, int outputPortIndex) const override;

    Ptr get_producing_operation() const override;
    Ptr get_producing_operation(int inputPortIndex) const override;
    Ptr get_producing_operation(const std::string& inputName) const override;
    Ptr get_producing_operation(const std::string& inputName, int inputPortIndex) const override;

    Ptr get_source_tensor() const override;
    Ptr get_source_tensor(int inputPortIndex) const override;
    Ptr get_source_tensor(const std::string& inputName) const override;
    Ptr get_source_tensor(const std::string& inputName, int inputPortIndex) const override;

    Ptr get_target_tensor() const override;
    Ptr get_target_tensor(int outputPortIndex) const override;
    Ptr get_target_tensor(const std::string& outputName) const override;
    Ptr get_target_tensor(const std::string& outputName, int outputPortIndex) const override;

private:
    const ::paddle::framework::proto::OpDesc& m_op_desc;  // TODO: to conceal it behind decoder.
    std::shared_ptr<DecoderBase> m_op_decoder;
    std::map<std::string, std::vector<std::shared_ptr<InPortPlace>>> m_input_ports;
    std::map<std::string, std::vector<std::shared_ptr<OutPortPlace>>> m_output_ports;
};

class TensorPlace : public Place {
public:
    TensorPlace(const ov::frontend::InputModel& input_model,
                const std::vector<std::string>& names,
                const ::paddle::framework::proto::VarDesc& var_desc);

    TensorPlace(const ov::frontend::InputModel& input_model, const ::paddle::framework::proto::VarDesc& var_desc);

    void add_producing_port(const std::shared_ptr<OutPortPlace>& out_port);
    void add_consuming_port(const std::shared_ptr<InPortPlace>& in_port);

    // Internal usage
    const PartialShape& get_partial_shape() const {
        return m_pshape;
    }
    const element::Type& get_element_type() const {
        return m_type;
    }
    void set_partial_shape(const PartialShape& pshape) {
        m_pshape = pshape;
    }
    void set_element_type(const element::Type& type) {
        m_type = type;
    }
    const ::paddle::framework::proto::VarDesc& get_desc() const;

    // External usage
    Ptr get_producing_operation() const override;
    std::vector<Place::Ptr> get_consuming_operations() const override;
    std::vector<Place::Ptr> get_consuming_ports() const override;
    Ptr get_producing_port() const override;
    bool is_equal_data(const Ptr& another) const override;

private:
    const ::paddle::framework::proto::VarDesc& m_var_desc;
    PartialShape m_pshape;
    element::Type m_type;

    std::vector<std::weak_ptr<OutPortPlace>> m_producing_ports;
    std::vector<std::weak_ptr<InPortPlace>> m_consuming_ports;
};

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
