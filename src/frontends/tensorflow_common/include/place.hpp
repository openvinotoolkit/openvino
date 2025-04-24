// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/decoder.hpp"
#include "openvino/frontend/frontend.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

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
    void set_names(const std::vector<std::string>& names) {
        m_names = names;
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
    std::shared_ptr<TensorPlace> get_source_tensor_tf() const;
    std::shared_ptr<OpPlace> get_op();

    // External usage
    std::vector<Ptr> get_consuming_operations() const override;
    Ptr get_producing_operation() const override;
    ov::frontend::Place::Ptr get_source_tensor() const override;
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

    std::shared_ptr<TensorPlace> get_target_tensor_tf() const;

    // External usage
    std::vector<Ptr> get_consuming_operations() const override;
    ov::frontend::Place::Ptr get_producing_operation() const override;
    std::vector<ov::frontend::Place::Ptr> get_consuming_ports() const override;
    Ptr get_target_tensor() const override;
    bool is_equal_data(const Ptr& another) const override;

private:
    std::weak_ptr<OpPlace> m_op;
    std::weak_ptr<TensorPlace> m_target_tensor;
};

class OpPlace : public Place {
public:
    OpPlace(const ov::frontend::InputModel& input_model, std::shared_ptr<DecoderBase> op_decoder);

    void add_in_port(const std::shared_ptr<InPortPlace>& input, const std::string& name);
    void add_out_port(const std::shared_ptr<OutPortPlace>& output, int idx);

    // Internal usage
    const std::vector<std::shared_ptr<OutPortPlace>>& get_output_ports() const;
    const std::map<std::string, std::vector<std::shared_ptr<InPortPlace>>>& get_input_ports() const;
    std::shared_ptr<InPortPlace> get_input_port_tf(const std::string& inputName, int inputPortIndex) const;
    std::shared_ptr<DecoderBase> get_decoder() const;

    // External API methods
    std::vector<ov::frontend::Place::Ptr> get_consuming_ports() const override;

    Ptr get_output_port() const override;
    Ptr get_output_port(int outputPortIndex) const override;

    Ptr get_input_port() const override;
    Ptr get_input_port(int inputPortIndex) const override;
    Ptr get_input_port(const std::string& inputName) const override;
    Ptr get_input_port(const std::string& inputName, int inputPortIndex) const override;

    std::vector<Ptr> get_consuming_operations() const override;
    std::vector<Ptr> get_consuming_operations(int outputPortIndex) const override;

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

    // set back edge for OpPlace of NextIteration operation
    // this is needed since we break a cycle in a graph
    void set_next_iteration_back_edge(const std::string& next_iteration_producer_name,
                                      size_t next_iteration_producer_output_port_idx);
    void get_next_iteration_back_edge(std::string& next_iteration_producer_name,
                                      size_t& next_iteration_producer_output_port_idx) const;

private:
    std::shared_ptr<DecoderBase> m_op_decoder;
    std::map<std::string, std::vector<std::shared_ptr<InPortPlace>>> m_input_ports;
    std::vector<std::shared_ptr<OutPortPlace>> m_output_ports;

    // flag if back edge is set
    bool m_back_edge_set;
    std::string m_next_iteration_producer_name;
    size_t m_next_iteration_producer_output_port_idx;
};

class TensorPlace : public Place {
public:
    TensorPlace(const ov::frontend::InputModel& input_model,
                const ov::PartialShape& pshape,
                ov::element::Type type,
                const std::vector<std::string>& names);

    TensorPlace(const ov::frontend::InputModel& input_model,
                const ov::PartialShape& pshape,
                ov::element::Type type,
                const std::vector<std::string>& names,
                const std::string& operation_name);

    void add_producing_port(const std::shared_ptr<OutPortPlace>& out_port);
    void add_consuming_port(const std::shared_ptr<InPortPlace>& in_port);

    // Internal usage
    const PartialShape& get_partial_shape() const {
        return m_pshape;
    }
    const element::Type& get_element_type() const {
        return m_type;
    }
    const std::string& get_operation_name() const {
        return m_operation_name;
    }
    void set_partial_shape(const PartialShape& pshape) {
        m_pshape = pshape;
    }
    void set_element_type(const element::Type& type) {
        m_type = type;
    }

    // External usage
    Ptr get_producing_operation() const override;
    std::vector<ov::frontend::Place::Ptr> get_consuming_operations() const override;
    std::vector<ov::frontend::Place::Ptr> get_consuming_ports() const override;
    Ptr get_producing_port() const override;
    bool is_equal_data(const Ptr& another) const override;

private:
    PartialShape m_pshape;
    element::Type m_type;
    // store original node name from which tensor place is created
    std::string m_operation_name;

    std::vector<std::weak_ptr<OutPortPlace>> m_producing_ports;
    std::vector<std::weak_ptr<InPortPlace>> m_consuming_ports;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
