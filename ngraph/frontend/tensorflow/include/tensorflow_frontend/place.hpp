// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend.hpp>

#include "node_def.pb.h"
//#include "node_context_new.hpp"
//#include "decoder_new.hpp"

/*
namespace tensorflow {
namespace framework {
namespace proto {
class NodeDef;
}  // namespace proto
}  // namespace framework
}  // namespace tensorflow
*/

namespace ngraph {
namespace frontend {
using InPortName = size_t;
using OutPortName = size_t;
using NamedOutputs = std::map<OutPortName, OutputVector>;
using NamedInputs = std::map<InPortName, OutputVector>;

class DecoderBase {
public:
    /// \brief Get attribute value by name and requested type
    ///
    /// \param name Attribute name
    /// \param type_info Attribute type information
    /// \return Shared pointer to appropriate value if it exists, 'nullptr' otherwise
    virtual std::shared_ptr<Variant> get_attribute(const std::string& name, const VariantTypeInfo& type_info) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual void get_input_node(const size_t input_port_idx, std::string& producer_name,
        size_t& producer_output_port_index) const = 0;

    virtual std::vector<OutPortName> get_output_names() const = 0;

    virtual size_t get_output_size() const = 0;

    /// \brief Get output port type
    ///
    /// Current API assumes that output port has only one output type.
    /// If decoder supports multiple types for specified port, it shall throw general
    /// exception
    ///
    /// \param port_name Port name for the node
    ///
    /// \return Type of specified output port
    virtual ngraph::element::Type get_out_port_type(const size_t& port_index) const = 0;

    virtual std::string get_op_type() const = 0;

    virtual std::string get_op_name() const = 0;
};

class TensorPlaceTF;
class OpPlaceTF;

namespace tensorflow {
namespace detail {
class TFNodeDecoder;
}  // namespace detail
}  // namespace tensorflow

class PlaceTF : public Place {
public:
    PlaceTF(const InputModel& input_model, const std::vector<std::string>& names)
        : m_input_model(input_model),
          m_names(names) {}

    explicit PlaceTF(const InputModel& input_model) : PlaceTF(input_model, std::vector<std::string>{}) {}

    ~PlaceTF() override = default;

    bool is_input() const override;
    bool is_output() const override;
    bool is_equal(Ptr another) const override {
        return this == another.get();
    }

    std::vector<std::string> get_names() const override {
        return m_names;
    }

private:
    const InputModel& m_input_model;
    std::vector<std::string> m_names;
};

class InPortPlaceTF : public PlaceTF {
public:
    explicit InPortPlaceTF(const InputModel& input_model) : PlaceTF(input_model) {}

    void set_op(const std::weak_ptr<OpPlaceTF>& op) {
        m_op = op;
    }
    void set_source_tensor(const std::weak_ptr<TensorPlaceTF>& source_tensor);

    // Internal usage
    std::shared_ptr<TensorPlaceTF> get_source_tensor_tf() const;
    std::shared_ptr<OpPlaceTF> get_op();

    // External usage
    std::vector<Ptr> get_consuming_operations() const override;
    Ptr get_producing_operation() const override;
    Place::Ptr get_source_tensor() const override;
    Ptr get_producing_port() const override;

    bool is_equal_data(Ptr another) const override;

private:
    std::weak_ptr<TensorPlaceTF> m_source_tensor;
    std::weak_ptr<OpPlaceTF> m_op;
};

class OutPortPlaceTF : public PlaceTF {
public:
    explicit OutPortPlaceTF(const InputModel& input_model) : PlaceTF(input_model) {}

    void set_op(const std::weak_ptr<OpPlaceTF>& op) {
        m_op = op;
    }
    void set_target_tensor(const std::weak_ptr<TensorPlaceTF>& target_tensor);

    std::shared_ptr<TensorPlaceTF> get_target_tensor_tf() const;

    // External usage
    std::vector<Ptr> get_consuming_operations() const override;
    Place::Ptr get_producing_operation() const override;
    std::vector<Place::Ptr> get_consuming_ports() const override;
    Ptr get_target_tensor() const override;
    bool is_equal_data(Ptr another) const override;

private:
    std::weak_ptr<OpPlaceTF> m_op;
    std::weak_ptr<TensorPlaceTF> m_target_tensor;
};

class OpPlaceTF : public PlaceTF {
public:
    OpPlaceTF(const InputModel& input_model,
              std::shared_ptr<ngraph::frontend::tensorflow::detail::TFNodeDecoder> op_def,
              const std::vector<std::string>& names);

    OpPlaceTF(const InputModel& input_model,
              std::shared_ptr<ngraph::frontend::tensorflow::detail::TFNodeDecoder> op_def);

    OpPlaceTF(const InputModel& input_model, std::shared_ptr<DecoderBase> op_decoder);

    void add_in_port(const std::shared_ptr<InPortPlaceTF>& input, const std::string& name);
    void add_out_port(const std::shared_ptr<OutPortPlaceTF>& output, int idx);

    // Internal usage
    const std::vector<std::shared_ptr<OutPortPlaceTF>>& get_output_ports() const;
    const std::map<std::string, std::vector<std::shared_ptr<InPortPlaceTF>>>& get_input_ports() const;
    std::shared_ptr<InPortPlaceTF> get_input_port_tf(const std::string& inputName, int inputPortIndex) const;
    std::shared_ptr<ngraph::frontend::tensorflow::detail::TFNodeDecoder> get_desc() const;
    std::shared_ptr<DecoderBase> get_desc_new() const;
    //::ngraph::frontend::DecoderTFProto* get_desc_new() const;

    // External API methods
    std::vector<Place::Ptr> get_consuming_ports() const override;

    Ptr get_output_port() const override;
    Ptr get_output_port(int outputPortIndex) const override;
    // TODO: implement
    // Ptr get_output_port(int outputPortIndex) const override;

    Ptr get_input_port() const override;
    Ptr get_input_port(int inputPortIndex) const override;
    Ptr get_input_port(const std::string& inputName) const override;
    Ptr get_input_port(const std::string& inputName, int inputPortIndex) const override;

    std::vector<Ptr> get_consuming_operations() const override;
    std::vector<Ptr> get_consuming_operations(int outputPortIndex) const override;
    // TODO: implement
    // std::vector<Ptr> get_consuming_operations(int outputPortIndex) const override;

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
    // TODO: implement (this method may be not needed)
    // Ptr get_target_tensor(int outputPortIndex) const override;

private:
    std::shared_ptr<DecoderBase> m_op_decoder;
    std::shared_ptr<ngraph::frontend::tensorflow::detail::TFNodeDecoder> m_op_def;
    std::map<std::string, std::vector<std::shared_ptr<InPortPlaceTF>>> m_input_ports;
    std::vector<std::shared_ptr<OutPortPlaceTF>> m_output_ports;
};

class TensorPlaceTF : public PlaceTF {
public:
    TensorPlaceTF(const InputModel& input_model,
                  ngraph::PartialShape pshape,
                  ngraph::element::Type type,
                  const std::vector<std::string>& names);

    void add_producing_port(const std::shared_ptr<OutPortPlaceTF>& out_port);
    void add_consuming_port(const std::shared_ptr<InPortPlaceTF>& in_port);

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

    // External usage
    Ptr get_producing_operation() const override;
    std::vector<Place::Ptr> get_consuming_operations() const override;
    std::vector<Place::Ptr> get_consuming_ports() const override;
    Ptr get_producing_port() const override;
    bool is_equal_data(Ptr another) const override;

private:
    PartialShape m_pshape;
    element::Type m_type;

    std::vector<std::weak_ptr<OutPortPlaceTF>> m_producing_ports;
    std::vector<std::weak_ptr<InPortPlaceTF>> m_consuming_ports;
};

}  // namespace frontend
}  // namespace ngraph
