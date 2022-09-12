#include <openvino/frontend/exception.hpp>
#include <openvino/frontend/node_context.hpp>
#include <openvino/opsets/opset8.hpp>

#include "exception.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

typedef std::map<size_t, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(std::shared_ptr<Decoder> decoder, TensorMap* tensor_map, ParameterVector* external_parameters)
        :  // TODO: why the following ctor is explicit?
          frontend::NodeContext(decoder->get_op_type()),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_external_parameters(external_parameters) {}

    // Do not search for input in tensor map; try to access it as a constant of specified type T and return its value
    template <typename T>
    T const_input(size_t index) const;

    size_t get_input_size() const override {
        return m_decoder->inputs().size();
    };

    // Search for input in tensor map and return an output port for already converted op
    // TODO: int due to base class uses it, but naturally it should be size_t for PT
    Output<Node> get_input(int index) const override {
        // std::cerr << "Trying to map input to ngraph...";
        OV_FRONTEND_REQUIRE(!m_decoder->input_is_none(index));
        auto input = m_decoder->input(index);
        OV_FRONTEND_REQUIRE(m_tensor_map->count(input));
        return m_tensor_map->at(input);
    }

    // TODO: upstream to base class
    OutputVector inputs() const {
        OutputVector res;
        for (size_t input : m_decoder->inputs()) {
            // std::cerr << "Searching for input: " << input->unique() << "\n";
            OV_FRONTEND_REQUIRE(m_tensor_map->find(input) != m_tensor_map->end());
            res.push_back(m_tensor_map->at(input));
        }
        return res;
    }

    bool input_is_none(size_t index) const {
        return m_decoder->input_is_none(index);
    }

    // Convert the resulting value of this node to ngraph Constant; works correctly only for nodes that produce
    // constant value, naturally for prim::Constant
    OutputVector as_constant() const {
        return m_decoder->as_constant();
    }

    /*
    TODO: Should be uncommented when explicit NodeContext ctor won't require passing op_type
    const std::string& get_op_type() const override {
        return m_decoder->get_op_type();
    }
    */

    std::string get_schema() const {
        return m_decoder->get_schema();
    }

    size_t num_of_outputs() const {
        return m_decoder->num_of_outputs();
    }

    std::vector<size_t> outputs() const {
        return m_decoder->outputs();
    }

    std::shared_ptr<Node> mark_node(std::shared_ptr<Node> ov_node) const {
        return m_decoder->mark_node(ov_node);
    }

    void mark_nodes(std::vector<std::shared_ptr<Node>> ov_nodes) const {
        return m_decoder->mark_nodes(ov_nodes);
    }

    Output<Node> mark_output(Output<Node> ov_output) const {
        return m_decoder->mark_node(ov_output.get_node_shared_ptr());
    }

    Any get_attribute_as_any(const std::string&) const override {
        throw std::runtime_error(
            "There is no any named attributes in Pytorch node, query by attribute name is not implemented");
    }

    void mutate_input(size_t index, Output<Node> ov_output) {
        OV_FRONTEND_REQUIRE(!m_decoder->input_is_none(index));
        auto input = m_decoder->input(index);
        OV_FRONTEND_REQUIRE(m_tensor_map->count(input));
        m_tensor_map->at(input).get_tensor().set_names({std::to_string(input) + "_"});
        // TODO: find out why this doesn't work
        ov_output.get_tensor().add_names({std::to_string(input)});
        (*m_tensor_map)[input] = ov_output;
        m_mutated_tensors.insert(input);
    }

    std::set<size_t> get_mutated_tensors() const {
        return m_mutated_tensors;
    }

    std::shared_ptr<Decoder> get_decoder() const {
        return m_decoder;
    }

    void add_tensor_to_external_context(size_t index, Output<Node> ov_output) {
        if (m_tensor_map->count(index)) {
            std::cerr << "[ WARNING ] External context has tensor. Rewriting." << std::endl;
        }
        ov_output.get_tensor().add_names({std::to_string(index)});
        (*m_tensor_map)[index] = ov_output;
    }

    Output<Node> get_tensor_from_ext(size_t index) {
        if (m_tensor_map->find(index) != m_tensor_map->end()) {
            return m_tensor_map->at(index);
        } else {
            return Output<Node>();
        }
    }

    Output<Node> get_tensor_from_ext_or_create_ext_input(size_t index) {
        if (m_tensor_map->find(index) != m_tensor_map->end()) {
            return m_tensor_map->at(index);
        } else {
            // nested subgraphs case
            auto parameter = std::make_shared<opset8::Parameter>(element::dynamic, PartialShape::dynamic());
            parameter->get_output_tensor(0).add_names({std::to_string(index)});
            (*m_tensor_map)[index] = parameter;
            m_external_parameters->push_back(parameter);
            std::cout << "Nested case, created: " << parameter << std::endl;
            return parameter;
        }
    }

private:
    std::shared_ptr<opset8::Constant> get_constant_at_input(size_t index) const;

    std::shared_ptr<Decoder> m_decoder;
    std::set<size_t> m_mutated_tensors;
    TensorMap* m_tensor_map;
    ParameterVector* m_external_parameters;
};

std::shared_ptr<opset8::Constant> NodeContext::get_constant_at_input(size_t index) const {
    OV_FRONTEND_REQUIRE(!input_is_none(index));
    auto input_node = get_input(index).get_node_shared_ptr();
    auto input = std::dynamic_pointer_cast<opset8::Constant>(input_node);
    FRONT_END_GENERAL_CHECK(input, "Input with index ", index, " cannot be interpretted as Constant: ", input_node);
    return input;
}

template <>
std::vector<int64_t> NodeContext::const_input<std::vector<int64_t>>(size_t index) const {
    return get_constant_at_input(index)->cast_vector<int64_t>();
}

template <>
std::string NodeContext::const_input<std::string>(size_t index) const {
    throw std::runtime_error("Cannot represent string as OV constant: lack of strings support");
    // return get_constant_at_input(index)->cast_vector<std::string>()[0];
}

template <>
ngraph::Strides NodeContext::const_input<ngraph::Strides>(size_t index) const {
    return get_constant_at_input(index)->cast_vector<ngraph::Strides::value_type>();
}

template <>
ngraph::CoordinateDiff NodeContext::const_input<ngraph::CoordinateDiff>(size_t index) const {
    return get_constant_at_input(index)->cast_vector<ngraph::CoordinateDiff::value_type>();
}

template <>
ngraph::Shape NodeContext::const_input<ngraph::Shape>(size_t index) const {
    return get_constant_at_input(index)->cast_vector<ngraph::Shape::value_type>();
}

template <>
int64_t NodeContext::const_input<int64_t>(size_t index) const {
    return get_constant_at_input(index)->cast_vector<int64_t>()[0];
}

template <>
bool NodeContext::const_input<bool>(size_t index) const {
    return get_constant_at_input(index)->cast_vector<bool>()[0];
}

template <>
double NodeContext::const_input<double>(size_t index) const {
    return get_constant_at_input(index)->cast_vector<double>()[0];
}

template <>
float NodeContext::const_input<float>(size_t index) const {
    return get_constant_at_input(index)->cast_vector<float>()[0];
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
