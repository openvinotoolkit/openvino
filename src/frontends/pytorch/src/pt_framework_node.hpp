#include <openvino/op/util/framework_node.hpp>

#pragma once

namespace ov {
namespace frontend {
namespace pytorch {
class PtFrameworkNode : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("PtFrameworkNode", "util", ::ov::op::util::FrameworkNode);

    PtFrameworkNode(const std::shared_ptr<Decoder>& decoder, const OutputVector& inputs, size_t output_size)
        : ov::op::util::FrameworkNode(inputs, output_size, decoder->get_subgraph_size()),
          m_decoder(decoder) {
        ov::op::util::FrameworkNodeAttrs attrs;
        // std::cerr << "[ DEBUG ] Making  PtFrameworkNode for " << m_decoder->get_op_type() << "\n";
        attrs.set_type_name("PTFrameworkNode");
        attrs["PtTypeName"] = m_decoder->get_op_type();
        attrs["PtSchema"] = m_decoder->get_schema();
        set_attrs(attrs);

        // std::cout << attrs["PtTypeName"] << std::endl;

        // Set output shapes and types if recognized

        for (size_t i = 0; i < output_size; ++i) {
            PartialShape ps;
            // TODO: Try to decode PT type as a custom type
            Any type = element::dynamic;
            // FIXME: ROUGH
            if (i < decoder->num_of_outputs()) {
                try {
                    ps = m_decoder->get_output_shape(i);
                } catch (...) {
                    // nothing, means the info cannot be queried and remains unknown
                }
                // FIXME: ROUGH
                try {
                    type = m_decoder->get_output_type(i);
                } catch (std::runtime_error& e) {
                    // nothing, means the info cannot be queried and remains unknown
                    std::cerr << "[ ERROR ] Cannot retrieve type\n" << e.what() << std::endl;
                }
            } else {
                // std::cerr << "[ WARNING ] Cannot retrieve type for output not existent in pt node: "
                //          << m_decoder->get_op_type() << " with 0 input: " << m_decoder->input(0) << std::endl;
            }
            // Let's see what type we have
            // std::cout << "Can be represented as element::Type: " << type.is<element::Type>() << std::endl;
            // std::cout << "element::Type value: " << type.as<element::Type>() << "\n";
            // std::exit(0);
            set_custom_output_type(i, type, ps);
        }
    }

    PtFrameworkNode(const std::shared_ptr<Decoder>& decoder, const OutputVector& inputs)
        : PtFrameworkNode(decoder, inputs, decoder->num_of_outputs()) {}

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        auto op = std::make_shared<PtFrameworkNode>(m_decoder, inputs, get_output_size());

        for (auto body_index = 0; body_index < m_bodies.size(); ++body_index) {
            op->set_function(body_index, clone_model(*get_function(body_index)));
            for (const auto& m_input_descr : m_input_descriptions[body_index]) {
                op->m_input_descriptions[body_index].push_back(m_input_descr->copy());
            }
            for (const auto& m_output_descr : m_output_descriptions[body_index]) {
                op->m_output_descriptions[body_index].push_back(m_output_descr->copy());
            }
        }
        op->validate_and_infer_types();

        return op;
    }

    std::string get_op_type() const {
        return m_decoder->get_op_type();
    }

    Decoder* get_decoder() const {
        return m_decoder.get();
    }

    bool visit_attributes(AttributeVisitor& visitor) override {
        bool parent_visit_result = FrameworkNode::visit_attributes(visitor);
        // TODO: serialize bodies and descriptors
        /*for (size_t i = 0; i < m_bodies.size(); ++i) {
            //visitor.on_attribute("body", m_bodies[i]);
            //visitor.on_attribute("input_descriptions", m_input_descriptions[i]);
            //visitor.on_attribute("output_descriptions", m_output_descriptions[i]);
        }*/
        return parent_visit_result;
    }

    void validate_and_infer_types() override {
        for (int i = 0; i < m_bodies.size(); i++) {
            // Input
            for (const auto& input_description : m_input_descriptions[i]) {
                auto index = input_description->m_input_index;
                if (auto invariant_input_description =
                        ov::as_type_ptr<op::v0::TensorIterator::InvariantInputDescription>(input_description)) {
                    auto body_parameter =
                        m_bodies[i]->get_parameters().at(invariant_input_description->m_body_parameter_index);

                    auto body_param_partial_shape = body_parameter->get_partial_shape();
                    auto input_partial_shape = input(index).get_partial_shape();

                    body_parameter->set_partial_shape(input_partial_shape);
                }
            }

            // Body
            m_bodies[i]->validate_nodes_and_infer_types();

            // Output
            for (const auto& output_description : m_output_descriptions[i]) {
                auto index = output_description->m_output_index;

                auto body_value = m_bodies[i]->get_results().at(output_description->m_body_value_index)->input_value(0);

                if (auto body_output_description =
                        ov::as_type_ptr<op::v0::TensorIterator::BodyOutputDescription>(output_description)) {
                    const ov::PartialShape& ps = body_value.get_partial_shape();
                    set_output_type(index, body_value.get_element_type(), ps);
                }
            }
        }
    }

private:
    std::shared_ptr<Decoder> m_decoder;
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
