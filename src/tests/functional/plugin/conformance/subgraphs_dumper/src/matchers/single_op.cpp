 // Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/single_op.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/validation_util.hpp"
#include <cstdlib>

using namespace SubgraphsDumper;

template<typename dType>
bool compare_constants_data(const std::shared_ptr<ov::op::v0::Constant> &op,
                            const std::shared_ptr<ov::op::v0::Constant> &ref) {
    size_t elements_count = ov::shape_size(op->get_shape());
    if (elements_count != ov::shape_size(ref->get_shape())) {
        return false;
    }
    const auto &op_data = op->cast_vector<dType>();
    const auto &ref_data = ref->cast_vector<dType>();
    for (size_t i = 0; i < elements_count; ++i) {
        // std:abs doesn't implemented for unsigned types, compare explicitly to keep code universal for all dTypes
        dType diff = op_data[i] > ref_data[i] ? op_data[i] - ref_data[i] : ref_data[i] - op_data[i];
        if (diff > std::numeric_limits<dType>::epsilon()) {
            return false;
        }
    }
    return true;
}

// TODO: Move to some utils?
bool compare_constants_data(const std::shared_ptr<ov::op::v0::Constant> &op,
                            const std::shared_ptr<ov::op::v0::Constant> &ref) {
    switch (op->get_element_type()) {
        case ov::element::Type_t::boolean:
            return compare_constants_data<bool>(op, ref);
        case ov::element::Type_t::bf16:
            return compare_constants_data<ov::bfloat16>(op, ref);
        case ov::element::Type_t::f16:
            return compare_constants_data<ov::float16>(op, ref);
        case ov::element::Type_t::f32:
            return compare_constants_data<float>(op, ref);
        case ov::element::Type_t::f64:
            return compare_constants_data<double>(op, ref);
        case ov::element::Type_t::i8:
            return compare_constants_data<int8_t>(op, ref);
        case ov::element::Type_t::i16:
            return compare_constants_data<int16_t>(op, ref);
        case ov::element::Type_t::i32:
            return compare_constants_data<int32_t>(op, ref);
        case ov::element::Type_t::i64:
            return compare_constants_data<int64_t>(op, ref);
            // TODO cast_vector doesn't support u1 now
//        case ov::element::Type_t::u1:
//            return compare_constants_data<char>(op, ref);
        case ov::element::Type_t::u8:
            return compare_constants_data<uint8_t>(op, ref);
        case ov::element::Type_t::u16:
            return compare_constants_data<uint16_t>(op, ref);
        case ov::element::Type_t::u32:
            return compare_constants_data<uint32_t>(op, ref);
        case ov::element::Type_t::u64:
            return compare_constants_data<uint64_t>(op, ref);
        default:
            std::cout << "Can't compare constants" << op << " with " << ref << "\n" << "Unsupported data type";
            return false;
    }
}

bool SingleOpMatcher::same_op_type(const std::shared_ptr<ov::Node> &node,
                                   const std::shared_ptr<ov::Node> &ref,
                                   const LayerTestsUtils::OPInfo &op_info) const {
    return node->get_type_info() == ref->get_type_info();
}

bool SingleOpMatcher::match_inputs(const std::shared_ptr<ov::Node> &node,
                                   const std::shared_ptr<ov::Node> &ref,
                                   const LayerTestsUtils::OPInfo &op_info) const {
    if (node->get_input_size() != ref->get_input_size()) {
        return false;
    }
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        bool rankIsEqual = node->get_input_tensor(i).get_partial_shape().rank() ==
                           ref->get_input_tensor(i).get_partial_shape().rank();
        bool elemTypeIsEqual = node->get_input_tensor(i).get_element_type() ==
                               ref->get_input_tensor(i).get_element_type();
        bool dynamismIsEqual = node->get_input_partial_shape(i).is_dynamic() ==
                               ref->get_input_partial_shape(i).is_dynamic();
        if (!rankIsEqual || !elemTypeIsEqual || !dynamismIsEqual) {
            return false;
        }
    }

    return true;
}

bool
SingleOpMatcher::match_outputs(const std::shared_ptr<ov::Node> &node,
                               const std::shared_ptr<ov::Node> &ref,
                               const LayerTestsUtils::OPInfo &op_info) const {
    if (node->get_output_size() != ref->get_output_size()) {
        return false;
    }
    // Match output element type, shape rank & dynamism
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        if (node->get_output_tensor(i).get_element_type() !=
            ref->get_output_tensor(i).get_element_type()) {
            return false;
        }
        if (node->get_output_tensor(i).get_partial_shape().is_dynamic() !=
            ref->get_output_tensor(i).get_partial_shape().is_dynamic()) {
            return false;
        }
        if (node->get_output_tensor(i).get_partial_shape().rank()!=
            ref->get_output_tensor(i).get_partial_shape().rank()) {
            return false;
        }
    }

    return true;
}

bool SingleOpMatcher::same_attrs(const std::shared_ptr<ov::Node> &node,
                                 const std::shared_ptr<ov::Node> &ref,
                                 const LayerTestsUtils::OPInfo &op_info) const {
    return attributes::compare(node.get(), ref.get(), Comparator::CmpValues::ATTRIBUTES).valid;
}

bool SingleOpMatcher::match_ports(const std::shared_ptr<ov::Node> &node,
                                  const std::shared_ptr<ov::Node> &ref,
                                  const LayerTestsUtils::OPInfo &op_info) const {
    const auto &cfg = get_config(node);
    const std::vector<size_t> &ignored_ports = cfg->ignored_ports;

    for (size_t port_id = 0; port_id < node->get_input_size(); ++port_id) {
        if (std::any_of(begin(ignored_ports), end(ignored_ports), [=](size_t p) { return p == port_id; })) {
            continue;
        }
        const auto &cur_node_input = node->input_value(port_id);
        const auto &ref_node_input = ref->input_value(port_id);

        const auto &cur_const_input = ov::get_constant_from_source(cur_node_input);
        const auto &ref_const_input = ov::get_constant_from_source(ref_node_input);

        // Check that both OP an reference port inputs are constant and have same data
        if (cur_const_input && ref_const_input &&
            !compare_constants_data(cur_const_input, ref_const_input)) {
            return false;
            // Check that input nodes on the port both not constants
        } else if ((cur_const_input && !ref_const_input) || (!cur_const_input && ref_const_input)) {
            return false;
        }
    }
    return true;
}

bool SingleOpMatcher::match(const std::shared_ptr<ov::Node> &node,
                            const std::shared_ptr<ov::Node> &ref,
                            const LayerTestsUtils::OPInfo &op_info) const {
    for (const auto& input_node : node->inputs()) {
        if (input_node.get_partial_shape().is_dynamic()) {
            break;
        }
    }
    const auto &cfg = get_config(node);
    if (match_only_configured_ops() && cfg->is_fallback_config) {
        return false;
    }
    if (cfg->ignore_matching) {
        return false;
    }
    return same_op_type(node, ref, op_info) &&
           match_inputs(node, ref, op_info) &&
           match_outputs(node, ref, op_info) &&
           same_attrs(node, ref, op_info) &&
           match_ports(node, ref, op_info);
}

SingleOpMatcher::SingleOpMatcher() {
    default_configs = {
            std::make_shared<MatcherConfig<>>(std::vector<std::string>{}, std::vector<size_t>{0}),
            std::make_shared<MatcherConfig<ov::opset8::FakeQuantize>>(std::vector<std::string>{},
                                                                      std::vector<size_t>{0, 1, 2, 3, 4}),
            std::make_shared<MatcherConfig<
                    ov::op::v0::MatMul,
                    ov::op::v1::Add,
                    ov::op::v1::Multiply,
                    ov::op::v1::Subtract,
                    ov::op::v1::Power>>(std::vector<std::string>{}, std::vector<size_t>{0, 1}),

            std::make_shared<MatcherConfig<
                    ov::op::v1::Convolution,
                    ov::op::v1::ConvolutionBackpropData,
                    ov::op::v1::GroupConvolution,
                    ov::op::v1::GroupConvolutionBackpropData>>(true)
    };
}
