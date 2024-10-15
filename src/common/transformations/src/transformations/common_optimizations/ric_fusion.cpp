// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/ric_fusion.hpp"

#include <memory>
#include <utility>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/pad_base.hpp"
#include "openvino/pass/backward_graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/log.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace pass {
namespace ric_attr {

namespace {
std::shared_ptr<ov::op::v0::Constant> create_1d_const(const std::vector<int64_t>& values) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}
}  // namespace

// Attribute describes RIC type which we propagate.
// Also, it contains callback which can expand this attribute to the real RIC sub-graph.
// In addition, attribute has some functionality and properties for propagation.
class Attribute {
public:
    Attribute(std::vector<int64_t> order, int64_t axis, bool is_final = false, bool is_initial = false)
        : m_order(std::move(order)),
          m_axis(axis),
          m_is_final(is_final),
          m_is_initial(is_initial) {
        m_can_be_fused.emplace_back(std::make_shared<bool>(true));
    }

    // Method which is used to create a copy of attribute for further propagation.
    // TODO: can be removed and replaced with regular copy but we need to get rid of
    //       is_initial flag and use some other way to detect original RIC output.
    Attribute propagate() const {
        Attribute attr(m_order, m_axis);
        attr.m_can_be_fused = m_can_be_fused;
        return attr;
    }

    void set_is_final(bool is_final) {
        m_is_final = is_final;
    }

    void set_can_be_fused(bool can_be_fused) {
        std::for_each(m_can_be_fused.cbegin(),
                      m_can_be_fused.cend(),
                      [can_be_fused](const std::shared_ptr<bool>& state) {
                          *state = can_be_fused;
                      });
    }

    // Apply callback to materialize RIC inside graph
    void materialize(Input<Node> input, const ov::NodeVector& nodes) const {
        const auto& input_pshape = input.get_partial_shape();
        const auto input_rank = input_pshape.rank();
        if (input_rank.is_dynamic()) {
            OPENVINO_DEBUG("Axis calculated to materialize RIC on input: input rank is dynamic");
            return;
        }
        const auto axis = get_axis();
        // Despite of m_axis is signed integer this transformartion does not handle negative axes values
        if (axis < 0 || axis >= static_cast<int64_t>(input_pshape.size())) {
            OPENVINO_DEBUG("Axis calculated to materialize RIC on input: ", input, " is out of range");
            return;
        }
        const auto& axis_dim = input_pshape[axis];
        if (axis_dim.is_dynamic()) {
            OPENVINO_DEBUG("Axis calculated to materialize RIC on input: ", input, " is dynamic");
            return;
        }
        auto output = input.get_source_output();
        // Handle case when the RIC order is default
        auto order = get_order();
        if (order.empty()) {
            order.resize(axis_dim.get_length());
            std::iota(order.rbegin(), order.rend(), 0);
        }
        auto gather =
            std::make_shared<ov::op::v8::Gather>(output, create_1d_const(order), create_1d_const({get_axis()}));
        input.replace_source_output(gather);
        ov::copy_runtime_info(nodes, gather);
    }

    bool can_be_fused() const {
        return std::all_of(m_can_be_fused.cbegin(), m_can_be_fused.cend(), [](const std::shared_ptr<bool>& state) {
            return *state;
        });
    }

    // For cases when we propagate through operation with multiple inputs like Eltwise
    // we have to merge RIC attrs from all inputs. To check that given attr be merged with
    // current we check the order and axis which must be the same.
    bool can_be_merged_with(const Attribute& other) {
        return (m_order.empty() || other.m_order.empty() || m_order == other.m_order) && m_axis == other.m_axis;
    }

    // When merging two and more attrs for further propagation we have to keep can_be_fused references
    // for cases when fusion is not possible, so we can update all related attrs.
    void merge_with(const Attribute& other) {
        m_can_be_fused.insert(m_can_be_fused.end(), other.m_can_be_fused.begin(), other.m_can_be_fused.end());
    }

    const std::vector<int64_t>& get_order() const {
        return m_order;
    }

    void set_order(const std::vector<int64_t>& order) {
        m_order = order;
    }

    int64_t get_axis() const {
        return m_axis;
    }

    void set_axis(int64_t axis) {
        m_axis = axis;
    }

    bool is_final() const {
        return m_is_final;
    }

    bool is_initial() const {
        return m_is_initial;
    }

private:
    // empty order means that the order is default and must be n, n-1, ..., 0
    // according to the dimension values specified by m_axis
    std::vector<int64_t> m_order;
    int64_t m_axis;

    // Specifies whether RIC can be fused or not. vector is needed to keep references to other
    // attributes that were participated during merge.
    std::vector<std::shared_ptr<bool>> m_can_be_fused;

    // true - means that current RIC attribute is final and can be materialized
    // false - means that current RIC attribute is temporary and need only for propagation
    bool m_is_final;

    // true - means that current RIC attribute is an initial attribute and belongs to real RIC output
    // false - means that current RIC attribute is temporary and need only for propagation
    bool m_is_initial;
};

namespace {

template <typename T>
using is_port = typename std::enable_if<!std::is_convertible<T, std::shared_ptr<Node>>::value>::type;

template <typename T, typename = is_port<T>>
void set(T port, const Attribute& ric_attr) {
    auto& attrs = port.get_rt_info();
    attrs["reverse_input_channel_index"] = ric_attr;
}

// Available only for output ports
void init(Output<Node> output, std::vector<int64_t> order, int64_t axis) {
    set(output, Attribute(std::move(order), axis, false, true));
}

template <typename T, typename = is_port<T>>
bool has(const T& port) {
    const auto& attrs = port.get_rt_info();
    return attrs.count("reverse_input_channel_index");
}

template <typename T, typename = is_port<T>>
Attribute get(const T& port) {
    const auto& attrs = port.get_rt_info();
    auto res = attrs.find("reverse_input_channel_index");
    if (res != attrs.end()) {
        return res->second.template as<Attribute>();
    }
    OPENVINO_THROW("reverse_input_channel_index is missing in given port");
}

template <typename T, typename = is_port<T>>
void erase(T port) {
    auto& rt_info = port.get_rt_info();
    rt_info.erase("reverse_input_channel_index");
}
}  // namespace
}  // namespace ric_attr

namespace init {

namespace {

void add_node_with_inputs_to_vector(const std::shared_ptr<ov::Node>& node, NodeVector& vector) {
    vector.push_back(node);
    const auto& inputs = node->inputs();
    for (const auto& input : inputs) {
        vector.push_back(input.get_source_output().get_node_shared_ptr());
    }
}

}  // namespace
class SplitConcat : public ov::pass::MatcherPass {
public:
    SplitConcat(NodeVector& nodes_to_fuse) {
        MATCHER_SCOPE(SplitConcat);
        auto split_p = pattern::wrap_type<ov::op::v1::Split>();
        auto pattern_root = pattern::wrap_type<ov::op::v0::Concat>({split_p, split_p, split_p});

        auto callback = [=, &nodes_to_fuse](pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            auto concat = ov::as_type_ptr<ov::op::v0::Concat>(pattern_map.at(pattern_root).get_node_shared_ptr());
            auto split = ov::as_type_ptr<ov::op::v1::Split>(pattern_map.at(split_p).get_node_shared_ptr());
            if (!concat || !split)
                return false;

            // Avoid cases with two consecutive Split->Concat
            if (ric_attr::has(split->input_value(0))) {
                return false;
            }

            std::vector<int64_t> order;
            order.reserve(split->get_num_splits());

            for (const auto& input : concat->inputs()) {
                auto split_output = input.get_source_output();
                if (split_output.get_node() != split.get())
                    return false;

                // Check that Concat is the only Split consumer and order of Split outputs
                // satisfies expected order for reverse input channel case.
                for (const auto& target_input : split_output.get_target_inputs()) {
                    if (target_input.get_node() != concat.get()) {
                        return false;
                    }
                    order.emplace_back(split_output.get_index());
                }
            }

            // Check that all order values are unique, otherwise it is not RIC
            std::set<int64_t> unique_values(order.cbegin(), order.cend());
            if (unique_values.size() != order.size()) {
                return false;
            }

            // Mark-up RIC output
            ric_attr::init(concat, order, concat->get_axis());

            nodes_to_fuse.push_back(concat);
            add_node_with_inputs_to_vector(split, nodes_to_fuse);
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class Gather : public ov::pass::MatcherPass {
public:
    Gather(NodeVector& nodes_to_fuse) {
        MATCHER_SCOPE(Gather);
        auto input_p = pattern::any_input(pattern::has_static_rank());
        auto indices_p = pattern::any_input();
        auto axis_p = pattern::wrap_type<ov::op::v0::Constant>();
        auto pattern_root = pattern::wrap_type<ov::op::v8::Gather>({input_p, indices_p, axis_p});

        auto callback = [=, &nodes_to_fuse](pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            const auto& output = pattern_map.at(pattern_root);

            auto axis = ov::util::get_constant_from_source(pattern_map.at(axis_p));
            if (!axis)
                return false;

            const auto axis_value = axis->cast_vector<int64_t>().at(0);
            auto gather = output.get_node_shared_ptr();
            if (ov::is_preprocesing_node(gather)) {
                ric_attr::init(output, {}, axis_value);
                add_node_with_inputs_to_vector(gather, nodes_to_fuse);
                return true;
            }

            auto order = ov::util::get_constant_from_source(pattern_map.at(indices_p));
            if (!order)
                return false;

            // Avoid cases with two consecutive Gathers
            if (ric_attr::has(pattern_map.at(input_p))) {
                return false;
            }

            // This constraint helps to avoid detection of other Gathers that do not perform RIC
            const auto& data_shape = m.get_match_root()->input(0).get_partial_shape();
            if (shape_size(order->get_shape()) == 1 || axis_value < 0 || axis_value >= data_shape.rank().get_length() ||
                data_shape[axis_value].is_dynamic() ||
                shape_size(order->get_shape()) != static_cast<size_t>(data_shape[axis_value].get_length())) {
                return false;
            }

            // Check that all order values are unique, otherwise it is not RIC
            const auto& order_values = order->cast_vector<int64_t>();
            std::set<int64_t> unique_values(order_values.cbegin(), order_values.cend());
            if (unique_values.size() != order_values.size()) {
                return false;
            }
            ric_attr::init(output, order_values, axis_value);
            add_node_with_inputs_to_vector(gather, nodes_to_fuse);
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};
}  // namespace init

namespace prop {

class Binary : public ov::pass::MatcherPass {
public:
    Binary() {
        MATCHER_SCOPE(Binary);
        auto pattern_root = pattern::wrap_type<op::util::BinaryElementwiseArithmetic, ov::op::v0::FakeQuantize>();

        auto callback = [=](pattern::Matcher& m) {
            const auto& root = m.get_match_root();
            const auto& inputs = root->inputs();

            std::map<size_t, ric_attr::Attribute> attrs;
            for (const auto& input : inputs) {
                auto output = input.get_source_output();
                if (ric_attr::has(output)) {
                    attrs.insert({input.get_index(), ric_attr::get(output).propagate()});
                } else if (!ov::is_type<ov::op::v0::Constant>(output.get_node())) {
                    // If number of non-constant inputs and without RIC attr is greater than 0 we have to skip
                    // propagation because it is not efficient to have a lot of RIC copies on data path.
                    return false;
                }
            }

            if (attrs.empty())
                return false;

            // Check that all RIC attrs can be merged and then merge them
            auto ric = attrs.begin()->second;
            auto rank = root->get_input_partial_shape(attrs.begin()->first).rank();
            if (rank.is_dynamic())
                return false;
            auto data_rank = rank.get_length();

            for (const auto& item : attrs) {
                const auto& input_rank = root->get_input_partial_shape(item.first).rank();
                if (input_rank.is_static() && input_rank.get_length() == data_rank &&
                    ric.can_be_merged_with(item.second)) {
                    ric.merge_with(item.second);
                } else {
                    return false;
                }
            }

            for (const auto& input : inputs) {
                // Skip input that have RIC attribute
                if (attrs.count(input.get_index()))
                    continue;

                auto const_output = input.get_source_output();
                const auto& shape = const_output.get_shape();
                const int64_t& shape_rank = static_cast<int64_t>(shape.size());
                if (shape_rank > data_rank) {
                    // TODO: handle case when constant input broadcast another one
                    return false;
                }

                if (data_rank - shape_rank > ric.get_axis()) {
                    // we don't have to insert RIC for constant, so we keep propagating
                    ric_attr::set(m.get_match_value(), ric);
                    continue;
                }

                const int64_t& new_axis = ric.get_axis() - (data_rank - shape_rank);
                const auto& axis_dim = shape[new_axis];
                if (axis_dim == 1) {
                    // we don't have to insert RIC for constant, so we keep propagating
                    ric_attr::set(m.get_match_value(), ric);
                    continue;
                }

                // finally, insert RIC
                auto ric_const = ric;
                ric_const.set_axis(new_axis);
                ric_const.set_is_final(true);
                ric_attr::set(input, ric_const);
            }

            ric_attr::set(m.get_match_value(), ric);
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class Convolution : public ov::pass::MatcherPass {
public:
    Convolution() {
        MATCHER_SCOPE(Convolution);
        auto input_p = pattern::any_input(ric_attr::has<Output<Node>>);
        auto pattern_root = pattern::wrap_type<ov::op::v1::Convolution>(
            {input_p, pattern::any_input(pattern::has_static_dim(1 /*output channel*/))});
        auto callback = [=](pattern::Matcher& m) {
            auto conv = m.get_match_root();
            auto ric = ric_attr::get(conv->input_value(0)).propagate();
            if (ric.get_axis() != 1)
                return false;

            ric_attr::set(conv->input(1), ric);
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class GroupConvolution : public ov::pass::MatcherPass {
public:
    GroupConvolution() {
        MATCHER_SCOPE(GroupConvolution);
        auto input_p = pattern::any_input(ric_attr::has<Output<Node>>);
        auto pattern_root = pattern::wrap_type<ov::op::v1::GroupConvolution>(
            {input_p, pattern::any_input(pattern::has_static_shape())});

        auto callback = [=](pattern::Matcher& m) {
            auto conv = m.get_match_root();
            const auto& weights_shape = conv->input_value(1).get_shape();
            const int64_t& group = static_cast<int64_t>(weights_shape.at(0));
            const int64_t& channels = static_cast<int64_t>(weights_shape.at(1));
            const int64_t& in_channels = static_cast<int64_t>(weights_shape.at(2));

            auto ric = ric_attr::get(conv->input_value(0)).propagate();
            auto order = ric.get_order();
            // Handle case when the RIC order is default
            if (order.empty()) {
                order.resize(group);
                std::iota(order.rbegin(), order.rend(), 0);
                ric.set_order(order);
            }

            if (in_channels != 1 || ric.get_order().size() != static_cast<size_t>(group) || ric.get_axis() != 1) {
                // TODO: insert RIC when group == 1
                return false;
            }

            // Update weights with RIC attribute
            auto ric_weights = ric;
            ric_weights.set_axis(0);

            ric_attr::set(conv->input(1), ric_weights);

            // Calculate new order for RIC propagation
            const int64_t output_channels = group * channels;
            std::vector<int64_t> new_order;
            new_order.reserve(output_channels);
            for (const auto& index : ric.get_order()) {
                for (int64_t pos = index * channels, i = 0; i < channels; ++i, ++pos) {
                    new_order.emplace_back(pos);
                }
            }
            assert(new_order.size() == static_cast<size_t>(output_channels));

            ric.set_order(new_order);
            ric_attr::set(conv->output(0), ric);
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class ShapeOf : public ov::pass::MatcherPass {
public:
    ShapeOf() {
        MATCHER_SCOPE(ShapeOf);
        auto pattern_root = pattern::wrap_type<ov::op::v0::ShapeOf, ov::op::v3::ShapeOf>();

        auto callback = [=](pattern::Matcher& m) {
            // Skip propagation for ShapeOf path
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class PassThrough : public ov::pass::MatcherPass {
public:
    PassThrough() {
        MATCHER_SCOPE(PassThrough);
        auto pattern_root = pattern::wrap_type<op::util::UnaryElementwiseArithmetic,
                                               ov::op::v0::Convert,
                                               op::util::PadBase,
                                               ov::op::v0::PRelu>();

        auto callback = [=](pattern::Matcher& m) {
            auto root = m.get_match_root();
            if (!ric_attr::has(root->input_value(0)))
                return false;
            ric_attr::set(root->output(0), ric_attr::get(root->input_value(0)).propagate());
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class Transpose : public ov::pass::MatcherPass {
public:
    Transpose() {
        MATCHER_SCOPE(Transpose);
        auto input_p = pattern::any_input(ric_attr::has<Output<Node>>);
        auto order_p = pattern::wrap_type<ov::op::v0::Constant>();
        auto pattern_root = pattern::wrap_type<ov::op::v1::Transpose>({input_p, order_p});

        auto callback = [=](pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            auto input = pattern_map.at(input_p);
            auto ric = ric_attr::get(input).propagate();

            auto order_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(order_p).get_node_shared_ptr());
            auto order = order_node->cast_vector<int64_t>();

            int64_t new_axis = std::find(order.begin(), order.end(), ric.get_axis()) - order.begin();
            ric.set_axis(new_axis);

            ric_attr::set(m.get_match_value(), ric);
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class Unsupported : public ov::pass::MatcherPass {
public:
    Unsupported() {
        MATCHER_SCOPE(Unsupported);
        auto pattern_root = pattern::any_input();
        auto callback = [=](pattern::Matcher& m) {
            for (const auto& input : m.get_match_root()->input_values()) {
                if (ric_attr::has(input)) {
                    auto ric = ric_attr::get(input);
                    if (ric.is_final()) {
                        continue;
                    }
                    ric.set_can_be_fused(false);
                    OPENVINO_DEBUG("Node is unsupported by RIC Fusion: ", *m.get_match_root(), "\n");
                }
            }
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};
}  // namespace prop

namespace fuse {
namespace {
bool need_to_erase_ric(const Output<Node>& output) {
    if (!ric_attr::has(output))
        return false;
    const auto& ric = ric_attr::get(output);
    return ric.can_be_fused() && ric.is_initial();
}
}  // namespace

class InsertReverseInputChannel : public ov::pass::MatcherPass {
public:
    InsertReverseInputChannel(NodeVector& fused_nodes) {
        MATCHER_SCOPE(InsertReverseInputChannel);
        auto pattern_root = pattern::any_input();
        auto callback = [&fused_nodes](pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            for (const auto& input : node->inputs()) {
                if (!ric_attr::has(input))
                    continue;
                const auto& ric = ric_attr::get(input);
                if (ric.can_be_fused() && ric.is_final()) {
                    ric.materialize(input, fused_nodes);
                }
            }
            return false;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class EraseSplitConcat : public ov::pass::MatcherPass {
public:
    EraseSplitConcat() {
        MATCHER_SCOPE(EraseSplitConcat);
        auto input_p = pattern::any_input();
        auto split_p = pattern::wrap_type<ov::op::v1::Split>({input_p, pattern::any_input()});
        auto pattern_root = pattern::wrap_type<ov::op::v0::Concat>({split_p, split_p, split_p}, need_to_erase_ric);

        auto callback = [=](pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            auto output = pattern_map.at(pattern_root);
            auto input = pattern_map.at(input_p);
            output.replace(input);
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class EraseGather : public ov::pass::MatcherPass {
public:
    EraseGather() {
        MATCHER_SCOPE(EraseGather);
        auto input_p = pattern::any_input();
        auto pattern_root =
            pattern::wrap_type<ov::op::v8::Gather>({input_p, pattern::any_input(), pattern::any_input()},
                                                   need_to_erase_ric);
        auto callback = [=](pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            auto output = pattern_map.at(pattern_root);
            auto input = pattern_map.at(input_p);
            output.replace(input);
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};
}  // namespace fuse

namespace back_prop {
class Binary : public ov::pass::MatcherPass {
public:
    Binary() {
        MATCHER_SCOPE(Binary);
        auto fake_quantize_pattern =
            pattern::wrap_type<ov::op::v0::FakeQuantize>({pattern::any_input(pattern::has_static_rank()),
                                                          pattern::any_input(pattern::has_static_rank()),
                                                          pattern::any_input(pattern::has_static_rank()),
                                                          pattern::any_input(pattern::has_static_rank()),
                                                          pattern::any_input(pattern::has_static_rank())},
                                                         pattern::has_static_rank());
        auto binary_elementwise_pattern = pattern::wrap_type<op::util::BinaryElementwiseArithmetic>(
            {pattern::any_input(pattern::has_static_rank()), pattern::any_input(pattern::has_static_rank())},
            pattern::has_static_rank());

        auto pattern_root =
            std::make_shared<pattern::op::Or>(OutputVector{fake_quantize_pattern, binary_elementwise_pattern});

        auto callback = [=](pattern::Matcher& m) {
            const auto& root = m.get_match_root();
            const auto& output = root->output(0);
            auto inputs = output.get_target_inputs();

            // Check if an output of matched root is consumed as input labeled with reverse_input_channel_index
            std::vector<ric_attr::Attribute> attrs;
            for (const auto& input : inputs) {
                if (ric_attr::has(input)) {
                    attrs.push_back(ric_attr::get(input).propagate());
                } else {
                    return false;
                }
            }

            if (attrs.empty())
                return false;

            // Check that all RIC attrs from consumers can be merged and then merge them
            auto ric = attrs[0];
            for (const auto& item : attrs) {
                if (ric.can_be_merged_with(item)) {
                    ric.merge_with(item);
                } else {
                    return false;
                }
            }

            auto data_rank = root->get_output_partial_shape(0).rank().get_length();
            for (const auto& input : root->inputs()) {
                auto output = input.get_source_output();
                const auto& shape = output.get_partial_shape();
                const int64_t& shape_rank = shape.rank().get_length();
                if (shape_rank > data_rank) {
                    // TODO: handle case when constant input broadcast another one
                    return false;
                }

                if (data_rank - shape_rank > ric.get_axis()) {
                    // we don't have to insert RIC for constant, so we keep propagating
                    continue;
                }

                const int64_t& new_axis = ric.get_axis() - (data_rank - shape_rank);
                const auto& axis_dim = shape[new_axis];
                if (axis_dim.is_dynamic())
                    return false;
                if (axis_dim == 1) {
                    // we don't have to insert RIC, because the channel dimension is 1
                    continue;
                }

                // finally, insert RIC
                auto ric_const = ric;
                ric_const.set_axis(new_axis);
                ric_attr::set(input, ric_const);
            }
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class ConvertPassThrough : public ov::pass::MatcherPass {
public:
    ConvertPassThrough() {
        MATCHER_SCOPE(ConvertPassThrough);
        auto pattern_root = pattern::wrap_type<ov::op::v0::Convert>(pattern::has_static_rank());
        auto callback = [=](pattern::Matcher& m) {
            auto root = m.get_match_root();
            const auto& output = root->output(0);
            auto consumers = output.get_target_inputs();
            std::vector<ric_attr::Attribute> attrs;

            for (const auto& consumer : consumers) {
                if (ric_attr::has(consumer)) {
                    attrs.push_back(ric_attr::get(consumer).propagate());
                } else {
                    return false;
                }
            }

            auto ric = attrs[0];
            auto data_rank = root->get_output_partial_shape(0).rank().get_length();

            for (const auto& item : attrs) {
                if (ric.can_be_merged_with(item)) {
                    ric.merge_with(item);
                } else {
                    return false;
                }
            }
            auto input = root->input(0);
            auto const_output = input.get_source_output();
            const auto& shape = const_output.get_partial_shape();
            if (shape.rank().is_dynamic())
                return false;

            const int64_t& shape_rank = shape.rank().get_length();
            const int64_t& new_axis = ric.get_axis() - (data_rank - shape_rank);

            // finally, insert RIC
            ric.set_axis(new_axis);
            ric_attr::set(input, ric);
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pattern_root, matcher_name);
        register_matcher(m, callback);
    }
};

class Constant : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("Constant", "0");
    Constant() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        RUN_ON_FUNCTION_SCOPE(Constant);
        for (const auto& node : model->get_ordered_ops()) {
            if ((std::dynamic_pointer_cast<op::util::BinaryElementwiseArithmetic>(node) ||
                 ov::as_type_ptr<ov::op::v0::FakeQuantize>(node) || ov::as_type_ptr<ov::op::v0::Convert>(node)) &&
                node->get_output_partial_shape(0).rank().is_static()) {
                continue;
            }
            for (const auto& output : node->outputs()) {
                for (const auto& consumer : output.get_target_inputs()) {
                    if (ric_attr::has(consumer)) {
                        auto ric = ric_attr::get(consumer);
                        if (ov::as_type_ptr<ov::op::v0::Constant>(node)) {
                            ric.set_is_final(true);
                            ric_attr::set(consumer, ric);
                        } else {  // Unsupported
                            if (!ric.is_final()) {
                                ric.set_can_be_fused(false);
                            }
                        }
                    }
                }
            }
        }
        return true;
    }
};

}  // namespace back_prop

bool ov::pass::ReverseInputChannelsFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(ReverseInputChannelsFusion);

    NodeVector nodes_to_fuse;
    // First we need to initialize and propagate RIC attributes through entire graph
    {
        using namespace init;
        Manager m("ReverseInputChannelsFusion");
        m.set_per_pass_validation(false);
        auto ric_init = m.register_pass<GraphRewrite>();
        ADD_MATCHER(ric_init, SplitConcat, nodes_to_fuse)
        ADD_MATCHER(ric_init, Gather, nodes_to_fuse)
        if (!m.run_passes(model)) {
            return false;
        }
    }

    Manager m;
    m.set_per_pass_validation(false);

    auto ric_prop = m.register_pass<GraphRewrite>();
    {
        using namespace prop;
        ADD_MATCHER(ric_prop, Convolution)
        ADD_MATCHER(ric_prop, GroupConvolution)
        ADD_MATCHER(ric_prop, Binary)
        ADD_MATCHER(ric_prop, ShapeOf)
        ADD_MATCHER(ric_prop, Transpose)
        ADD_MATCHER(ric_prop, PassThrough)
        ADD_MATCHER(ric_prop, Unsupported)
    }

    // Handle quantized weights case (dequantize sub-graph is on the weights path)
    auto ric_back_prop = m.register_pass<ov::pass::BackwardGraphRewrite>();
    {
        using namespace back_prop;
        ADD_MATCHER(ric_back_prop, Binary)
        ADD_MATCHER(ric_back_prop, ConvertPassThrough)
        REGISTER_PASS(m, Constant)
    }
    // TODO: validate attributes by request

    // Second we fuse available RIC into nodes and remove original nodes related to fused RIC
    auto ric_fuse = m.register_pass<GraphRewrite>();
    {
        using namespace fuse;
        ADD_MATCHER(ric_fuse, InsertReverseInputChannel, nodes_to_fuse)
        ADD_MATCHER(ric_fuse, EraseSplitConcat)
        ADD_MATCHER(ric_fuse, EraseGather)
    }

    m.run_passes(model);
    return false;
}
}  // namespace pass
}  // namespace ov
