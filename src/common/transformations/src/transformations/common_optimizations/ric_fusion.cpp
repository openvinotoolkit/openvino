// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/ric_fusion.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <utility>
#include <vector>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/op/util/binary_elementwise_arithmetic.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "itt.hpp"


NGRAPH_RTTI_DEFINITION(ngraph::pass::ReverseInputChannelsFusion, "ReverseInputChannelsFusion", 0);

namespace ngraph {
namespace pass {
namespace ric_attr {

class Attribute {
public:
    using callback_t = std::function<void(Input<Node>, const Attribute &)>;

    Attribute(std::vector<int64_t> order, int64_t axis, bool is_final = false, bool is_initial = false)
        : m_order(std::move(order)), m_axis(axis), m_is_final(is_final), m_is_initial(is_initial) {
        m_can_be_fused.emplace_back(std::make_shared<bool>(true));
    }

    Attribute propagate() const {
        Attribute attr(m_order, m_axis);
        attr.m_can_be_fused = m_can_be_fused;
        return attr;
    }

    bool operator==(const Attribute & other) const {
        return m_can_be_fused == other.m_can_be_fused &&
               m_is_final == other.m_is_final &&
               m_is_initial == other.m_is_initial &&
               m_order == other.m_order &&
               m_axis == other.m_axis;
    }

    void set_is_final(bool is_final) { m_is_final = is_final; }

    void set_can_be_fused(bool can_be_fused) {
        std::for_each(m_can_be_fused.cbegin(), m_can_be_fused.cend(),
                      [can_be_fused](const std::shared_ptr<bool> & state) {
               *state = can_be_fused;
        });
    }

    void set_callback(callback_t callback) {
        m_callback = std::move(callback);
    }

    void operator() (Input<Node> input) const {
        m_callback(input, *this);
    }

    bool can_be_fused() const {
        return std::all_of(m_can_be_fused.cbegin(), m_can_be_fused.cend(),
                           [](const std::shared_ptr<bool> & state) {
            return *state;
        });
    }

    bool can_be_merged_with(const Attribute & other) {
        return m_order == other.m_order && m_axis == other.m_axis;
    }

    void merge_with(const Attribute & other) {
        m_can_be_fused.insert(m_can_be_fused.end(),
                              other.m_can_be_fused.begin(),
                              other.m_can_be_fused.end());
    }

    const std::vector<int64_t> & get_order() const { return m_order; }

    void set_order(const std::vector<int64_t> & order) { m_order = order; }

    int64_t get_axis() const { return m_axis; }

    void set_axis(int64_t axis) { m_axis = axis; }

    bool is_final() const { return m_is_final; }

    bool is_initial() const { return m_is_initial; }

private:
    std::vector<int64_t> m_order;
    int64_t m_axis;

    std::vector<std::shared_ptr<bool>> m_can_be_fused;
    bool m_is_final;
    bool m_is_initial;

    std::function<void(Input<Node>, const Attribute &)> m_callback =
            [](Input<Node>, const Attribute &) {};
};

namespace {

template <typename T>
using is_port = typename std::enable_if<!std::is_convertible<T, std::shared_ptr<Node>>::value>::type;

template <typename T, typename = is_port<T>>
void set(T port, const Attribute & ric_attr) {
    auto & attrs = port.get_rt_info();
    attrs["reverse_input_channel_index"] = ric_attr;
}

// Available only for output ports
void init(Output<Node> output, std::vector<int64_t> order, int64_t axis) {
    set(output, Attribute(std::move(order), axis, false, true));
}

template <typename T, typename = is_port<T>>
bool has(const T & port) {
    const auto & attrs = port.get_rt_info();
    return attrs.count("reverse_input_channel_index");
}

template <typename T, typename = is_port<T>>
Attribute get(const T & port) {
    const auto & attrs = port.get_rt_info();
    auto res = attrs.find("reverse_input_channel_index");
    if (res != attrs.end()) {
        return res->second.template as<Attribute>();
    }
    throw ngraph_error("reverse_input_channel_index is missing in given port");
}

template <typename T, typename = is_port<T>>
void erase(T port) {
    auto & rt_info = port.get_rt_info();
    rt_info.erase("reverse_input_channel_index");
}
}// namespace
}// namespace ric_attr

namespace init {
// TODO: cover with tests
class SplitConcat : public ngraph::pass::MatcherPass {
public:
    SplitConcat() {
        MATCHER_SCOPE(SplitConcat);
        auto split_p = pattern::wrap_type<opset8::Split>();
        pattern_root = pattern::wrap_type<opset8::Concat>({split_p, split_p, split_p});

        callback = [=](pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_map();
            auto concat = ov::as_type_ptr<opset8::Concat>(pattern_map.at(pattern_root));
            auto split = ov::as_type_ptr<opset8::Split>(pattern_map.at(split_p));
            if (!concat || !split) return false;

            std::vector<int64_t> order;
            order.reserve(split->get_num_splits());

            for (const auto & input : concat->inputs()) {
                auto split_output = input.get_source_output();
                if (split_output.get_node() != split.get()) return false;

                // Check that Concat is the only Split consumer and order of Split outputs
                // satisfies expected order for reverse input channel case.
                for (const auto & target_input : split_output.get_target_inputs()) {
                    if (target_input.get_node() != concat.get()) {
                        return false;
                    }
                    order.emplace_back(split_output.get_index());
                }
            }

            // Mark-up RIC output
            ric_attr::init(concat, order, concat->get_axis());
            return true;
        };
    }
};

class Gather : public ngraph::pass::MatcherPass {
public:
    Gather() {
        MATCHER_SCOPE(Gather);
        auto indices_p = pattern::any_input();
        auto axis_p = pattern::wrap_type<opset8::Constant>();
        pattern_root = pattern::wrap_type<opset8::Gather>({pattern::any_input(), indices_p, axis_p});

        callback = [=](pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_map();
            const auto & output = pattern_map.at(pattern_root);

            auto order = ov::get_constant_from_source(pattern_map.at(indices_p));
            auto axis = ov::get_constant_from_source(pattern_map.at(axis_p));
            if (!order || !axis) {
                return false;
            }

            ric_attr::init(output, order->cast_vector<int64_t>(), axis->cast_vector<int64_t>().at(0));
            return true;
        };
    }
};
}// namespace init

namespace prop {
namespace {
std::shared_ptr<opset8::Constant> create_const(const std::vector<int64_t> & values) {
    return opset8::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}
}// namespace

class Convolution : public ngraph::pass::MatcherPass {
public:
    Convolution() {
        MATCHER_SCOPE(Convolution);
        auto input_p = pattern::any_input(ric_attr::has<Output<Node>>);
        pattern_root = pattern::wrap_type<opset8::Convolution>({input_p, pattern::any_input()});

        callback = [=](pattern::Matcher& m) {
            auto conv = m.get_match_root();
            auto ric = ric_attr::get(conv->input_value(0)).propagate();
            ric.set_is_final(true);
            ric.set_callback([](Input<Node> input, const ric_attr::Attribute & attr) {
                auto weights = input.get_source_output();
                auto gather = std::make_shared<opset8::Gather>(weights, create_const(attr.get_order()), create_const({1}));
                input.replace_source_output(gather);
                // TODO: copy runtime info from RIC sub-graph
            });
            ric_attr::set(conv->input(1), ric);
            return true;
        };
    }
};

class GroupConvolution : public ngraph::pass::MatcherPass {
public:
    GroupConvolution() {
        MATCHER_SCOPE(GroupConvolution);
        auto input_p = pattern::any_input(ric_attr::has<Output<Node>>);
        pattern_root = pattern::wrap_type<opset8::GroupConvolution>({input_p, pattern::any_input(pattern::has_static_shape())});

        callback = [=](pattern::Matcher& m) {
            auto conv = m.get_match_root();
            const auto & weights_shape = conv->input_value(1).get_shape();
            const int64_t & group = static_cast<int64_t>(weights_shape.at(0));
            const int64_t & channels = static_cast<int64_t>(weights_shape.at(1));

            auto ric = ric_attr::get(conv->input_value(0)).propagate();
            if (ric.get_order().size() != static_cast<size_t>(group)) {
                // TODO: insert RIC when group == 1
                return false;
            }

            const int64_t output_channels = group * channels;
            std::vector<int64_t> new_order;
            new_order.reserve(output_channels);
            for (const auto & index : ric.get_order()) {
                for (int64_t pos = index * channels, i = 0; i < channels; ++i, ++pos) {
                    new_order.emplace_back(pos);
                }
            }
            assert(new_order.size() == static_cast<size_t>(output_channels));

            ric.set_order(new_order);
            ric_attr::set(conv->output(0), ric);
            return true;
        };
    }
};

class Binary : public ngraph::pass::MatcherPass {
public:
    Binary() {
        MATCHER_SCOPE(Binary);
        auto input0_attr = pattern::any_input(ric_attr::has<Output<Node>>);
        auto input1_attr = pattern::any_input(ric_attr::has<Output<Node>>);
        auto attrs = pattern::wrap_type<op::util::BinaryElementwiseArithmetic>({input0_attr, input1_attr});

        // TODO: handle case when Constant has more than one consumer
        auto input1_const = pattern::wrap_type<opset8::Constant>(pattern::consumers_count(1));
        auto attr_with_const = pattern::wrap_type<op::util::BinaryElementwiseArithmetic>({input0_attr, input1_const});

        pattern_root = std::make_shared<pattern::op::Or>(OutputVector{attrs, attr_with_const});

        callback = [=](pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            const auto & input0 = pattern_map.at(input0_attr);
            auto ric = ric_attr::get(input0).propagate();
            if (pattern_map.count(input1_const)) {
                const auto & const_input = pattern_map.at(input1_const);
                const auto & shape = const_input.get_shape();
                const auto & data_shape = input0.get_partial_shape();
                const int64_t & shape_rank = static_cast<int64_t>(shape.size());
                if (data_shape.rank().is_dynamic() || shape_rank > data_shape.rank().get_length()) {
                    // TODO: handle case when constant input broadcast another one
                    return false;
                }
                const auto & data_rank = data_shape.rank().get_length();
                if (data_rank - shape_rank > ric.get_axis()) {
                    // we don't have to insert RIC for constant, so we keep propagating
                    ric_attr::set(m.get_match_value(), ric);
                    return true;
                }

                const int64_t & new_axis = ric.get_axis() - (data_rank - shape_rank);
                if (shape[new_axis] == 1) {
                    // we don't have to insert RIC for constant, so we keep propagating
                    ric_attr::set(m.get_match_value(), ric);
                    return true;
                }

                // finally, insert RIC
                auto ric_const = ric;
                ric_const.set_axis(new_axis);
                ric_const.set_is_final(true);
                ric_const.set_callback([](Input<Node> input, const ric_attr::Attribute & attr) {
                    auto output = input.get_source_output();
                    auto gather = std::make_shared<opset8::Gather>(output, create_const(attr.get_order()),
                                                                           create_const({attr.get_axis()}));
                    input.replace_source_output(gather);
                    // TODO: copy runtime info from RIC sub-graph
                });
                // TODO: find input for cases when Constant has multiple consumers
                ric_attr::set(*const_input.get_target_inputs().begin(), ric_const);
                ric_attr::set(m.get_match_value(), ric);
                return true;
            }

            auto ric1 = ric_attr::get(pattern_map.at(input1_attr)).propagate();
            if (!ric.can_be_merged_with(ric1)) {
                ric.set_can_be_fused(false);
                ric1.set_can_be_fused(false);
                return false;
            }

            ric.merge_with(ric1);
            ric_attr::set(m.get_match_value(), ric);
            return true;
        };
    }
};

class ShapeOf : public ngraph::pass::MatcherPass {
public:
    ShapeOf() {
        MATCHER_SCOPE(ShapeOf);
        pattern_root = pattern::wrap_type<opset1::ShapeOf, opset8::ShapeOf>();

        callback = [=](pattern::Matcher& m) {
            // Skip propagation for ShapeOf path
            return true;
        };
    }
};

class PassThrough : public ngraph::pass::MatcherPass {
public:
    PassThrough() {
        MATCHER_SCOPE(PassThrough);
        pattern_root = pattern::wrap_type<op::util::UnaryElementwiseArithmetic,
                                          opset8::Convert, opset8::Pad, opset8::PRelu>();

        callback = [=](pattern::Matcher& m) {
            auto root = m.get_match_root();
            if (!ric_attr::has(root->input_value(0))) return false;
            ric_attr::set(root->output(0), ric_attr::get(root->input_value(0)).propagate());
            return true;
        };
    }
};

class Transpose : public ngraph::pass::MatcherPass {
public:
    Transpose() {
        MATCHER_SCOPE(Transpose);
        auto input_p = pattern::any_input(ric_attr::has<Output<Node>>);
        auto order_p = pattern::wrap_type<opset8::Constant>();
        pattern_root = pattern::wrap_type<opset8::Transpose>({input_p, order_p});

        callback = [=](pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            auto input = pattern_map.at(input_p);
            auto ric = ric_attr::get(input).propagate();

            auto order_node = std::dynamic_pointer_cast<opset8::Constant>(pattern_map.at(order_p).get_node_shared_ptr());
            auto order = order_node->cast_vector<int64_t>();

            int64_t new_axis = std::find(order.begin(), order.end(), ric.get_axis()) - order.begin();
            ric.set_axis(new_axis);

            ric_attr::set(m.get_match_value(), ric);
            return true;
        };
    }
};

class Unsupported : public ngraph::pass::MatcherPass {
public:
    Unsupported() {
        MATCHER_SCOPE(Unsupported);
        pattern_root = pattern::any_input();
        callback = [=](pattern::Matcher& m) {
            for (const auto & input : m.get_match_root()->input_values()) {
                if (ric_attr::has(input)) {
                    auto ric = ric_attr::get(input);
                    ric.set_can_be_fused(false);
                    std::cout << "Node is unsupported: " << *m.get_match_root() << std::endl;
                }
            }
            return true;
        };
    }
};
}// namespace prop

namespace fuse {
namespace {
bool need_to_erase_ric(const Output<Node> & output) {
    if (!ric_attr::has(output)) return false;
    const auto & ric = ric_attr::get(output);
    return ric.can_be_fused() && ric.is_initial();
}
}// namespace

class InsertReverseInputChannel : public ngraph::pass::MatcherPass {
public:
    InsertReverseInputChannel() {
        MATCHER_SCOPE(InsertReverseInputChannel);
        pattern_root = pattern::any_input();
        callback = [](pattern::Matcher& m) {
            const auto & node = m.get_match_root();
            for (const auto & input : node->inputs()) {
                if (!ric_attr::has(input)) continue;
                const auto & ric = ric_attr::get(input);
                if (ric.can_be_fused() && ric.is_final()) {
                    ric(input);
                }
            }
            return false;
        };
    }
};

class EraseSplitConcat : public ngraph::pass::MatcherPass {
public:
    EraseSplitConcat() {
        MATCHER_SCOPE(EraseSplitConcat);
        auto input_p = pattern::any_input();
        auto split_p = pattern::wrap_type<opset8::Split>({input_p, pattern::any_input()});
        pattern_root = pattern::wrap_type<opset8::Concat>({split_p, split_p, split_p}, need_to_erase_ric);

        callback = [=](pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            auto output = pattern_map.at(pattern_root);
            auto input = pattern_map.at(input_p);
            output.replace(input);
            return true;
        };
    }
};

class EraseGather : public ngraph::pass::MatcherPass {
public:
    EraseGather() {
        MATCHER_SCOPE(EraseGather);
        auto input_p = pattern::any_input();
        pattern_root = pattern::wrap_type<opset8::Gather>({input_p, pattern::any_input(),
                                                                            pattern::any_input()},
                                                           need_to_erase_ric);
        callback = [=](pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            auto output = pattern_map.at(pattern_root);
            auto input = pattern_map.at(input_p);
            output.replace(input);
            return true;
        };
    }
};
}// namespace fuse

bool ngraph::pass::ReverseInputChannelsFusion::run_on_function(std::shared_ptr<Function> f) {
    Manager m;
    m.set_per_pass_validation(false);

    // First we need to initialize and propagate RIC attributes through entire graph
    auto ric_prop = m.register_pass<GraphRewrite>();
    ric_prop->add_matcher<init::SplitConcat>();
    ric_prop->add_matcher<init::Gather>();
    ric_prop->add_matcher<prop::Convolution>();
    ric_prop->add_matcher<prop::GroupConvolution>();
    ric_prop->add_matcher<prop::Binary>();
    ric_prop->add_matcher<prop::ShapeOf>();
    ric_prop->add_matcher<prop::Transpose>();
    ric_prop->add_matcher<prop::PassThrough>();
    ric_prop->add_matcher<prop::Unsupported>();

    // TODO: validate attributes by request

    // Second we fuse available RIC into nodes and remove original nodes related to fused RIC
    auto ric_fuse = m.register_pass<GraphRewrite>();
    ric_fuse->add_matcher<fuse::InsertReverseInputChannel>();
    ric_fuse->add_matcher<fuse::EraseSplitConcat>();
    ric_fuse->add_matcher<fuse::EraseGather>();

    m.run_passes(f);
    return false;
}
}// namespace pass
}// namespace ngraph