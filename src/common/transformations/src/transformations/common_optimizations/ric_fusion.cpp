// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/ric_fusion.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"


NGRAPH_RTTI_DEFINITION(ngraph::pass::ReverseInputChannelsFusion, "ReverseInputChannelsFusion", 0);

namespace ngraph {
namespace pass {
namespace ric_attr {

struct Attribute {
    std::shared_ptr<bool> can_be_fused;
    bool is_final{false};
    bool is_initial{false};

    std::function<void(Input<Node>)> callback = [](Input<Node>) {};

    Attribute() {
        can_be_fused = std::make_shared<bool>(true);
    }

    Attribute propagate() const {
        Attribute attr;
        attr.can_be_fused = can_be_fused;
        return attr;
    }

    bool operator==(const Attribute & other) const {
        return can_be_fused == other.can_be_fused &&
               is_final == other.is_final &&
               is_initial == other.is_initial;
    }
};

namespace {

template <typename T>
using is_port = typename std::enable_if<!std::is_convertible<T, std::shared_ptr<Node>>::value>::type;

template <typename T, typename = is_port<T>>
void set(T port, const std::vector<Attribute> & ric_attrs) {
    auto & attrs = port.get_rt_info();
    attrs["reverse_input_channel_index"] = ric_attrs;
}

template <typename T, typename = is_port<T>>
void set(T port, bool is_final = false, bool is_initial = false) {
    Attribute attr;
    attr.is_final = is_final;
    attr.is_initial = is_initial;
    set(port, {attr});
}

// Available only for output ports
void init(Output<Node> output) {
    set(output, false, true);
}

//bool has(const Output<Node> & output) {
//    const auto & attrs = output.get_rt_info();
//    auto res = attrs.find("reverse_input_channel_index");
//    if (res != attrs.end()) {
//        return !res->second.as<std::vector<Attribute>>().empty();
//    }
//    return false;
//}

template <typename T, typename = is_port<T>>
std::vector<Attribute> get(const T & port) {
    const auto & attrs = port.get_rt_info();
    auto res = attrs.find("reverse_input_channel_index");
    if (res != attrs.end()) {
        return res->second.template as<std::vector<Attribute>>();
    }
    return {};
}

template <typename T, typename = is_port<T>>
std::vector<Attribute> propagate(const T & port) {
    auto ric_attrs = get(port);
    std::vector<Attribute> new_attrs;
    std::for_each(ric_attrs.begin(), ric_attrs.end(), [&](Attribute & attr) {
        new_attrs.push_back(attr.propagate());
    });
    return new_attrs;
}

template <typename T, typename = is_port<T>>
void erase(T port) {
    auto & rt_info = port.get_rt_info();
    rt_info.erase("reverse_input_channel_index");
}
}// namespace
}// namespace ric_attr

namespace init {
class SplitConcat : public ngraph::pass::MatcherPass {
public:
    SplitConcat() {
        MATCHER_SCOPE(SplitConcat);
        auto split_p = pattern::wrap_type<opset8::Split>();
        auto concat_p = pattern::wrap_type<opset8::Concat>({split_p, split_p, split_p});

        auto callback = [=](pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_map();
            auto concat = ov::as_type_ptr<opset8::Concat>(pattern_map.at(concat_p));
            if (!concat || concat->get_axis() != 1) {
                return false;
            }

            auto split = ov::as_type_ptr<opset8::Split>(pattern_map.at(split_p));
            if (!split || split->get_num_splits() != 3 ||
                !op::util::has_constant_value(split->get_input_node_shared_ptr(1), 1)) {
                return false;
            }

            // Order of Split output indices
            const std::vector<size_t> & order = {2, 0, 1};

            for (auto input : concat->inputs()) {
                auto split_output = input.get_source_output();
                auto s = std::dynamic_pointer_cast<opset8::Split>(split_output.get_node_shared_ptr());
                if (!s || s != split) return false;

                // Check that Concat is the only Split consumer and order of Split outputs
                // satisfies expected order for reverse input channel case.
                if (split_output.get_target_inputs().size() != 1 ||
                    split_output.get_index() != order[input.get_index()]) {
                    return false;
                }
            }

            // Mark-up RIC output
            ric_attr::init(concat);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(concat_p, matcher_name);
        this->register_matcher(m, callback);
    }
};

class Gather : public ngraph::pass::MatcherPass {
public:
    Gather() {
        MATCHER_SCOPE(Gather);
        auto indices_p = pattern::any_input();
        auto axis_p = pattern::wrap_type<opset8::Constant>();
        auto gather_p = pattern::wrap_type<opset8::Gather>({pattern::any_input(), indices_p, axis_p});

        auto callback = [=](pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_map();
            auto value = ov::get_constant_from_source(pattern_map.at(indices_p));
            if (!op::util::has_constant_value(pattern_map.at(axis_p), 1) ||
                !op::util::has_constant_value(ov::get_constant_from_source(pattern_map.at(indices_p)),
                                             std::vector<int64_t>{2, 1, 0})) {
                return false;
            }

            ric_attr::init(m.get_match_value());
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(gather_p, matcher_name);
        this->register_matcher(m, callback);
    }
};
}// namespace init

namespace prop {
class Convolutions : public ngraph::pass::MatcherPass {
public:
    Convolutions() {
        MATCHER_SCOPE(Convolutions);
        auto input_p = pattern::any_input();
        auto conv_p = pattern::wrap_type<opset8::Convolution,
                                         opset8::GroupConvolution>({input_p, pattern::any_input()});

        auto callback = [=](pattern::Matcher& m) {
            auto conv = m.get_match_root();
            auto ric_attrs = ric_attr::propagate(conv->input_value(0));
            std::for_each(ric_attrs.begin(), ric_attrs.end(), [](ric_attr::Attribute & attr) {
                attr.is_final = true;
                attr.callback = [](Input<Node> input) {
                    auto weights = input.get_source_output();
                    auto gather = std::make_shared<opset8::Gather>(weights, opset8::Constant::create(element::i64, Shape{3}, {2, 1, 0}),
                                                                   opset8::Constant::create(element::i64, Shape{}, {1}));
                    input.replace_source_output(gather);
                    // TODO: copy runtime info from RIC sub-graph
                };
            });
            ric_attr::set(conv->input(1), ric_attrs);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(conv_p, matcher_name);
        this->register_matcher(m, callback);
    }
};

class Binary : public ngraph::pass::MatcherPass {
public:
    Binary() {
        MATCHER_SCOPE(Binary);
        auto binary_p = pattern::wrap_type<opset8::Multiply, opset8::Add, opset8::Subtract, opset8::Divide>();

        auto callback = [=](pattern::Matcher& m) {
            auto root = m.get_match_root();
            auto lhs_ric = ric_attr::propagate(root->input_value(0));
            auto rhs_ric = ric_attr::propagate(root->input_value(1));

            std::vector<ric_attr::Attribute> new_attrs;
            new_attrs.insert(new_attrs.end(), lhs_ric.begin(), lhs_ric.end());
            new_attrs.insert(new_attrs.end(), rhs_ric.begin(), rhs_ric.end());

            // For cases when RIC came only from one branch we have to put annotation to insert RIC on another branch
            if (lhs_ric.empty() || rhs_ric.empty()) {
                auto insert_ric_attrs = new_attrs;
                std::for_each(insert_ric_attrs.begin(), insert_ric_attrs.end(), [](ric_attr::Attribute & attr) {
                    attr.is_final = true;
                    attr.callback = [](Input<Node> input) {
                        // TODO: check eltwise before insertion
                        auto output = input.get_source_output();
                        auto gather = std::make_shared<opset8::Gather>(output, opset8::Constant::create(element::i64, Shape{3}, {2, 1, 0}),
                                                                       opset8::Constant::create(element::i64, Shape{}, {1}));
                        input.replace_source_output(gather);
                        // TODO: copy runtime info from RIC sub-graph
                    };
                });

                auto input = lhs_ric.empty() ? root->input(0) : root->input(1);
                ric_attr::set(input, insert_ric_attrs);
            }

            ric_attr::set(root->output(0), new_attrs);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(binary_p, matcher_name);
        this->register_matcher(m, callback);
    }
};

class Unary : public ngraph::pass::MatcherPass {
public:
    Unary() {
        MATCHER_SCOPE(Unary);
        auto unary_p = pattern::wrap_type<opset8::Relu>();

        auto callback = [=](pattern::Matcher& m) {
            auto root = m.get_match_root();
            ric_attr::set(root->output(0), ric_attr::propagate(root->input_value(0)));
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(unary_p, matcher_name);
        this->register_matcher(m, callback);
    }
};

class Unsupported : public ngraph::pass::MatcherPass {
public:
    Unsupported() {
        MATCHER_SCOPE(Unsupported);
        auto input = pattern::any_input();

        auto callback = [=](pattern::Matcher& m) {
            auto ric_attrs = ric_attr::get(m.get_match_value());
            // In case if operation is not supported we reset can_be_fused for all related attributes
            std::for_each(ric_attrs.begin(), ric_attrs.end(), [](ric_attr::Attribute & attr) {
                *attr.can_be_fused = false;
            });
            ric_attr::set(m.get_match_value(), ric_attrs);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(input, matcher_name);
        this->register_matcher(m, callback);
    }
};
}// namespace prop

namespace fuse {
namespace {
bool need_to_insert_ric(Input<Node> output) {
    const auto & ric_attrs = ric_attr::get(output);
    return !ric_attrs.empty() && std::all_of(ric_attrs.cbegin(), ric_attrs.cend(), [](const ric_attr::Attribute & attr) {
        return attr.is_final && *attr.can_be_fused;
    });
}

bool need_to_erase_ric(Output<Node> output) {
    const auto & ric_attrs = ric_attr::get(output);
    return !ric_attrs.empty() && std::all_of(ric_attrs.cbegin(), ric_attrs.cend(), [](const ric_attr::Attribute & attr) {
        return attr.is_initial && *attr.can_be_fused;
    });
}
}// namespace

class InsertReverseInputChannel : public ngraph::pass::MatcherPass {
public:
    InsertReverseInputChannel() {
        MATCHER_SCOPE(InsertReverseInputChannel);
        auto output = pattern::any_input();

        auto callback = [](pattern::Matcher& m) {
            const auto & node = m.get_match_root();
            for (auto && input : node->inputs()) {
                if (need_to_insert_ric(input)) {
                    auto attrs = ric_attr::get(input);
                    attrs[0].callback(input);
                }
            }
            return false;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(output, matcher_name);
        this->register_matcher(m, callback);
    }
};

class EraseSplitConcat : public ngraph::pass::MatcherPass {
public:
    EraseSplitConcat() {
        MATCHER_SCOPE(EraseSplitConcat);
        auto input_p = pattern::any_input();
        auto split_p = pattern::wrap_type<opset8::Split>({input_p, pattern::any_input()});
        auto concat_p = pattern::wrap_type<opset8::Concat>({split_p, split_p, split_p}, need_to_erase_ric);

        auto callback = [=](pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            auto output = pattern_map.at(concat_p);
            auto input = pattern_map.at(input_p);
            output.replace(input);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(concat_p, matcher_name);
        this->register_matcher(m, callback);
    }
};

class EraseGather : public ngraph::pass::MatcherPass {
public:
    EraseGather() {
        MATCHER_SCOPE(EraseGather);
        auto input_p = pattern::any_input();
        auto gather_p = pattern::wrap_type<opset8::Gather>({input_p, pattern::any_input(),
                                                                            pattern::any_input()},
                                                           need_to_erase_ric);
        auto callback = [=](pattern::Matcher& m) {
            const auto & pattern_map = m.get_pattern_value_map();
            auto output = pattern_map.at(gather_p);
            auto input = pattern_map.at(input_p);
            output.replace(input);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(gather_p, matcher_name);
        this->register_matcher(m, callback);
    }
};
}// namespace fuse

bool ngraph::pass::ReverseInputChannelsFusion::run_on_function(std::shared_ptr<Function> f) {
    Manager m;
    // TODO: enable
    // m.set_per_pass_validation(false);

    // First we need to initialize and propagate RIC attributes through entire graph
    auto ric_prop = m.register_pass<GraphRewrite>();
    ric_prop->add_matcher<init::SplitConcat>();
    ric_prop->add_matcher<init::Gather>();
    ric_prop->add_matcher<prop::Convolutions>();
    ric_prop->add_matcher<prop::Binary>();
    ric_prop->add_matcher<prop::Unary>();
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