// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/node.hpp"

#pragma once

#ifdef ENABLE_OPENVINO_DEBUG

namespace ov {
namespace util {

class OPENVINO_API LevelString {
private:
    LevelString(const std::string& level_identifier_)
        : level_identifier(level_identifier_),
          level_str(level_identifier_) {
        level_str.reserve(level_identifier_.size() * 10);
    }

public:
    static ov::util::LevelString& get() {
        static ov::util::LevelString instance("│  ");
        return instance;
    }

    LevelString& operator++() {
        level_str += level_identifier;
        return *this;
    }

    LevelString operator++(int) {
        LevelString res = *this;
        level_str += level_identifier;
        return res;
    }

    LevelString& operator--() {
        if (level_str.length() > level_identifier.size()) {
            level_str.erase(level_str.size() - level_identifier.size(), level_identifier.size());
        }
        return *this;
    }

    LevelString operator--(int) {
        LevelString res = *this;
        if (level_str.length() > level_identifier.size()) {
            level_str.erase(level_str.size() - level_identifier.size(), level_identifier.size());
        }
        return res;
    }

    friend std::ostream& operator<<(std::ostream& stream, const LevelString& level_string) {
        return stream << level_string.level_str;
    }

private:
    const std::string level_identifier;
    std::string level_str;
};

OPENVINO_API std::string node_version_type_str(const std::shared_ptr<ov::Node>& node);
OPENVINO_API std::string node_version_type_name_str(const std::shared_ptr<ov::Node>& node);
OPENVINO_API std::string node_with_arguments(const std::shared_ptr<ov::Node>& node);

// For each file, that has the matcher logging functionality present,
// there's a set of macro for avoiding the additional clutter
// of the matching code.

// transformations/utils/gen_pattern.hpp
#    define OPENVINO_LOG_GENPATTERN1(matcher, pattern_value, graph_value)  \
        OPENVINO_LOG_MATCHING(matcher,                                     \
                              ov::util::LevelString::get(),                \
                              OPENVINO_BLOCK_END,                          \
                              OPENVINO_RED,                                \
                              "  OUTPUT INDICES DIDN'T MATCH. EXPECTED: ", \
                              pattern_value.get_index(),                   \
                              ". OBSERVED: ",                              \
                              graph_value.get_index());

#    define OPENVINO_LOG_GENPATTERN2(matcher, pattern_value, graph_value)                           \
        OPENVINO_LOG_MATCHING(matcher,                                                              \
                              ov::util::LevelString::get(),                                         \
                              OPENVINO_BLOCK_END,                                                   \
                              OPENVINO_RED,                                                         \
                              "  NODES' TYPE DIDN'T MATCH. EXPECTED: ",                             \
                              ov::util::node_version_type_str(pattern_value.get_node_shared_ptr()), \
                              ". OBSERVED: ",                                                       \
                              ov::util::node_version_type_str(graph_value.get_node_shared_ptr()));

#    define OPENVINO_LOG_GENPATTERN3(matcher)               \
        OPENVINO_LOG_MATCHING(matcher,                      \
                              ov::util::LevelString::get(), \
                              OPENVINO_BLOCK_END,           \
                              OPENVINO_RED,                 \
                              "  PREDICATE DIDN'T MATCH.");

#    define OPENVINO_LOG_GENPATTERN4(matcher)               \
        OPENVINO_LOG_MATCHING(matcher,                      \
                              ov::util::LevelString::get(), \
                              OPENVINO_BLOCK_END,           \
                              OPENVINO_RED,                 \
                              "  ATTRIBUTES DIDN'T MATCH.");

#    define OPENVINO_LOG_GENPATTERN5(matcher)               \
        OPENVINO_LOG_MATCHING(matcher,                      \
                              ov::util::LevelString::get(), \
                              OPENVINO_BLOCK_BODY_RIGHT,    \
                              " TYPE MATCHED. CHECKING PATTERN ARGUMENTS");

#    define OPENVINO_LOG_GENPATTERN6(matcher, status)                   \
        OPENVINO_LOG_MATCHING(matcher,                                  \
                              ov::util::LevelString::get(),             \
                              OPENVINO_BLOCK_BODY,                      \
                              '\n',                                     \
                              ov::util::LevelString::get(),             \
                              OPENVINO_BLOCK_END,                       \
                              (status ? OPENVINO_GREEN : OPENVINO_RED), \
                              (status ? "  ALL ARGUMENTS MATCHED" : "  ARGUMENTS DIDN'T MATCH"));

// core/src/node.cpp
#    define OPENVINO_LOG_NODE1(matcher, pattern_value, graph_value)                                     \
        OPENVINO_LOG_MATCHING(matcher,                                                                  \
                              ov::util::LevelString::get(),                                             \
                              OPENVINO_BLOCK_END,                                                       \
                              "  INDEX, ELEMENT TYPE or PARTIAL SHAPE MISMATCH. EXPECTED IN PATTERN: ", \
                              pattern_value.get_index(),                                                \
                              ", ",                                                                     \
                              pattern_value.get_element_type(),                                         \
                              ", ",                                                                     \
                              pattern_value.get_partial_shape(),                                        \
                              ". OBSERVED IN GRAPH: ",                                                  \
                              graph_value.get_index(),                                                  \
                              ", ",                                                                     \
                              graph_value.get_element_type(),                                           \
                              ", ",                                                                     \
                              graph_value.get_partial_shape());

#    define OPENVINO_LOG_NODE2(matcher)                     \
        OPENVINO_LOG_MATCHING(matcher,                      \
                              ov::util::LevelString::get(), \
                              OPENVINO_BLOCK_BODY_RIGHT,    \
                              " TYPE MATCHED. CHECKING PATTERN ARGUMENTS");

#    define OPENVINO_LOG_NODE3(matcher)                     \
        OPENVINO_LOG_MATCHING(matcher,                      \
                              ov::util::LevelString::get(), \
                              OPENVINO_BLOCK_BODY,          \
                              '\n',                         \
                              ov::util::LevelString::get(), \
                              OPENVINO_BLOCK_END,           \
                              OPENVINO_GREEN,               \
                              "  ALL ARGUMENTS MATCHED");

#    define OPENVINO_LOG_NODE4(matcher)                     \
        OPENVINO_LOG_MATCHING(matcher,                      \
                              ov::util::LevelString::get(), \
                              OPENVINO_BLOCK_BODY,          \
                              '\n',                         \
                              ov::util::LevelString::get(), \
                              OPENVINO_BLOCK_END,           \
                              OPENVINO_RED,                 \
                              "  ARGUMENTS DIDN'T MATCH");

#    define OPENVINO_LOG_NODE5(matcher, pattern_value, graph_value)           \
        OPENVINO_LOG_MATCHING(matcher,                                        \
                              ov::util::LevelString::get(),                   \
                              OPENVINO_BLOCK_END,                             \
                              OPENVINO_RED,                                   \
                              "  NODES' TYPE DIDN'T MATCH. EXPECTED: ",       \
                              ov::util::node_version_type_str(pattern_value), \
                              ". OBSERVED: ",                                 \
                              ov::util::node_version_type_str(graph_value.get_node_shared_ptr()));

// src/pass/graph_rewrite.cpp
#    define OPENVINO_LOG_GRAPH_REWRITE1(matcher, node_ptr)                       \
        OPENVINO_LOG_MATCHING(matcher,                                           \
                              OPENVINO_BLOCK_BEG,                                \
                              OPENVINO_YELLOW,                                   \
                              "  [",                                             \
                              matcher->get_name(),                               \
                              "] START: trying to start pattern matching with ", \
                              ov::util::node_version_type_name_str(node_ptr));

#    define OPENVINO_LOG_GRAPH_REWRITE2(matcher, status)                \
        OPENVINO_LOG_MATCHING(matcher,                                  \
                              OPENVINO_BLOCK_BODY,                      \
                              '\n',                                     \
                              OPENVINO_BLOCK_END,                       \
                              (status ? OPENVINO_GREEN : OPENVINO_RED), \
                              "  [",                                    \
                              matcher->get_name(),                      \
                              "] END: PATTERN MATCHED, CALLBACK ",      \
                              (status ? "SUCCEDED" : "FAILED"),         \
                              "\n");

#    define OPENVINO_LOG_GRAPH_REWRITE3(matcher, exception)                    \
        OPENVINO_LOG_MATCHING(matcher,                                         \
                              OPENVINO_BLOCK_BODY,                             \
                              '\n',                                            \
                              OPENVINO_BLOCK_END,                              \
                              OPENVINO_RED,                                    \
                              "  [",                                           \
                              matcher->get_name(),                             \
                              "] END: PATTERN MATCHED, CALLBACK HAS THROWN: ", \
                              exception.what());

#    define OPENVINO_LOG_GRAPH_REWRITE4(matcher)   \
        OPENVINO_LOG_MATCHING(matcher,             \
                              OPENVINO_BLOCK_BODY, \
                              '\n',                \
                              OPENVINO_BLOCK_END,  \
                              OPENVINO_RED,        \
                              "  [",               \
                              matcher->get_name(), \
                              "] END: PATTERN DIDN'T MATCH\n");

// src/pattern/matcher.cpp
#    define OPENVINO_LOG_MATCHER1(matcher, pattern_value, graph_value)                            \
        OPENVINO_LOG_MATCHING(matcher,                                                            \
                              ov::util::LevelString::get(),                                       \
                              '\n',                                                               \
                              ov::util::LevelString::get(),                                       \
                              OPENVINO_BLOCK_BEG,                                                 \
                              "  MATCHING PATTERN NODE: ",                                        \
                              ov::util::node_with_arguments(pattern_value.get_node_shared_ptr()), \
                              '\n',                                                               \
                              ov::util::LevelString::get(),                                       \
                              OPENVINO_BLOCK_BODY_RIGHT,                                          \
                              " AGAINST  GRAPH   NODE: ",                                         \
                              ov::util::node_with_arguments(graph_value.get_node_shared_ptr()));

#    define OPENVINO_LOG_MATCHER2(matcher, idx)                         \
        OPENVINO_LOG_MATCHING(matcher, ++ov::util::LevelString::get()); \
        OPENVINO_LOG_MATCHING(matcher, ov::util::LevelString::get()++, OPENVINO_BLOCK_BEG, "  ARGUMENT ", idx);

#    define OPENVINO_LOG_MATCHER3(matcher, idx)                                              \
        OPENVINO_LOG_MATCHING(matcher, --ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
        OPENVINO_LOG_MATCHING(matcher,                                                       \
                              ov::util::LevelString::get()--,                                \
                              OPENVINO_BLOCK_END,                                            \
                              OPENVINO_RED,                                                  \
                              "  ARGUMENT ",                                                 \
                              idx,                                                           \
                              " DIDN'T MATCH ");

#    define OPENVINO_LOG_MATCHER4(matcher, idx)                                              \
        OPENVINO_LOG_MATCHING(matcher, --ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
        OPENVINO_LOG_MATCHING(matcher,                                                       \
                              ov::util::LevelString::get()--,                                \
                              OPENVINO_BLOCK_END,                                            \
                              OPENVINO_GREEN,                                                \
                              "  ARGUMENT ",                                                 \
                              idx,                                                           \
                              " MATCHED");

#    define OPENVINO_LOG_MATCHER5(matcher, pattern_arg_size, graph_arg_size)                    \
        OPENVINO_LOG_MATCHING(matcher,                                                          \
                              ov::util::LevelString::get(),                                     \
                              OPENVINO_BLOCK_BODY_RIGHT,                                        \
                              OPENVINO_RED,                                                     \
                              " NUMBER OF ARGUMENTS DOESN'T MATCH. EXPECTED IN PATTERN NODE: ", \
                              pattern_arg_size,                                                 \
                              ". OBSERVED IN GRAPH NODE: ",                                     \
                              graph_arg_size);

#    define OPENVINO_LOG_MATCHER6(matcher)                              \
        OPENVINO_LOG_MATCHING(matcher, ++ov::util::LevelString::get()); \
        OPENVINO_LOG_MATCHING(matcher, ov::util::LevelString::get(), OPENVINO_BLOCK_BEG, "  NEW PERMUTATION");

#    define OPENVINO_LOG_MATCHER7(matcher)                                                 \
        OPENVINO_LOG_MATCHING(matcher, ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
        OPENVINO_LOG_MATCHING(matcher,                                                     \
                              ov::util::LevelString::get()--,                              \
                              OPENVINO_BLOCK_END,                                          \
                              OPENVINO_GREEN,                                              \
                              "  PERMUTATION MATCHED");

#    define OPENVINO_LOG_MATCHER8(matcher)                                                 \
        OPENVINO_LOG_MATCHING(matcher, ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
        OPENVINO_LOG_MATCHING(matcher,                                                     \
                              ov::util::LevelString::get()--,                              \
                              OPENVINO_BLOCK_END,                                          \
                              OPENVINO_RED,                                                \
                              "  PERMUTATION DIDN'T MATCH");

#    define OPENVINO_LOG_MATCHER9(matcher)                              \
        OPENVINO_LOG_MATCHING(matcher, ++ov::util::LevelString::get()); \
        OPENVINO_LOG_MATCHING(matcher,                                  \
                              ov::util::LevelString::get(),             \
                              OPENVINO_BLOCK_BEG,                       \
                              "  GRAPH NODE IS NOT COMMUTATIVE, A SINGLE PERMUTATION IS PRESENT ONLY");

#    define OPENVINO_LOG_MATCHER10(matcher, status)                                        \
        OPENVINO_LOG_MATCHING(matcher, ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
        OPENVINO_LOG_MATCHING(matcher,                                                     \
                              ov::util::LevelString::get()--,                              \
                              OPENVINO_BLOCK_END,                                          \
                              (status ? OPENVINO_GREEN : OPENVINO_RED),                    \
                              "  PERMUTATION ",                                            \
                              (status ? "MATCHED" : "DIDN'T MATCH"));

#    define OPENVINO_LOG_MATCHER11(matcher)                                                \
        OPENVINO_LOG_MATCHING(matcher, ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
        OPENVINO_LOG_MATCHING(matcher,                                                     \
                              ov::util::LevelString::get(),                                \
                              OPENVINO_BLOCK_BODY_RIGHT,                                   \
                              OPENVINO_RED,                                                \
                              " NONE OF PERMUTATIONS MATCHED");

// pattern/op/label.cpp
#    define OPENVINO_LOG_LABEL1(matcher, label_name)        \
        OPENVINO_LOG_MATCHING(matcher,                      \
                              ov::util::LevelString::get(), \
                              OPENVINO_BLOCK_END,           \
                              OPENVINO_GREEN,               \
                              "  LABEL MATCHED: ",          \
                              label_name);

#    define OPENVINO_LOG_LABEL2(matcher, label_name)          \
        OPENVINO_LOG_MATCHING(matcher,                        \
                              ov::util::LevelString::get()++, \
                              OPENVINO_BLOCK_BODY_RIGHT,      \
                              " CHECKING INSIDE LABEL: ",     \
                              label_name);

#    define OPENVINO_LOG_LABEL3(matcher)                                                     \
        OPENVINO_LOG_MATCHING(matcher, --ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
        OPENVINO_LOG_MATCHING(matcher,                                                       \
                              ov::util::LevelString::get(),                                  \
                              OPENVINO_BLOCK_END,                                            \
                              OPENVINO_GREEN,                                                \
                              "  LABEL MATCHED");

#    define OPENVINO_LOG_LABEL4(matcher)                                                   \
        OPENVINO_LOG_MATCHING(matcher, ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
        OPENVINO_LOG_MATCHING(matcher,                                                     \
                              ov::util::LevelString::get(),                                \
                              OPENVINO_BLOCK_END,                                          \
                              OPENVINO_RED,                                                \
                              "  LABEL DIDN'T MATCH");

// pattern/op/optional.cpp
#    define OPENVINO_LOG_OPTIONAL1(matcher, or_node, wrap_node, opt_name)                                    \
        OPENVINO_LOG_MATCHING(matcher,                                                                       \
                              ov::util::LevelString::get()++,                                                \
                              OPENVINO_BLOCK_BODY_RIGHT,                                                     \
                              (or_node == wrap_node ? " LEAVING OPTIONAL AS WRAP TYPE AND TRYING TO MATCH: " \
                                                    : " UNFOLDING OPTIONAL INTO OR AND TRYING TO MATCH: "),  \
                              opt_name);

#    define OPENVINO_LOG_OPTIONAL2(matcher)                                                  \
        OPENVINO_LOG_MATCHING(matcher, --ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
        OPENVINO_LOG_MATCHING(matcher,                                                       \
                              ov::util::LevelString::get(),                                  \
                              OPENVINO_BLOCK_END,                                            \
                              OPENVINO_GREEN,                                                \
                              "  OPTIONAL MATCHED");

#    define OPENVINO_LOG_OPTIONAL3(matcher)                                                  \
        OPENVINO_LOG_MATCHING(matcher, --ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
        OPENVINO_LOG_MATCHING(matcher,                                                       \
                              ov::util::LevelString::get(),                                  \
                              OPENVINO_BLOCK_END,                                            \
                              OPENVINO_RED,                                                  \
                              "  OPTIONAL DIDN'T MATCH");

// pattern/op/or.cpp
#    define OPENVINO_LOG_OR1(matcher, input_size, or_name)  \
        OPENVINO_LOG_MATCHING(matcher,                      \
                              ov::util::LevelString::get(), \
                              OPENVINO_BLOCK_BODY_RIGHT,    \
                              " CHECKING ",                 \
                              input_size,                   \
                              " OR BRANCHES: ",             \
                              or_name);

#    define OPENVINO_LOG_OR2(matcher, idx, input_value)                 \
        OPENVINO_LOG_MATCHING(matcher, ++ov::util::LevelString::get()); \
        OPENVINO_LOG_MATCHING(matcher,                                  \
                              ov::util::LevelString::get()++,           \
                              OPENVINO_BLOCK_BEG,                       \
                              "  BRANCH ",                              \
                              idx,                                      \
                              ": ",                                     \
                              ov::util::node_version_type_str(input_value.get_node_shared_ptr()));

#    define OPENVINO_LOG_OR3(matcher, idx)                                                   \
        OPENVINO_LOG_MATCHING(matcher, --ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
        OPENVINO_LOG_MATCHING(matcher,                                                       \
                              ov::util::LevelString::get()--,                                \
                              OPENVINO_BLOCK_END,                                            \
                              OPENVINO_GREEN,                                                \
                              "  BRANCH ",                                                   \
                              idx,                                                           \
                              " MATCHED");                                                   \
        OPENVINO_LOG_MATCHING(matcher, ov::util::LevelString::get(), OPENVINO_BLOCK_BODY);   \
        OPENVINO_LOG_MATCHING(matcher,                                                       \
                              ov::util::LevelString::get(),                                  \
                              OPENVINO_BLOCK_END,                                            \
                              OPENVINO_GREEN,                                                \
                              "  BRANCH ",                                                   \
                              idx,                                                           \
                              " HAS MATCHED");

#    define OPENVINO_LOG_OR4(matcher, idx)                                                   \
        OPENVINO_LOG_MATCHING(matcher, --ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
        OPENVINO_LOG_MATCHING(matcher,                                                       \
                              ov::util::LevelString::get()--,                                \
                              OPENVINO_BLOCK_END,                                            \
                              OPENVINO_RED,                                                  \
                              "  BRANCH ",                                                   \
                              idx,                                                           \
                              " DIDN'T MATCH");

#    define OPENVINO_LOG_OR5(matcher)                                                      \
        OPENVINO_LOG_MATCHING(matcher, ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
        OPENVINO_LOG_MATCHING(matcher,                                                     \
                              ov::util::LevelString::get(),                                \
                              OPENVINO_BLOCK_END,                                          \
                              OPENVINO_RED,                                                \
                              "  NONE OF OR BRANCHES MATCHED");

// pattern/op/true.cpp
#    define OPENVINO_LOG_TRUE1(matcher)                     \
        OPENVINO_LOG_MATCHING(matcher,                      \
                              ov::util::LevelString::get(), \
                              OPENVINO_BLOCK_END,           \
                              OPENVINO_GREEN,               \
                              "  TRUE ALWAYS MATCHES");

// pattern/op/wrap_type.cpp
#    define OPENVINO_LOG_WRAPTYPE1(matcher, pattern_value, graph_value)                             \
        OPENVINO_LOG_MATCHING(matcher,                                                              \
                              ov::util::LevelString::get(),                                         \
                              OPENVINO_BLOCK_END,                                                   \
                              OPENVINO_RED,                                                         \
                              "  NODES' TYPE DIDN'T MATCH. EXPECTED: ",                             \
                              ov::util::node_version_type_str(pattern_value.get_node_shared_ptr()), \
                              ". OBSERVED: ",                                                       \
                              ov::util::node_version_type_str(graph_value.get_node_shared_ptr()));

#    define OPENVINO_LOG_WRAPTYPE2(matcher)                 \
        OPENVINO_LOG_MATCHING(matcher,                      \
                              ov::util::LevelString::get(), \
                              OPENVINO_BLOCK_END,           \
                              OPENVINO_RED,                 \
                              "  NODES' TYPE MATCHED, but PREDICATE FAILED");

#    define OPENVINO_LOG_WRAPTYPE3(matcher, num_arguments)                     \
        OPENVINO_LOG_MATCHING(matcher,                                         \
                              ov::util::LevelString::get(),                    \
                              OPENVINO_BLOCK_BODY_RIGHT,                       \
                              OPENVINO_GREEN,                                  \
                              " NODES' TYPE and PREDICATE MATCHED. CHECKING ", \
                              num_arguments,                                   \
                              " PATTERN ARGUMENTS: ");

#    define OPENVINO_LOG_WRAPTYPE4(matcher, status)                     \
        OPENVINO_LOG_MATCHING(matcher,                                  \
                              ov::util::LevelString::get(),             \
                              OPENVINO_BLOCK_BODY,                      \
                              '\n',                                     \
                              ov::util::LevelString::get(),             \
                              OPENVINO_BLOCK_END,                       \
                              (status ? OPENVINO_GREEN : OPENVINO_RED), \
                              (status ? "  ALL ARGUMENTS MATCHED" : "  ARGUMENTS DIDN'T MATCH"));

}  // namespace util
}  // namespace ov

#else

#    define OPENVINO_LOG_GENPATTERN1(...)
#    define OPENVINO_LOG_GENPATTERN2(...)
#    define OPENVINO_LOG_GENPATTERN3(...)
#    define OPENVINO_LOG_GENPATTERN4(...)
#    define OPENVINO_LOG_GENPATTERN5(...)
#    define OPENVINO_LOG_GENPATTERN6(...)

#    define OPENVINO_LOG_NODE1(...)
#    define OPENVINO_LOG_NODE2(...)
#    define OPENVINO_LOG_NODE3(...)
#    define OPENVINO_LOG_NODE4(...)
#    define OPENVINO_LOG_NODE5(...)

#    define OPENVINO_LOG_MATCHER1(...)
#    define OPENVINO_LOG_MATCHER2(...)
#    define OPENVINO_LOG_MATCHER3(...)
#    define OPENVINO_LOG_MATCHER4(...)
#    define OPENVINO_LOG_MATCHER5(...)
#    define OPENVINO_LOG_MATCHER6(...)
#    define OPENVINO_LOG_MATCHER7(...)
#    define OPENVINO_LOG_MATCHER8(...)
#    define OPENVINO_LOG_MATCHER9(...)
#    define OPENVINO_LOG_MATCHER10(...)
#    define OPENVINO_LOG_MATCHER11(...)

#    define OPENVINO_LOG_GRAPH_REWRITE1(...)
#    define OPENVINO_LOG_GRAPH_REWRITE2(...)
#    define OPENVINO_LOG_GRAPH_REWRITE3(...)
#    define OPENVINO_LOG_GRAPH_REWRITE4(...)

#    define OPENVINO_LOG_LABEL1(...)
#    define OPENVINO_LOG_LABEL2(...)
#    define OPENVINO_LOG_LABEL3(...)
#    define OPENVINO_LOG_LABEL4(...)

#    define OPENVINO_LOG_OR1(...)
#    define OPENVINO_LOG_OR2(...)
#    define OPENVINO_LOG_OR3(...)

#    define OPENVINO_LOG_TRUE1(...)

#    define OPENVINO_LOG_OPTIONAL1(...)
#    define OPENVINO_LOG_OPTIONAL2(...)
#    define OPENVINO_LOG_OPTIONAL3(...)

#    define OPENVINO_LOG_WRAPTYPE1(...)
#    define OPENVINO_LOG_WRAPTYPE2(...)
#    define OPENVINO_LOG_WRAPTYPE3(...)
#    define OPENVINO_LOG_WRAPTYPE4(...)

#endif