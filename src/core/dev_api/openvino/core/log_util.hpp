// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/node.hpp"

#ifdef ENABLE_OPENVINO_DEBUG

namespace ov {
namespace util {

class OPENVINO_API LevelString {
private:
    LevelString(const std::string& level_identifier_) : level_identifier(level_identifier_) {}

public:
    static LevelString& get() {
        static LevelString instance("│  ");
        return instance;
    }

    LevelString& operator++() {
        level += 1;
        return *this;
    }

    LevelString operator++(int) {
        LevelString res = *this;
        level += 1;
        return res;
    }

    LevelString& operator--() {
        if (level > 1) {
            level -= 1;
        }
        return *this;
    }

    LevelString operator--(int) {
        LevelString res = *this;
        if (level > 1) {
            level -= 1;
        }
        return res;
    }

    friend std::ostream& operator<<(std::ostream& stream, const LevelString& level_string) {
        for (int i = 0; i < level_string.level; ++i) {
            stream << level_string.level_identifier;
        }
        return stream;
    }

private:
    const std::string level_identifier;
    int level = 1;
};

OPENVINO_API std::string node_version_type_str(const ov::Node& node);
OPENVINO_API std::string node_version_type_name_str(const ov::Node& node);
OPENVINO_API std::string node_with_arguments(const ov::Node& node);

// For each file, that has the matcher logging functionality present,
// there's a set of macro for avoiding the additional clutter
// of the matching code.

// transformations/utils/gen_pattern.hpp
#    define OPENVINO_LOG_GENPATTERN1(matcher, pattern_value, graph_value)      \
        do {                                                                   \
            OPENVINO_LOG_MATCHING(matcher,                                     \
                                  ov::util::LevelString::get(),                \
                                  OPENVINO_BLOCK_END,                          \
                                  OPENVINO_RED,                                \
                                  "  OUTPUT INDICES DIDN'T MATCH. EXPECTED: ", \
                                  pattern_value.get_index(),                   \
                                  ". OBSERVED: ",                              \
                                  graph_value.get_index());                    \
        } while (0)

#    define OPENVINO_LOG_GENPATTERN2(matcher, pattern_value, graph_value)                                \
        do {                                                                                             \
            OPENVINO_LOG_MATCHING(matcher,                                                               \
                                  ov::util::LevelString::get(),                                          \
                                  OPENVINO_BLOCK_END,                                                    \
                                  OPENVINO_RED,                                                          \
                                  "  NODES' TYPE DIDN'T MATCH. EXPECTED: ",                              \
                                  ov::util::node_version_type_str(*pattern_value.get_node_shared_ptr()), \
                                  ". OBSERVED: ",                                                        \
                                  ov::util::node_version_type_str(*graph_value.get_node_shared_ptr()));  \
        } while (0)

#    define OPENVINO_LOG_GENPATTERN3(matcher)                   \
        do {                                                    \
            OPENVINO_LOG_MATCHING(matcher,                      \
                                  ov::util::LevelString::get(), \
                                  OPENVINO_BLOCK_END,           \
                                  OPENVINO_RED,                 \
                                  "  PREDICATE DIDN'T MATCH."); \
        } while (0)

#    define OPENVINO_LOG_GENPATTERN4(matcher)                    \
        do {                                                     \
            OPENVINO_LOG_MATCHING(matcher,                       \
                                  ov::util::LevelString::get(),  \
                                  OPENVINO_BLOCK_END,            \
                                  OPENVINO_RED,                  \
                                  "  ATTRIBUTES DIDN'T MATCH."); \
        } while (0)

#    define OPENVINO_LOG_GENPATTERN5(matcher)                                   \
        do {                                                                    \
            OPENVINO_LOG_MATCHING(matcher,                                      \
                                  ov::util::LevelString::get(),                 \
                                  OPENVINO_BLOCK_BODY_RIGHT,                    \
                                  " TYPE MATCHED. CHECKING PATTERN ARGUMENTS"); \
        } while (0)

#    define OPENVINO_LOG_GENPATTERN6(matcher, status)                                                 \
        do {                                                                                          \
            OPENVINO_LOG_MATCHING(matcher,                                                            \
                                  ov::util::LevelString::get(),                                       \
                                  OPENVINO_BLOCK_BODY,                                                \
                                  '\n',                                                               \
                                  ov::util::LevelString::get(),                                       \
                                  OPENVINO_BLOCK_END,                                                 \
                                  (status ? OPENVINO_GREEN : OPENVINO_RED),                           \
                                  (status ? "  ALL ARGUMENTS MATCHED" : "  ARGUMENTS DIDN'T MATCH")); \
        } while (0)

// core/src/node.cpp
#    define OPENVINO_LOG_NODE1(matcher, pattern_value, graph_value)                                         \
        do {                                                                                                \
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
                                  graph_value.get_partial_shape());                                         \
        } while (0)

#    define OPENVINO_LOG_NODE2(matcher)                                         \
        do {                                                                    \
            OPENVINO_LOG_MATCHING(matcher,                                      \
                                  ov::util::LevelString::get(),                 \
                                  OPENVINO_BLOCK_BODY_RIGHT,                    \
                                  " TYPE MATCHED. CHECKING PATTERN ARGUMENTS"); \
        } while (0)

#    define OPENVINO_LOG_NODE3(matcher)                         \
        do {                                                    \
            OPENVINO_LOG_MATCHING(matcher,                      \
                                  ov::util::LevelString::get(), \
                                  OPENVINO_BLOCK_BODY,          \
                                  '\n',                         \
                                  ov::util::LevelString::get(), \
                                  OPENVINO_BLOCK_END,           \
                                  OPENVINO_GREEN,               \
                                  "  ALL ARGUMENTS MATCHED");   \
        } while (0)

#    define OPENVINO_LOG_NODE4(matcher)                         \
        do {                                                    \
            OPENVINO_LOG_MATCHING(matcher,                      \
                                  ov::util::LevelString::get(), \
                                  OPENVINO_BLOCK_BODY,          \
                                  '\n',                         \
                                  ov::util::LevelString::get(), \
                                  OPENVINO_BLOCK_END,           \
                                  OPENVINO_RED,                 \
                                  "  ARGUMENTS DIDN'T MATCH");  \
        } while (0)

#    define OPENVINO_LOG_NODE5(matcher, pattern_value, graph_value)                                     \
        do {                                                                                            \
            OPENVINO_LOG_MATCHING(matcher,                                                              \
                                  ov::util::LevelString::get(),                                         \
                                  OPENVINO_BLOCK_END,                                                   \
                                  OPENVINO_RED,                                                         \
                                  "  NODES' TYPE DIDN'T MATCH. EXPECTED: ",                             \
                                  ov::util::node_version_type_str(*pattern_value),                      \
                                  ". OBSERVED: ",                                                       \
                                  ov::util::node_version_type_str(*graph_value.get_node_shared_ptr())); \
        } while (0)

// src/pass/graph_rewrite.cpp
#    define OPENVINO_LOG_GRAPH_REWRITE1(matcher, node_ptr)                           \
        do {                                                                         \
            OPENVINO_LOG_MATCHING(matcher,                                           \
                                  OPENVINO_BLOCK_BEG,                                \
                                  OPENVINO_YELLOW,                                   \
                                  "  [",                                             \
                                  matcher->get_name(),                               \
                                  "] START: trying to start pattern matching with ", \
                                  ov::util::node_version_type_name_str(*node_ptr));  \
        } while (0)

#    define OPENVINO_LOG_GRAPH_REWRITE2(matcher, status)                    \
        do {                                                                \
            OPENVINO_LOG_MATCHING(matcher,                                  \
                                  OPENVINO_BLOCK_BODY,                      \
                                  '\n',                                     \
                                  OPENVINO_BLOCK_END,                       \
                                  (status ? OPENVINO_GREEN : OPENVINO_RED), \
                                  "  [",                                    \
                                  matcher->get_name(),                      \
                                  "] END: PATTERN MATCHED, CALLBACK ",      \
                                  (status ? "SUCCEDED" : "FAILED"),         \
                                  "\n");                                    \
        } while (0)

#    define OPENVINO_LOG_GRAPH_REWRITE3(matcher, exception)                        \
        do {                                                                       \
            OPENVINO_LOG_MATCHING(matcher,                                         \
                                  OPENVINO_BLOCK_BODY,                             \
                                  '\n',                                            \
                                  OPENVINO_BLOCK_END,                              \
                                  OPENVINO_RED,                                    \
                                  "  [",                                           \
                                  matcher->get_name(),                             \
                                  "] END: PATTERN MATCHED, CALLBACK HAS THROWN: ", \
                                  exception.what());                               \
        } while (0)

#    define OPENVINO_LOG_GRAPH_REWRITE4(matcher)                    \
        do {                                                        \
            OPENVINO_LOG_MATCHING(matcher,                          \
                                  OPENVINO_BLOCK_BODY,              \
                                  '\n',                             \
                                  OPENVINO_BLOCK_END,               \
                                  OPENVINO_RED,                     \
                                  "  [",                            \
                                  matcher->get_name(),              \
                                  "] END: PATTERN DIDN'T MATCH\n"); \
        } while (0)

// src/pattern/matcher.cpp
#    define OPENVINO_LOG_MATCHER1(matcher, pattern_value, graph_value)                                 \
        do {                                                                                           \
            OPENVINO_LOG_MATCHING(matcher,                                                             \
                                  ov::util::LevelString::get(),                                        \
                                  '\n',                                                                \
                                  ov::util::LevelString::get(),                                        \
                                  OPENVINO_BLOCK_BEG,                                                  \
                                  "  MATCHING PATTERN NODE: ",                                         \
                                  ov::util::node_with_arguments(*pattern_value.get_node_shared_ptr()), \
                                  '\n',                                                                \
                                  ov::util::LevelString::get(),                                        \
                                  OPENVINO_BLOCK_BODY_RIGHT,                                           \
                                  " AGAINST  GRAPH   NODE: ",                                          \
                                  ov::util::node_with_arguments(*graph_value.get_node_shared_ptr()));  \
        } while (0)

#    define OPENVINO_LOG_MATCHER2(matcher, idx)                                                                     \
        do {                                                                                                        \
            OPENVINO_LOG_MATCHING(matcher, ++ov::util::LevelString::get());                                         \
            OPENVINO_LOG_MATCHING(matcher, ov::util::LevelString::get()++, OPENVINO_BLOCK_BEG, "  ARGUMENT ", idx); \
        } while (0)

#    define OPENVINO_LOG_MATCHER3(matcher, idx)                                                  \
        do {                                                                                     \
            OPENVINO_LOG_MATCHING(matcher, --ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
            OPENVINO_LOG_MATCHING(matcher,                                                       \
                                  ov::util::LevelString::get()--,                                \
                                  OPENVINO_BLOCK_END,                                            \
                                  OPENVINO_RED,                                                  \
                                  "  ARGUMENT ",                                                 \
                                  idx,                                                           \
                                  " DIDN'T MATCH ");                                             \
        } while (0)

#    define OPENVINO_LOG_MATCHER4(matcher, idx)                                                  \
        do {                                                                                     \
            OPENVINO_LOG_MATCHING(matcher, --ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
            OPENVINO_LOG_MATCHING(matcher,                                                       \
                                  ov::util::LevelString::get()--,                                \
                                  OPENVINO_BLOCK_END,                                            \
                                  OPENVINO_GREEN,                                                \
                                  "  ARGUMENT ",                                                 \
                                  idx,                                                           \
                                  " MATCHED");                                                   \
        } while (0)

#    define OPENVINO_LOG_MATCHER5(matcher, pattern_arg_size, graph_arg_size)                        \
        do {                                                                                        \
            OPENVINO_LOG_MATCHING(matcher,                                                          \
                                  ov::util::LevelString::get(),                                     \
                                  OPENVINO_BLOCK_BODY_RIGHT,                                        \
                                  OPENVINO_RED,                                                     \
                                  " NUMBER OF ARGUMENTS DOESN'T MATCH. EXPECTED IN PATTERN NODE: ", \
                                  pattern_arg_size,                                                 \
                                  ". OBSERVED IN GRAPH NODE: ",                                     \
                                  graph_arg_size);                                                  \
        } while (0)

#    define OPENVINO_LOG_MATCHER6(matcher)                                                                         \
        do {                                                                                                       \
            OPENVINO_LOG_MATCHING(matcher, ++ov::util::LevelString::get());                                        \
            OPENVINO_LOG_MATCHING(matcher, ov::util::LevelString::get(), OPENVINO_BLOCK_BEG, "  NEW PERMUTATION"); \
        } while (0)

#    define OPENVINO_LOG_MATCHER7(matcher)                                                     \
        do {                                                                                   \
            OPENVINO_LOG_MATCHING(matcher, ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
            OPENVINO_LOG_MATCHING(matcher,                                                     \
                                  ov::util::LevelString::get()--,                              \
                                  OPENVINO_BLOCK_END,                                          \
                                  OPENVINO_GREEN,                                              \
                                  "  PERMUTATION MATCHED");                                    \
        } while (0)

#    define OPENVINO_LOG_MATCHER8(matcher)                                                     \
        do {                                                                                   \
            OPENVINO_LOG_MATCHING(matcher, ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
            OPENVINO_LOG_MATCHING(matcher,                                                     \
                                  ov::util::LevelString::get()--,                              \
                                  OPENVINO_BLOCK_END,                                          \
                                  OPENVINO_RED,                                                \
                                  "  PERMUTATION DIDN'T MATCH");                               \
        } while (0)

#    define OPENVINO_LOG_MATCHER9(matcher)                                                                  \
        do {                                                                                                \
            OPENVINO_LOG_MATCHING(matcher, ++ov::util::LevelString::get());                                 \
            OPENVINO_LOG_MATCHING(matcher,                                                                  \
                                  ov::util::LevelString::get(),                                             \
                                  OPENVINO_BLOCK_BEG,                                                       \
                                  "  GRAPH NODE IS NOT COMMUTATIVE, A SINGLE PERMUTATION IS PRESENT ONLY"); \
        } while (0)

#    define OPENVINO_LOG_MATCHER10(matcher, status)                                            \
        do {                                                                                   \
            OPENVINO_LOG_MATCHING(matcher, ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
            OPENVINO_LOG_MATCHING(matcher,                                                     \
                                  ov::util::LevelString::get()--,                              \
                                  OPENVINO_BLOCK_END,                                          \
                                  (status ? OPENVINO_GREEN : OPENVINO_RED),                    \
                                  "  PERMUTATION ",                                            \
                                  (status ? "MATCHED" : "DIDN'T MATCH"));                      \
        } while (0)

#    define OPENVINO_LOG_MATCHER11(matcher)                                                    \
        do {                                                                                   \
            OPENVINO_LOG_MATCHING(matcher, ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
            OPENVINO_LOG_MATCHING(matcher,                                                     \
                                  ov::util::LevelString::get(),                                \
                                  OPENVINO_BLOCK_BODY_RIGHT,                                   \
                                  OPENVINO_RED,                                                \
                                  " NONE OF PERMUTATIONS MATCHED");                            \
        } while (0)

// pattern/op/label.cpp
#    define OPENVINO_LOG_LABEL1(matcher, label_name)            \
        do {                                                    \
            OPENVINO_LOG_MATCHING(matcher,                      \
                                  ov::util::LevelString::get(), \
                                  OPENVINO_BLOCK_END,           \
                                  OPENVINO_GREEN,               \
                                  "  LABEL MATCHED: ",          \
                                  label_name);                  \
        } while (0);

#    define OPENVINO_LOG_LABEL2(matcher, label_name)              \
        do {                                                      \
            OPENVINO_LOG_MATCHING(matcher,                        \
                                  ov::util::LevelString::get()++, \
                                  OPENVINO_BLOCK_BODY_RIGHT,      \
                                  " CHECKING INSIDE LABEL: ",     \
                                  label_name);                    \
        } while (0);

#    define OPENVINO_LOG_LABEL3(matcher)                                                         \
        do {                                                                                     \
            OPENVINO_LOG_MATCHING(matcher, --ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
            OPENVINO_LOG_MATCHING(matcher,                                                       \
                                  ov::util::LevelString::get(),                                  \
                                  OPENVINO_BLOCK_END,                                            \
                                  OPENVINO_GREEN,                                                \
                                  "  LABEL MATCHED");                                            \
        } while (0);

#    define OPENVINO_LOG_LABEL4(matcher)                                                       \
        do {                                                                                   \
            OPENVINO_LOG_MATCHING(matcher, ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
            OPENVINO_LOG_MATCHING(matcher,                                                     \
                                  ov::util::LevelString::get(),                                \
                                  OPENVINO_BLOCK_END,                                          \
                                  OPENVINO_RED,                                                \
                                  "  LABEL DIDN'T MATCH");                                     \
        } while (0);

// pattern/op/optional.cpp
#    define OPENVINO_LOG_OPTIONAL1(matcher, or_node, wrap_node, opt_name)                                        \
        do {                                                                                                     \
            OPENVINO_LOG_MATCHING(matcher,                                                                       \
                                  ov::util::LevelString::get()++,                                                \
                                  OPENVINO_BLOCK_BODY_RIGHT,                                                     \
                                  (or_node == wrap_node ? " LEAVING OPTIONAL AS WRAP TYPE AND TRYING TO MATCH: " \
                                                        : " UNFOLDING OPTIONAL INTO OR AND TRYING TO MATCH: "),  \
                                  opt_name);                                                                     \
        } while (0);

#    define OPENVINO_LOG_OPTIONAL2(matcher)                                                      \
        do {                                                                                     \
            OPENVINO_LOG_MATCHING(matcher, --ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
            OPENVINO_LOG_MATCHING(matcher,                                                       \
                                  ov::util::LevelString::get(),                                  \
                                  OPENVINO_BLOCK_END,                                            \
                                  OPENVINO_GREEN,                                                \
                                  "  OPTIONAL MATCHED");                                         \
        } while (0);

#    define OPENVINO_LOG_OPTIONAL3(matcher)                                                      \
        do {                                                                                     \
            OPENVINO_LOG_MATCHING(matcher, --ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
            OPENVINO_LOG_MATCHING(matcher,                                                       \
                                  ov::util::LevelString::get(),                                  \
                                  OPENVINO_BLOCK_END,                                            \
                                  OPENVINO_RED,                                                  \
                                  "  OPTIONAL DIDN'T MATCH");                                    \
        } while (0);

// pattern/op/or.cpp
#    define OPENVINO_LOG_OR1(matcher, input_size, or_name)      \
        do {                                                    \
            OPENVINO_LOG_MATCHING(matcher,                      \
                                  ov::util::LevelString::get(), \
                                  OPENVINO_BLOCK_BODY_RIGHT,    \
                                  " CHECKING ",                 \
                                  input_size,                   \
                                  " OR BRANCHES: ",             \
                                  or_name);                     \
        } while (0);

#    define OPENVINO_LOG_OR2(matcher, idx, input_value)                                                 \
        do {                                                                                            \
            OPENVINO_LOG_MATCHING(matcher, ++ov::util::LevelString::get());                             \
            OPENVINO_LOG_MATCHING(matcher,                                                              \
                                  ov::util::LevelString::get()++,                                       \
                                  OPENVINO_BLOCK_BEG,                                                   \
                                  "  BRANCH ",                                                          \
                                  idx,                                                                  \
                                  ": ",                                                                 \
                                  ov::util::node_version_type_str(*input_value.get_node_shared_ptr())); \
        } while (0);

#    define OPENVINO_LOG_OR3(matcher, idx)                                                       \
        do {                                                                                     \
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
                                  " HAS MATCHED");                                               \
        } while (0);

#    define OPENVINO_LOG_OR4(matcher, idx)                                                       \
        do {                                                                                     \
            OPENVINO_LOG_MATCHING(matcher, --ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
            OPENVINO_LOG_MATCHING(matcher,                                                       \
                                  ov::util::LevelString::get()--,                                \
                                  OPENVINO_BLOCK_END,                                            \
                                  OPENVINO_RED,                                                  \
                                  "  BRANCH ",                                                   \
                                  idx,                                                           \
                                  " DIDN'T MATCH");                                              \
        } while (0);

#    define OPENVINO_LOG_OR5(matcher)                                                          \
        do {                                                                                   \
            OPENVINO_LOG_MATCHING(matcher, ov::util::LevelString::get(), OPENVINO_BLOCK_BODY); \
            OPENVINO_LOG_MATCHING(matcher,                                                     \
                                  ov::util::LevelString::get(),                                \
                                  OPENVINO_BLOCK_END,                                          \
                                  OPENVINO_RED,                                                \
                                  "  NONE OF OR BRANCHES MATCHED");                            \
        } while (0);

// pattern/op/true.cpp
#    define OPENVINO_LOG_TRUE1(matcher)                         \
        do {                                                    \
            OPENVINO_LOG_MATCHING(matcher,                      \
                                  ov::util::LevelString::get(), \
                                  OPENVINO_BLOCK_END,           \
                                  OPENVINO_GREEN,               \
                                  "  TRUE ALWAYS MATCHES");     \
        } while (0);

// pattern/op/wrap_type.cpp
#    define OPENVINO_LOG_WRAPTYPE1(matcher, pattern_value, graph_value)                                  \
        do {                                                                                             \
            OPENVINO_LOG_MATCHING(matcher,                                                               \
                                  ov::util::LevelString::get(),                                          \
                                  OPENVINO_BLOCK_END,                                                    \
                                  OPENVINO_RED,                                                          \
                                  "  NODES' TYPE DIDN'T MATCH. EXPECTED: ",                              \
                                  ov::util::node_version_type_str(*pattern_value.get_node_shared_ptr()), \
                                  ". OBSERVED: ",                                                        \
                                  ov::util::node_version_type_str(*graph_value.get_node_shared_ptr()));  \
        } while (0);

#    define OPENVINO_LOG_WRAPTYPE2(matcher)                                       \
        do {                                                                      \
            OPENVINO_LOG_MATCHING(matcher,                                        \
                                  ov::util::LevelString::get(),                   \
                                  OPENVINO_BLOCK_END,                             \
                                  OPENVINO_RED,                                   \
                                  "  NODES' TYPE MATCHED, but PREDICATE FAILED"); \
        } while (0);

#    define OPENVINO_LOG_WRAPTYPE3(matcher, num_arguments)                         \
        do {                                                                       \
            OPENVINO_LOG_MATCHING(matcher,                                         \
                                  ov::util::LevelString::get(),                    \
                                  OPENVINO_BLOCK_BODY_RIGHT,                       \
                                  OPENVINO_GREEN,                                  \
                                  " NODES' TYPE and PREDICATE MATCHED. CHECKING ", \
                                  num_arguments,                                   \
                                  " PATTERN ARGUMENTS: ");                         \
        } while (0);

#    define OPENVINO_LOG_WRAPTYPE4(matcher, status)                                                   \
        do {                                                                                          \
            OPENVINO_LOG_MATCHING(matcher,                                                            \
                                  ov::util::LevelString::get(),                                       \
                                  OPENVINO_BLOCK_BODY,                                                \
                                  '\n',                                                               \
                                  ov::util::LevelString::get(),                                       \
                                  OPENVINO_BLOCK_END,                                                 \
                                  (status ? OPENVINO_GREEN : OPENVINO_RED),                           \
                                  (status ? "  ALL ARGUMENTS MATCHED" : "  ARGUMENTS DIDN'T MATCH")); \
        } while (0);

}  // namespace util
}  // namespace ov

#else

#    define OPENVINO_LOG_GENPATTERN1(...) \
        do {                              \
        } while (0)
#    define OPENVINO_LOG_GENPATTERN2(...) \
        do {                              \
        } while (0)
#    define OPENVINO_LOG_GENPATTERN3(...) \
        do {                              \
        } while (0)
#    define OPENVINO_LOG_GENPATTERN4(...) \
        do {                              \
        } while (0)
#    define OPENVINO_LOG_GENPATTERN5(...) \
        do {                              \
        } while (0)
#    define OPENVINO_LOG_GENPATTERN6(...) \
        do {                              \
        } while (0)

#    define OPENVINO_LOG_NODE1(...) \
        do {                        \
        } while (0)
#    define OPENVINO_LOG_NODE2(...) \
        do {                        \
        } while (0)
#    define OPENVINO_LOG_NODE3(...) \
        do {                        \
        } while (0)
#    define OPENVINO_LOG_NODE4(...) \
        do {                        \
        } while (0)
#    define OPENVINO_LOG_NODE5(...) \
        do {                        \
        } while (0)

#    define OPENVINO_LOG_MATCHER1(...) \
        do {                           \
        } while (0)
#    define OPENVINO_LOG_MATCHER2(...) \
        do {                           \
        } while (0)
#    define OPENVINO_LOG_MATCHER3(...) \
        do {                           \
        } while (0)
#    define OPENVINO_LOG_MATCHER4(...) \
        do {                           \
        } while (0)
#    define OPENVINO_LOG_MATCHER5(...) \
        do {                           \
        } while (0)
#    define OPENVINO_LOG_MATCHER6(...) \
        do {                           \
        } while (0)
#    define OPENVINO_LOG_MATCHER7(...) \
        do {                           \
        } while (0)
#    define OPENVINO_LOG_MATCHER8(...) \
        do {                           \
        } while (0)
#    define OPENVINO_LOG_MATCHER9(...) \
        do {                           \
        } while (0)
#    define OPENVINO_LOG_MATCHER10(...) \
        do {                            \
        } while (0)
#    define OPENVINO_LOG_MATCHER11(...) \
        do {                            \
        } while (0)

#    define OPENVINO_LOG_LABEL1(...) \
        do {                         \
        } while (0)
#    define OPENVINO_LOG_LABEL2(...) \
        do {                         \
        } while (0)
#    define OPENVINO_LOG_LABEL3(...) \
        do {                         \
        } while (0)
#    define OPENVINO_LOG_LABEL4(...) \
        do {                         \
        } while (0)

#    define OPENVINO_LOG_GRAPH_REWRITE1(...) \
        do {                                 \
        } while (0)
#    define OPENVINO_LOG_GRAPH_REWRITE2(...) \
        do {                                 \
        } while (0)
#    define OPENVINO_LOG_GRAPH_REWRITE3(...) \
        do {                                 \
        } while (0)
#    define OPENVINO_LOG_GRAPH_REWRITE4(...) \
        do {                                 \
        } while (0)

#    define OPENVINO_LOG_OR1(...) \
        do {                      \
        } while (0)
#    define OPENVINO_LOG_OR2(...) \
        do {                      \
        } while (0)
#    define OPENVINO_LOG_OR3(...) \
        do {                      \
        } while (0)
#    define OPENVINO_LOG_OR4(...) \
        do {                      \
        } while (0)
#    define OPENVINO_LOG_OR5(...) \
        do {                      \
        } while (0)

#    define OPENVINO_LOG_TRUE1(...) \
        do {                        \
        } while (0)

#    define OPENVINO_LOG_OPTIONAL1(...) \
        do {                            \
        } while (0)
#    define OPENVINO_LOG_OPTIONAL2(...) \
        do {                            \
        } while (0)
#    define OPENVINO_LOG_OPTIONAL3(...) \
        do {                            \
        } while (0)

#    define OPENVINO_LOG_WRAPTYPE1(...) \
        do {                            \
        } while (0)
#    define OPENVINO_LOG_WRAPTYPE2(...) \
        do {                            \
        } while (0)
#    define OPENVINO_LOG_WRAPTYPE3(...) \
        do {                            \
        } while (0)
#    define OPENVINO_LOG_WRAPTYPE4(...) \
        do {                            \
        } while (0)

#endif