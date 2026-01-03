// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "isection.hpp"

namespace intel_npu {

class CRE final {
public:
    using Token = uint16_t;

    static constexpr Token AND = 50000;
    static constexpr Token OR = 50001;
    static constexpr Token OPEN = 50002;
    static constexpr Token CLOSE = 50003;

    static inline const std::unordered_set<Token> RESERVED_TOKENS{AND, OR, OPEN, CLOSE};

    static std::unordered_set<Token> plugin_capabilities;

    // TODO tie this to the plugin object
    static void register_plugin_capability(const Token capability_id);

    static void check_plugin_capability(const Token capability_id);

    CRE();

    void append_to_expression(const CRE::Token requirement_token);

    void append_to_expression(const std::vector<CRE::Token>& requirement_tokens);

    size_t write(std::ostream& stream);

    // bool read_and_validate(std::istream& stream) override;

private:
    std::vector<Token> m_expression;
};

class CRESection final : public ISection {
public:
    CRESection();

    void write(std::ostream& stream, BlobWriter* writer) override;

    // void read(BlobReader* reader) override;

    void append_to_expression(const CRE::Token requirement_token);

    void append_to_expression(const std::vector<CRE::Token>& requirement_tokens);

private:
    CRE m_cre;
};

}  // namespace intel_npu
