// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/*
    This file implements a parser for compatibility strings used in the NPU
    software stack. The compatibility string encodes requirements of a compiled
    blob and metadata such as a target platform, minimum runtime and compiler
    version. The string consists of a human-readable list of name=value pairs
    called attributes. Basic encapsulation across SW layers is supported via
    nested lists. Optional attributes are supported with brace-enclosed
    attribute list.

    The grammar for the compatibility string is defined as follows:

        string  ::=  expr (';' expr)*
        expr    ::=  attr | '{' string '}'
        attr    ::=  name '=' value | name '=' list
        list    ::=  '[' string ('|' string)* ']'
        name    ::=  [a-z][_a-z0-9]*
        value   ::=  [A-Z0-9][_A-Z0-9\.]*

    No whitespaces are allowed.
    Attribute names are lower case. Attribute values are upper case.


    Example usage:

        std::string_view compatibilityString = "name=A;nested=[key=X;ver=1.2|key=Y;ver=1.3];{optional=V1.4}";
        std::array legalAttributes = {"name", "nested"};
        Parser parser(compatibilityString, legalAttributes);
        parser.getAttribute("name");                  // returns "A"
        auto nested = parser.getAttribute("nested");  // returns "[key=X;ver=1.2|key=Y;ver=1.3]"
        Parser::splitList(nested);                    // returns ["key=X;ver=1.2", "key=Y;ver=1.3"]
                                                      // to be parsed separately
        parser.getOptionalAttributes();               // returns { "optional": "V1.4"} as map


    Compatibility string guidelines:

    The compatibility string represents the blob's runtime requirements in a
    concise form: the target hardware and the minimum version of required
    SW/FW runtime components. The compatibility string MUST be sufficient to
    determine whether the blob will run correctly in the current environment without
    accessing the blob content.
    The string is human-readable but is not intended to be interpreted by the
    end user nor edited.  The string consists of attributes which primarily
    represent HW target requirements, runtime component versions and may include
    feature flags.

    The string should include the version of the components contributing to the
    blob's generation (such as compiler). While not strictly required to
    evaluate the runtime requirements a component version may be used to reject
    known bad blobs in future releases or for debugging purposes.
    Each component SHOULD serialize its own version as an attribute in the
    compatibility string. The component encapsulating a blob in a binary
    container MUST serialize the inner blob compatibility string in one of its
    attributes as a list (e.g. `inner=[string]`). The component MUST NOT repeat
    the information contained in the inner blob attributes.

    Attribute names should be short and may use abbreviations. For clarity
    attribute names SHOULD be unique across components, e.g. `version` is not a
    good attribute name. As attributes normally represent versions, the `_ver`
    suffix is not recommended either. For consistency, versions SHOULD be
    encoded as dot-separated integers, e.g. `compiler=8.1`.
    In case of boolean flags the attribute should be set to `1` for true. False
    values SHOULD NOT be serialized. The absence of a flag MUST be treated as
    feature not being required.

    A component MUST reject the string if an attribute name is not recognized,
    so a new requirement is automatically rejected by prior SW releases that did
    not support given feature.
    An unknown optional attribute (enclosed in braces) MUST be ignored. This allows
    for augmenting metadata without breaking compatibility with earlier SW
    versions.

    Version history:
        2026-05-06 Initial version.

*/

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace intel_npu::compat {

class Parser {
public:
    using attr_map_type = std::map<std::string, std::string, std::less<>>;

    Parser() = default;

    template <typename Iterable>
    Parser(std::string_view input, const Iterable& legalAttributeNames);

    template <typename Iterable>
    void parse(std::string_view input, const Iterable& legalAttributeNames);

    const std::string& getAttribute(const std::string& name) const;
    const attr_map_type& getAttributes() const {
        return _attributes;
    }
    const attr_map_type& getOptionalAttributes() const {
        return _optional_attributes;
    }

    template <typename Iterable>
    void validateAttributes(const Iterable& legalAttributeNames) const;

    // Breaks down a list of nested compatibility strings
    static std::vector<std::string> splitList(std::string_view list);

private:
    enum class CaptureMode { ATTR, STRING };

    char peek() const;
    char nextc();
    void expect(char c);
    void reset(std::string_view input, CaptureMode mode = CaptureMode::ATTR);
    void startCapture(CaptureMode mode);
    std::string_view stopCapture(CaptureMode mode);
    std::string_view parseString();
    void parseExpr();
    void parseAttr();
    std::vector<std::string_view> parseList();
    template <typename F>
    std::string_view readChars(F condition);
    std::string_view parseName();
    std::string_view parseValue();
    std::runtime_error errorAt(std::string_view message) const;
    void setAttribute(std::string_view name, std::string_view value);

    // Safe ctype helpers to avoid undefined behavior on signed char
    // clang-format off
    static bool isLower(char c) { return std::islower(static_cast<unsigned char>(c)); }
    static bool isUpper(char c) { return std::isupper(static_cast<unsigned char>(c)); }
    static bool isDigit(char c) { return std::isdigit(static_cast<unsigned char>(c)); }
    static bool isGraph(char c) { return std::isgraph(static_cast<unsigned char>(c)); }
    // clang-format on

    std::string_view::iterator _current;
    std::string_view::iterator _end;

    CaptureMode _captureMode = CaptureMode::ATTR;
    std::string_view::iterator _captureStart;
    std::string_view::iterator _captureEnd;
    int _nesting = 0;   // Nesting level for current capture mode
    int _optional = 0;  // Tracks top-level optional attributes

    attr_map_type _attributes;
    attr_map_type _optional_attributes;
};

template <typename Iterable>
Parser::Parser(std::string_view input, const Iterable& legalAttributeNames) {
    parse(input, legalAttributeNames);
}

template <typename Iterable>
void Parser::parse(std::string_view input, const Iterable& legalAttributeNames) {
    reset(input);
    _attributes.clear();
    _optional_attributes.clear();
    parseString();
    if (peek() != 0) {
        throw errorAt("at the end of the string");
    }
    validateAttributes(legalAttributeNames);
}

inline const std::string& Parser::getAttribute(const std::string& name) const {
    auto it = _attributes.find(name);
    if (it != _attributes.end()) {
        return it->second;
    }
    throw std::runtime_error("Attribute not found: " + name);
}

template <typename Iterable>
void Parser::validateAttributes(const Iterable& legalAttributeNames) const {
    if (legalAttributeNames.size() == 0) {
        return;  // No validation if no legal attribute names provided
    }
    for (const auto& entry : _attributes) {
        auto& name = entry.first;
        if (std::find(legalAttributeNames.begin(), legalAttributeNames.end(), name) == legalAttributeNames.end()) {
            throw std::runtime_error("Illegal attribute: " + name);
        }
    }
}

inline std::vector<std::string> Parser::splitList(std::string_view list) {
    Parser parser;
    parser.reset(list, CaptureMode::STRING);
    auto items = parser.parseList();
    if (parser.peek() != 0) {
        throw parser.errorAt("at the end of the list");
    }
    return std::vector<std::string>(items.begin(), items.end());
}

inline char Parser::peek() const {
    return _current != _end ? *_current : 0;
}

inline char Parser::nextc() {
    assert(_current != _end);
    return *_current++;
}

inline void Parser::expect(char c) {
    if (peek() != c) {
        throw errorAt(std::string("expected '") + c + "'");
    }
    nextc();
}

inline void Parser::reset(std::string_view input, CaptureMode mode) {
    _current = input.begin();
    _end = input.end();
    _optional = 0;
    _nesting = 0;
    _captureMode = mode;
    _captureStart = _end;
    _captureEnd = _end;
}

inline void Parser::startCapture(CaptureMode mode) {
    if (_captureMode == mode) {
        if (_nesting++ == 0) {
            _captureStart = _current;
        }
    }
}

inline std::string_view Parser::stopCapture(CaptureMode mode) {
    if (_captureMode == mode) {
        if (--_nesting == 0) {
            _captureEnd = _current;
            assert(_captureEnd != _captureStart);
            // return std::string_view(_captureStart, _captureEnd);  // C++20
            return std::string_view(&*_captureStart, _captureEnd - _captureStart);
        }
    }
    return {};
}

// string ::= expr (';' expr)*
inline std::string_view Parser::parseString() {
    startCapture(CaptureMode::STRING);
    parseExpr();
    while (peek() == ';') {
        nextc();  // ';'
        parseExpr();
    }
    return stopCapture(CaptureMode::STRING);
}

// expr ::= attr | '{' string '}'
inline void Parser::parseExpr() {
    if (peek() == '{') {
        ++_optional;
        nextc();  // '{'
        parseString();
        expect('}');
        --_optional;
    } else {
        parseAttr();
    }
}

// attr ::= name '=' value | name '=' list
inline void Parser::parseAttr() {
    auto name = parseName();
    expect('=');
    if (peek() == '[') {
        startCapture(CaptureMode::ATTR);
        parseList();
        auto list = stopCapture(CaptureMode::ATTR);
        if (!list.empty()) {
            setAttribute(name, list);
        }
        return;
    }
    auto value = parseValue();
    if (_captureMode == CaptureMode::ATTR && _nesting == 0) {
        setAttribute(name, value);
    }
}

// list ::= '[' string ('|' string)* ']'
inline std::vector<std::string_view> Parser::parseList() {
    std::vector<std::string_view> items;
    expect('[');
    items.push_back(parseString());
    while (peek() == '|') {
        nextc();  // '|'
        items.push_back(parseString());
    }
    expect(']');
    return items;
}

template <typename F>
std::string_view Parser::readChars(F condition) {
    auto start = _current;
    while (peek() != 0 && condition(peek())) {
        nextc();
    }
    // return std::string_view(start, _current); // C++20
    return std::string_view(&*start, _current - start);
}

// name ::= [a-z][_a-z0-9]*
inline std::string_view Parser::parseName() {
    if (!isLower(peek())) {
        throw errorAt("expected attribute name");
    }
    return readChars([](char c) {
        return isLower(c) || isDigit(c) || c == '_';
    });
}

// value ::= [A-Z0-9][_A-Z0-9\.]*
inline std::string_view Parser::parseValue() {
    if (!isUpper(peek()) && !isDigit(peek())) {
        throw errorAt("expected attribute value");
    }
    return readChars([](char c) {
        return isUpper(c) || isDigit(c) || c == '.' || c == '_';
    });
}

inline std::runtime_error Parser::errorAt(std::string_view message) const {
    if (peek() == 0) {
        return std::runtime_error(std::string(message) + " at the end of the string");
    } else {
        std::string msg = "Unexpected character ";
        char c = peek();
        if (isGraph(c)) {  // Printable non-space character
            msg += "'";
            msg += c;
            msg += "': ";
        }
        msg += message;
        return std::runtime_error(std::move(msg));
    }
}

inline void Parser::setAttribute(std::string_view name, std::string_view value) {
    if (_attributes.count(name) != 0 || _optional_attributes.count(name) != 0) {
        throw std::runtime_error("Duplicate attribute: " + std::string(name));
    }
    if (_optional > 0) {
        _optional_attributes.emplace(name, value);
    } else {
        _attributes.emplace(name, value);
    }
}

}  // namespace intel_npu::compat
