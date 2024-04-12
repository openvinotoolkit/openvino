// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <stack>
#include <string>

#include "utils.hpp"

namespace cldnn {
namespace ocl {

// Evaluate a simple C expression
//  e.g. shape_info[7] % 16 == 0, (shape_info[13] + (shape_info[1] + shape_infop[10])) % 32 == 0
// Current supported operators are +, -, *, /, %, ==, != and array accessor.
// Symbol names cannot start with a number.
int32_t evaluateJIT(const std::string& expression, const int32_t* shape_info_ptr) {
    enum TokenType {
        NUM,    // number
        SYM     // symbol (currently, only one symbol `shape_info`)
    };

    struct JitToken {
        TokenType type;
        int32_t value;
    };

    std::stack<struct JitToken> tokens;
    std::stack<char> ops;

    auto opPrecedence = [](char op) -> int32_t {
        switch (op) {
            case '*':
            case '/':
            case '%':
                return 1;
            case '+':
            case '-':
                return 2;
            case '=':
            case '!':
                return 3;
            default:
                return 4;
        }
    };

    auto calcSingleOp = [&ops, &tokens]() {
        char op = ops.top();
        auto Rval = tokens.top();
        if (Rval.type != TokenType::NUM) {
            OPENVINO_THROW("[GPU] evaluateJIT - incorrect R-value: ", Rval.type , ": " , Rval.value);
        }
        tokens.pop();
        auto Lval = tokens.top();
        if (Lval.type != TokenType::NUM) {
            OPENVINO_THROW("[GPU] evaluateJIT - incorrect L-value: ", Lval.type , ": " , Lval.value);
        }
        tokens.pop();
        switch (op) {
            case '+':
                tokens.push({TokenType::NUM, (Lval.value + Rval.value)}); break;
            case '-':
                tokens.push({TokenType::NUM, (Lval.value - Rval.value)}); break;
            case '*':
                tokens.push({TokenType::NUM, (Lval.value * Rval.value)}); break;
            case '/':
                tokens.push({TokenType::NUM, (Lval.value / Rval.value)}); break;
            case '%':
                tokens.push({TokenType::NUM, (Lval.value % Rval.value)}); break;
            case '=':
                tokens.push({TokenType::NUM, (Lval.value == Rval.value)}); break;
            case '!':
                tokens.push({TokenType::NUM, (Lval.value != Rval.value)}); break;
        }
        ops.pop();
    };

    for (size_t i = 0; i < expression.length(); i++) {
        const char& ch = expression[i];

        switch (ch) {
            case ' ':
                continue;
            case '(':
            case '[':
                ops.push(ch);
                break;
            case ')':
                while (!ops.empty() && ops.top() != '(') {
                    calcSingleOp();
                }
                if (!ops.empty() && ops.top() == '(')
                    ops.pop();
                break;
            case ']':
                while (!ops.empty() && ops.top() != '[') {
                    calcSingleOp();
                }
                if (!ops.empty() && ops.top() == '[') {
                    auto index = tokens.top();
                    if (index.type != TokenType::NUM) {
                        OPENVINO_THROW("[GPU] evaluateJIT - incorrect array index type: ", expression);
                        break;
                    }
                    tokens.pop();
                    auto symbol = tokens.top();
                    if (symbol.type != TokenType::SYM) {
                        OPENVINO_THROW("[GPU] evaluateJIT - incorrect array name: ", expression);
                        break;
                    }
                    tokens.pop();
                    OPENVINO_ASSERT(shape_info_ptr != nullptr, "[GPU] evaluateJIT - shape_info_ptr should not be nullptr.");
                    tokens.push({TokenType::NUM, shape_info_ptr[index.value]});
                    ops.pop();
                }
                break;
            case '+':
            case '-':
            case '*':
            case '/':
            case '%':
                while (!ops.empty() && (opPrecedence(ops.top()) <= opPrecedence(ch))) {
                    calcSingleOp();
                }
                ops.push(ch);
                break;
            case '=':
            case '!':
                if (expression[i + 1] == '=') {
                    while (!ops.empty() && (opPrecedence(ops.top()) <= opPrecedence(ch))) {
                        calcSingleOp();
                    }
                    ops.push(ch);
                    i += 1;
                } else {
                    OPENVINO_THROW("[GPU] evaluateJIT - unsupported operator: ", expression[i], expression[i + 1]);
                    break;
                }
                break;
            default:
                if (isdigit(ch)) {
                    int32_t value = 0;
                    while (i < expression.length() && isdigit(expression[i])) {
                        value = (value * 10) + (expression[i] - '0');
                        i += 1;
                    }
                    tokens.push({TokenType::NUM, value});
                } else {
                    if (expression.substr(i, 10).compare("shape_info") == 0) {
                        i += 10;
                        tokens.push({TokenType::SYM, 0});
                    } else {
                        OPENVINO_THROW("[GPU] evaluateJIT - unrecognized symbol name: ", expression);
                        break;
                    }
                }
                i -= 1;
        }
    }

    while (!ops.empty()) {
        calcSingleOp();
    }

    return tokens.top().value;
}

}  // namespace ocl
}  // namespace cldnn
