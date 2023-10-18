// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _MSC_VER
#    define SSCANF_S sscanf_s
#else
#    define SSCANF_S sscanf
#endif

namespace DumbVM {
// Tokenizer greatly simplified following tasks
struct Token {
    union {
        double f;     // `f`: floating point
        char o;       // `o`: op
        char s[128];  // `s`: symbol
    } value;
    char type;
    bool is_op(char op) {
        return type == 'o' && value.o == op;
    }
    bool is_type(char tp) {
        return type == tp;
    }
    bool is_end() {
        return type == '\0';
    }
    void assert_type(char tp) {
        if (type != tp)
            throw std::runtime_error(loc() + " expect token of type '" + tp + "' but got '" + type + "'");
    }
    void assert_op(char op) {
        if (type != 'o')
            throw std::runtime_error(loc() + " expect op '" + op + "' but got type '" + type + "'");
        if (value.o != op)
            throw std::runtime_error(loc() + " expect op '" + op + "' but got op '" + value.o + "'");
    }
    int line;
    int col;
    std::string loc() {
        return std::to_string(line) + ":" + std::to_string(col);
    }
};

std::vector<Token> Tokenize(const std::string& source, const std::string ops) {
    std::vector<Token> tokens;
    int cur_line_start_pos = 0;
    int cur_line = 1;
    size_t pos = 0;

    auto skip_white_spaces = [&]() {
        while (1) {
            while (source[pos] == ' ' || source[pos] == '\t' || source[pos] == '\r' || source[pos] == '\n') {
                if (source[pos] == '\n') {
                    cur_line++;
                    cur_line_start_pos = pos + 1;
                }
                ++pos;
            }
            // comment is special kind of white spaces
            if (source[pos] == '#') {
                while (source[pos] != '\n' && source[pos] != '\0')
                    ++pos;
                if (source[pos] == '\n')
                    ++pos;
            } else {
                break;
            }
        }
    };

    auto next_tok = [&]() {
        Token tok;
        tok.type = '\0';
        int n;
        skip_white_spaces();

        if (source[pos] == '\0')
            return tok;

        // debug purpose
        tok.line = cur_line;
        tok.col = pos - cur_line_start_pos;

        if (ops.find(source[pos]) != std::string::npos) {
            tok.value.o = source[pos];
            tok.type = 'o';
            pos++;
            return tok;
        }

        if (std::isalpha(source[pos]) || source[pos] == '_') {
            // must be symbol
            int i = 0;
            while (std::isalnum(source[pos]) || source[pos] == '_' || source[pos] == '.')
                tok.value.s[i++] = source[pos++];
            tok.value.s[i++] = '\0';
            tok.type = 's';
            return tok;
        }

        if (SSCANF_S(&source[pos], "%lf%n", &tok.value.f, &n) == 1) {
            pos += n;
            tok.type = 'f';
            return tok;
        }
        return tok;
    };

    do {
        tokens.push_back(next_tok());
    } while (tokens.back().type != '\0');

    return tokens;
}

// FunctionVM's source code is simplified, it only has assign statement
//
//    var = expression;
//
// the grammar of expression is similar to Basic language;
// the VM evaluate the expression statements sequentially and update variables;
// statements are compiled into an IR before execution
//
// expression is recurisvely defined using
//      1.sum of product term:
//          s = p1 + p2 + ...
//      2.product of element term:
//          p = e1*e2*...
//      3.element term:
//          e = number | (s) | func(s1,s2,...)
//
struct VM {
    // essentially it's a function based system, but expression
    // system provides extra syntax sugar for commonly used binary
    // function such as addition/subtract/multiply/divide. at runtime
    // all instructions are function calls (including +-*/):
    //   `5+a*6` is compiled into following IR
    //
    //          t0=5
    //          t1=mul(a, 6)
    //          t2=add(t0,t1)
    //
    // we can see there is no reason for t0 & t1 to keep exist
    // once t2 is derived from them, that's why stack based VM
    // is better choice, it doesn't need to maintain lifecycle of
    // temp-variables or registers, each ALU instruction consumes
    // or pops N elements from the top of stack and produce or push
    // 1 element onto the stack, the IR would be like following:
    //
    //          push 5
    //          push a
    //          push 6
    //          mul  # it pops a and 6, calculates result and push it
    //          add  # it pops result of mul and 5, calculates result and push it
    //
    // at the end of execution, the final result of expression is
    // always on top of the stack and all temporary values are popped out.
    //
    struct States {
        // enssential machine states:
        // 1.stack replaces register files
        // 2.vars is memory

        std::deque<double> stack;
        std::map<std::string, double> vars;

        // const is also implemented as specially-named variables
        int nconst = 0;
        std::string add_const(double const_value) {
            for (auto& v : vars) {
                if (v.second == const_value)
                    return v.first;
            }
            std::string const_name = "const";
            const_name += std::to_string(nconst++);
            vars[const_name] = const_value;
            return const_name;
        }

        std::string find_var(double* p) {
            for (auto& v : vars) {
                if (&(v.second) == p)
                    return v.first;
            }
            return "?";
        }
        void show() {
            std::cout << "===== State.stack: " << stack.size() << " entries:" << std::endl;
            for (int i = 0; i < stack.size(); i++)
                std::cout << "\t[" << i << "]\t" << stack[i] << std::endl;
            std::cout << "===== State.vars: " << vars.size() << " entries:" << std::endl;
            for (auto& v : vars)
                std::cout << "\t" << v.first << ":\t" << v.second << "\t @" << &(v.second) << std::endl;
        }
    } states;

    using function = std::function<void(States&, double&)>;
    using F = std::pair<std::string, function>;
    std::map<std::string, F> all_functions;

    struct Instruction {
        const F f;
        double* dst;
        Instruction(const F& f, double* dst) : f(f), dst(dst) {}
        Instruction(const std::string& inst, double* dst) : f(std::make_pair(inst, function())), dst(dst) {}
    };
    std::vector<Instruction> codes;

    virtual void call(const std::string& inst, States& states, double& var) {
        throw std::runtime_error(std::string("un-implemented instruction '") + inst + "' is met");
    }

    VM& disassemble() {
        states.show();
        std::cout << "==== Disassemble:" << std::endl;
        for (int i = 0; i < codes.size(); i++) {
            auto& inst = codes[i];
            std::cout << "#" << i << ":\t" << inst.f.first;
            if (inst.dst)
                std::cout << "\t" << states.find_var(inst.dst) << "\t  @" << std::hex << inst.dst;
            std::cout << std::endl;
        }
        return *this;
    }

    void codegen(const char* inst, std::string var = "") {
        double* dst = nullptr;
        if (!var.empty())
            dst = &(states.vars[var]);
        // un-registered function will be implemented with less efficient callbacks
        if (all_functions.count(inst))
            codes.emplace_back(all_functions[inst], dst);
        else
            codes.emplace_back(std::string(inst), dst);
    }

    void register_function(std::string name, function f) {
        assert(all_functions.count(name) == 0);
        all_functions[name] = std::make_pair(name, f);
    }

    // built-in function
    static void pop(States& s, double& var) {
        var = s.stack.back();
        s.stack.pop_back();
    }
    static void push(States& s, double& var) {
        s.stack.push_back(var);
    }
    static void add(States& s, double& var) {
        auto a = s.stack.back();
        s.stack.pop_back();
        s.stack.back() += a;  // optimized, saved 1 pop + 1 push.
    }
    static void sub(States& s, double& var) {
        auto a = s.stack.back();
        s.stack.pop_back();
        s.stack.back() -= a;  // optimized, saved 1 pop + 1 push.
    }
    static void mul(States& s, double& var) {
        auto a = s.stack.back();
        s.stack.pop_back();
        s.stack.back() *= a;  // optimized, saved 1 pop + 1 push.
    }
    static void div(States& s, double& var) {
        auto a = s.stack.back();
        s.stack.pop_back();
        s.stack.back() /= a;  // optimized, saved 1 pop + 1 push.
    }
    static void min(States& s, double& var) {
        auto a = s.stack.back();
        s.stack.pop_back();
        s.stack.back() = std::min(s.stack.back(), a);  // optimized, saved 1 pop + 1 push.
    }

    std::vector<Token>::iterator it;

    VM() {
        register_function("push", push);
        register_function("pop", pop);
        register_function("add", add);
        register_function("sub", sub);
        register_function("mul", mul);
        register_function("div", div);
        register_function("min", min);
    }

    VM& compile(const std::string& source) {
        std::vector<Token> toks = Tokenize(source, "()+-*/,=;");
        it = toks.begin();

        while (!it->is_end())
            compile_assign_st();
        return *this;
    }

    VM& assert_eq(const std::string& var, double v) {
        assert(states.vars[var] == v);
        return *this;
    }

    VM& execute() {
        for (auto& inst : codes) {
            if (inst.f.second)
                inst.f.second(states, *inst.dst);
            else
                call(inst.f.first, states, *inst.dst);
        }

        return *this;
    }

    // var = expression;
    void compile_assign_st() {
        it->assert_type('s');
        std::string var_name = it->value.s;
        ++it;
        it->assert_op('=');
        ++it;
        compile_expr_sum();
        codegen("pop", var_name);  // value of expr is on top of the stack
        it->assert_op(';');
        ++it;
    }

    void compile_expr_sum() {
        compile_expr_prod();
        while (1) {
            if (it->is_op('+')) {
                ++it;
                compile_expr_prod();
                codegen("add");
            } else if (it->is_op('-')) {
                ++it;
                compile_expr_prod();
                codegen("sub");
            } else {
                break;
            }
        }
    }

    void compile_expr_prod() {
        compile_term();
        while (1) {
            if (it->is_op('*')) {
                ++it;
                compile_term();
                codegen("mul");
            } else if (it->is_op('/')) {
                ++it;
                compile_term();
                codegen("div");
            } else {
                break;
            }
        }
    }

    void compile_term() {
        if (it->is_op('(')) {
            ++it;
            compile_expr_sum();
            it->assert_op(')');
            ++it;
            return;
        }

        int sign = 1;
        if (it->is_op('-')) {
            sign = -1;
            ++it;
        }
        if (it->is_type('f')) {
            auto cn = states.add_const(sign * it->value.f);
            codegen("push", cn);
            ++it;
            return;
        }
        if (it->is_type('s')) {
            auto name = it->value.s;
            // function name or variable
            if ((it + 1)->is_op('(')) {
                ++it;
                ++it;
                // (expr1, expr2,...)
                while (1) {
                    if (it->is_op(')')) {
                        ++it;
                        break;
                    }
                    compile_expr_sum();
                    if (it->is_op(','))
                        ++it;
                }
                // make the function call
                codegen(name);
            } else {
                ++it;
                codegen("push", name);
            }
            if (sign < 0) {
                auto cn = states.add_const(sign);
                codegen("mul", cn);
            }
        }
    }
};
};  // namespace DumbVM