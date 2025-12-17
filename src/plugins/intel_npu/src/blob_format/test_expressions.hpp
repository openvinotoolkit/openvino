#pragma once

#include <iostream>
#include <ostream>
#include <sstream>
#include <vector>

#include "header.hpp"
#include "isection.hpp"
#include "logical_operations.hpp"
#include "sections/batch_size.hpp"
#include "sections/compat_req_expr.hpp"
#include "sections/elf.hpp"
#include "sections/layouts.hpp"
#include "sections/ws.hpp"
#include "sections/unknown.hpp"
#include "sections/ws.hpp"
#include "parser.hpp"
#include "test_utils.hpp"

// TODO: maybe some test functions are not being called in run_expression_tests()?

void expr_elf_good() {
    std::cout << std::endl << "RUN: expr_elf_good" << std::endl;
    // it would be great to have all these expressions moved to another file and in comments have them drawn like trees
    // or as logical expressions
    std::vector<uint16_t> expression = {OPEN, AND, ELF, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return true;});

    test_assert(Evaluator::instance().evaluate(expression) == true, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_good" << std::endl;
}

void expr_elf_bad() {
    std::cout << std::endl << "RUN: expr_elf_bad" << std::endl;
    std::vector<uint16_t> expression = {OPEN, AND, ELF, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return false;});

    test_assert(Evaluator::instance().evaluate(expression) == false, "expression bad");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bad" << std::endl;
}

void expr_elf_missing() {
    std::cout << std::endl << "RUN: expr_elf_bad" << std::endl;
    std::vector<uint16_t> expression = {OPEN, AND, ELF, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});

    test_assert(Evaluator::instance().evaluate(expression) == false, "expression bad");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bad" << std::endl;
}

void expr_elf_bs_good() {
    std::cout << std::endl << "RUN: expr_elf_bs_good" << std::endl;
    // to test later: order of serialization vs order of expression
    // what if there is something in the expression in the capabilities found and vice versa
    std::vector<uint16_t> expression = {OPEN, AND, ELF, BS, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return true;});
    Registry::instance().registryEvaluator(BS, [&](){return true;});

    test_assert(Evaluator::instance().evaluate(expression) == true, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_good" << std::endl;
}

void expr_elf_bs_bad() {
    std::cout << std::endl << "RUN: expr_elf_bs_bad" << std::endl;
    // to test later: order of serialization vs order of expression
    // what if there is something in the expression in the capabilities found and vice versa
    std::vector<uint16_t> expression = {OPEN, AND, ELF, BS, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return false;});
    Registry::instance().registryEvaluator(BS, [&](){return true;});

    test_assert(Evaluator::instance().evaluate(expression) == false, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_bad" << std::endl;
}

void expr_elf_bs_bad_2() {
    std::cout << std::endl << "RUN: expr_elf_bs_bad_2" << std::endl;
    // to test later: order of serialization vs order of expression
    // what if there is something in the expression in the capabilities found and vice versa
    std::vector<uint16_t> expression = {OPEN, AND, ELF, BS, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return true;});
    Registry::instance().registryEvaluator(BS, [&](){return false;});

    test_assert(Evaluator::instance().evaluate(expression) == false, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_bad_2" << std::endl;
}

void expr_elf_bs_layouts_good() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_good" << std::endl;
    std::vector<uint16_t> expression = {OPEN, AND, ELF, BS, IO_LAYOUTS, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return true;});
    Registry::instance().registryEvaluator(BS, [&](){return true;});
    Registry::instance().registryEvaluator(IO_LAYOUTS, [&](){return true;});

    test_assert(Evaluator::instance().evaluate(expression) == true, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_good" << std::endl;
}

void expr_elf_bs_layouts_bad() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_bad" << std::endl;
    std::vector<uint16_t> expression = {OPEN, AND, ELF, BS, IO_LAYOUTS, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return true;});
    Registry::instance().registryEvaluator(BS, [&](){return false;});
    Registry::instance().registryEvaluator(IO_LAYOUTS, [&](){return true;});

    test_assert(Evaluator::instance().evaluate(expression) == false, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_bad" << std::endl;
}

void expr_elf_bs_layouts_bad_2() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_bad_2" << std::endl;
    std::vector<uint16_t> expression = {OPEN, AND, ELF, BS, IO_LAYOUTS, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return true;});

    test_assert(Evaluator::instance().evaluate(expression) == false, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_bad_2" << std::endl;
}

void expr_elf_bs_layouts_good_2() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_good_2" << std::endl;
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OR, BS, IO_LAYOUTS, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return true;});
    Registry::instance().registryEvaluator(BS, [&](){return true;});
    Registry::instance().registryEvaluator(IO_LAYOUTS, [&](){return true;});

    test_assert(Evaluator::instance().evaluate(expression) == true, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_good_2" << std::endl;
}

void expr_elf_bs_layouts_good_2_() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_good_2" << std::endl;
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OR, BS, IO_LAYOUTS, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return true;});
    Registry::instance().registryEvaluator(BS, [&](){return true;});
    Registry::instance().registryEvaluator(IO_LAYOUTS, [&](){return true;});

    test_assert(Evaluator::instance().evaluate(expression) == true, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_good_2" << std::endl;
}

void expr_elf_bs_layouts_bad_2_() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_bad_2_" << std::endl;
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OR, BS, IO_LAYOUTS, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return true;});
    Registry::instance().registryEvaluator(BS, [&](){return false;});
    Registry::instance().registryEvaluator(IO_LAYOUTS, [&](){return true;});

    test_assert(Evaluator::instance().evaluate(expression) == false, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_bad_2_" << std::endl;
}

void expr_elf_bs_layouts_bad_2_2() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_bad_2_2" << std::endl;
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OR, BS, IO_LAYOUTS, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return false;});
    Registry::instance().registryEvaluator(BS, [&](){return true;});
    Registry::instance().registryEvaluator(IO_LAYOUTS, [&](){return false;});

    test_assert(Evaluator::instance().evaluate(expression) == false, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_bad_2_2" << std::endl;
}

void expr_elf_bs_layouts_good_3() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_good_3" << std::endl;
    std::vector<uint16_t> expression = {OPEN, OR, ELF, AND, BS, IO_LAYOUTS, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return true;});
    Registry::instance().registryEvaluator(BS, [&](){return true;});
    Registry::instance().registryEvaluator(IO_LAYOUTS, [&](){return false;});

    test_assert(Evaluator::instance().evaluate(expression) == true, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_good_3" << std::endl;
}

void expr_elf_bs_layouts_bad_3() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_bad_3" << std::endl;
    std::vector<uint16_t> expression = {OPEN, OR, ELF, AND, BS, IO_LAYOUTS, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return false;});
    Registry::instance().registryEvaluator(BS, [&](){return false;});
    Registry::instance().registryEvaluator(IO_LAYOUTS, [&](){return false;});

    test_assert(Evaluator::instance().evaluate(expression) == false, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_bad_3" << std::endl;
}

void expr_elf_bs_layouts_good_3_2() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_good_3_2" << std::endl;
    std::vector<uint16_t> expression = {OPEN, OR, ELF, AND, BS, IO_LAYOUTS, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return false;});
    Registry::instance().registryEvaluator(BS, [&](){return true;});
    Registry::instance().registryEvaluator(IO_LAYOUTS, [&](){return true;});

    test_assert(Evaluator::instance().evaluate(expression) == true, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_good_3_2" << std::endl;
}

void expr_elf_bs_layouts_ws_good() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_good" << std::endl;
    std::vector<uint16_t> expression = {OPEN, OR, WS, OPEN, AND, BS, IO_LAYOUTS, OR, ELF, WS, CLOSE, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return true;});
    Registry::instance().registryEvaluator(BS, [&](){return true;});
    Registry::instance().registryEvaluator(IO_LAYOUTS, [&](){return true;});
    Registry::instance().registryEvaluator(WS, [&](){return false;});

    test_assert(Evaluator::instance().evaluate(expression) == true, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_ws_good" << std::endl;
}

void expr_elf_bs_layouts_ws_bad() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_bad" << std::endl;
    std::vector<uint16_t> expression = {OPEN, OR, WS, OPEN, AND, BS, IO_LAYOUTS, OR, ELF, WS, CLOSE, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return false;});
    Registry::instance().registryEvaluator(BS, [&](){return true;});
    Registry::instance().registryEvaluator(IO_LAYOUTS, [&](){return true;});
    Registry::instance().registryEvaluator(WS, [&](){return false;});

    test_assert(Evaluator::instance().evaluate(expression) == false, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_ws_bad" << std::endl;
}

void expr_elf_bs_layouts_ws_bad_2() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_bad_2" << std::endl;
    std::vector<uint16_t> expression = {OPEN, OR, WS, OPEN, AND, BS, IO_LAYOUTS, OR, ELF, WS, CLOSE, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return true;});
    Registry::instance().registryEvaluator(BS, [&](){return true;});
    Registry::instance().registryEvaluator(IO_LAYOUTS, [&](){return false;});
    Registry::instance().registryEvaluator(WS, [&](){return false;});

    test_assert(Evaluator::instance().evaluate(expression) == false, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_ws_bad_2" << std::endl;
}

void expr_elf_bs_layouts_ws_good_2() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_good_2" << std::endl;
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OPEN, OR, BS, IO_LAYOUTS, OR, WS, BS, CLOSE, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return true;});
    Registry::instance().registryEvaluator(BS, [&](){return true;});
    Registry::instance().registryEvaluator(IO_LAYOUTS, [&](){return true;});
    Registry::instance().registryEvaluator(WS, [&](){return true;});

    test_assert(Evaluator::instance().evaluate(expression) == true, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_ws_good_2" << std::endl;
}

void expr_elf_bs_layouts_ws_good_2_() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_good_2_" << std::endl;
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OPEN, OR, BS, IO_LAYOUTS, OR, WS, BS, CLOSE, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return true;});
    Registry::instance().registryEvaluator(BS, [&](){return false;});
    Registry::instance().registryEvaluator(IO_LAYOUTS, [&](){return true;});
    Registry::instance().registryEvaluator(WS, [&](){return true;});

    test_assert(Evaluator::instance().evaluate(expression) == true, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_ws_good_2_" << std::endl;
}

void expr_elf_bs_layouts_ws_good_2_2() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_good_2_2" << std::endl;
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OPEN, OR, BS, IO_LAYOUTS, OR, WS, BS, CLOSE, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return true;});
    Registry::instance().registryEvaluator(BS, [&](){return false;});
    Registry::instance().registryEvaluator(IO_LAYOUTS, [&](){return true;});
    Registry::instance().registryEvaluator(WS, [&](){return false;});

    test_assert(Evaluator::instance().evaluate(expression) == true, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_ws_good_2_2" << std::endl;
}

void expr_elf_bs_layouts_ws_bad_2_() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_bad_2_" << std::endl;
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OPEN, OR, BS, IO_LAYOUTS, OR, WS, BS, CLOSE, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return false;});
    Registry::instance().registryEvaluator(BS, [&](){return true;});
    Registry::instance().registryEvaluator(IO_LAYOUTS, [&](){return true;});
    Registry::instance().registryEvaluator(WS, [&](){return true;});

    test_assert(Evaluator::instance().evaluate(expression) == false, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_ws_bad_2_" << std::endl;
}

// TODO: fix expression
void expr_elf_bs_layouts_ws_bad_2_2() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_bad_2_2" << std::endl;
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OPEN, OR, BS, IO_LAYOUTS, CLOSE, OPEN, WS, BS, CLOSE, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return true;});
    Registry::instance().registryEvaluator(BS, [&](){return false;});
    Registry::instance().registryEvaluator(IO_LAYOUTS, [&](){return true;});
    Registry::instance().registryEvaluator(WS, [&](){return false;});

    test_assert(Evaluator::instance().evaluate(expression) == false, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_ws_bad_2_2" << std::endl;
}

void expr_elf_bs_layouts_ws_bad_2_3() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_bad_2_3" << std::endl;
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OPEN, OR, BS, IO_LAYOUTS, OR, WS, BS, CLOSE, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return true;});
    Registry::instance().registryEvaluator(BS, [&](){return false;});
    Registry::instance().registryEvaluator(IO_LAYOUTS, [&](){return false;});
    Registry::instance().registryEvaluator(WS, [&](){return false;});

    test_assert(Evaluator::instance().evaluate(expression) == false, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_ws_bad_2_3" << std::endl;
}

void expr_elf_bs_layouts_ws_good_3() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_good_3" << std::endl;
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OPEN, AND, BS, OPEN, AND, OPEN, AND, IO_LAYOUTS, CLOSE, CLOSE, WS, CLOSE, CLOSE};

    Registry::instance().registryEvaluator(CRE, [&](){return true;});
    Registry::instance().registryEvaluator(ELF, [&](){return true;});
    Registry::instance().registryEvaluator(BS, [&](){return true;});
    Registry::instance().registryEvaluator(IO_LAYOUTS, [&](){return true;});
    Registry::instance().registryEvaluator(WS, [&](){return true;});

    test_assert(Evaluator::instance().evaluate(expression) == true, "expression returns false");
    Registry::instance().clean();

    std::cout << "PASSED: expr_elf_bs_layouts_ws_good_3" << std::endl;
}

void run_expression_tests() {
    expr_elf_good();
    expr_elf_bs_good();
    expr_elf_missing();
    expr_elf_bs_layouts_good();
    expr_elf_bs_layouts_good_2();
    expr_elf_bs_layouts_good_2_();
    expr_elf_bs_layouts_bad_2_();
    expr_elf_bs_layouts_bad_2_2();
    expr_elf_bs_layouts_good_3();
    expr_elf_bs_layouts_bad_3();
    expr_elf_bs_layouts_good_3_2();
    expr_elf_bs_layouts_ws_good();
    expr_elf_bs_layouts_ws_bad();
    expr_elf_bs_layouts_ws_bad_2();
    expr_elf_bs_layouts_ws_good_2();
    expr_elf_bs_layouts_ws_good_2_();
    expr_elf_bs_layouts_ws_good_2_2();
    expr_elf_bs_layouts_ws_bad_2_();
    expr_elf_bs_layouts_ws_bad_2_2();
    expr_elf_bs_layouts_ws_bad_2_3();
    expr_elf_bs_layouts_ws_good_3();

}
