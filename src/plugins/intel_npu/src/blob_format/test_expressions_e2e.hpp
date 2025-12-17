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

// TODO: gotta make some functions which take care of creating the blob so as the test bodies are shorter and do not contain duplicate code
// TODO 2: add registries

void expr_elf_good() {
    std::cout << std::endl << "RUN: expr_elf_good" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, AND, ELF, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 2, "sections found != 2");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");

    std::unordered_set<uint16_t> supported = {CRE, ELF, WS, BS, IO_LAYOUTS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == true, "expression returns false");

    std::cout << "PASSED: expr_elf_good" << std::endl;
}

void expr_elf_bad() {
    std::cout << std::endl << "RUN: expr_elf_bad" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, AND, ELF, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 2, "sections found != 2");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");

    std::unordered_set<uint16_t> supported = {CRE, WS, BS, IO_LAYOUTS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == false, "expression bad");

    std::cout << "PASSED: expr_elf_bad" << std::endl;
}

void expr_elf_bs_good() {
    std::cout << std::endl << "RUN: expr_elf_bs_good" << std::endl;
    Header header;
    
    // to test later: order of serialization vs order of expression
    // what if there is something in the expression in the capabilities found and vice versa
    std::vector<uint16_t> expression = {OPEN, AND, ELF, BS, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 3, "sections found != 3");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");

    std::unordered_set<uint16_t> supported = {CRE, ELF, WS, BS, IO_LAYOUTS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == true, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_good" << std::endl;
}

void expr_elf_bs_bad() {
    std::cout << std::endl << "RUN: expr_elf_bs_bad" << std::endl;
    Header header;
    
    // to test later: order of serialization vs order of expression
    // what if there is something in the expression in the capabilities found and vice versa
    std::vector<uint16_t> expression = {OPEN, AND, ELF, BS, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 3, "sections found != 3");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");

    std::unordered_set<uint16_t> supported = {CRE, WS, BS, IO_LAYOUTS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == false, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_bad" << std::endl;
}

void expr_elf_bs_bad_2() {
    std::cout << std::endl << "RUN: expr_elf_bs_bad_2" << std::endl;
    Header header;
    
    // to test later: order of serialization vs order of expression
    // what if there is something in the expression in the capabilities found and vice versa
    std::vector<uint16_t> expression = {OPEN, AND, ELF, BS, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 3, "sections found != 3");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");

    std::unordered_set<uint16_t> supported = {CRE, ELF, WS, IO_LAYOUTS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == false, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_bad_2" << std::endl;
}

void expr_elf_bs_layouts_good() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_good" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, AND, ELF, BS, IO_LAYOUTS, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 4, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");

    std::unordered_set<uint16_t> supported = {CRE, ELF, WS, BS, IO_LAYOUTS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == true, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_good" << std::endl;
}

void expr_elf_bs_layouts_bad() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_bad" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, AND, ELF, BS, IO_LAYOUTS, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 4, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");

    std::unordered_set<uint16_t> supported = {CRE, ELF, WS, IO_LAYOUTS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == false, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_bad" << std::endl;
}

void expr_elf_bs_layouts_bad_2() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_bad_2" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, AND, ELF, BS, IO_LAYOUTS, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 4, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");

    std::unordered_set<uint16_t> supported = {CRE, ELF, WS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == false, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_bad_2" << std::endl;
}

void expr_elf_bs_layouts_good_2() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_good_2" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OR, BS, IO_LAYOUTS, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 4, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");

    std::unordered_set<uint16_t> supported = {CRE, ELF, WS, BS, IO_LAYOUTS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == true, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_good_2" << std::endl;
}

void expr_elf_bs_layouts_good_2_() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_good_2" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OR, BS, IO_LAYOUTS, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 4, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");

    std::unordered_set<uint16_t> supported = {CRE, ELF, WS, BS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == true, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_good_2" << std::endl;
}

void expr_elf_bs_layouts_bad_2_() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_bad_2_" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OR, BS, IO_LAYOUTS, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 4, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");

    std::unordered_set<uint16_t> supported = {CRE, ELF, WS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == false, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_bad_2_" << std::endl;
}

void expr_elf_bs_layouts_bad_2_2() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_bad_2_2" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OR, BS, IO_LAYOUTS, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 4, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");

    std::unordered_set<uint16_t> supported = {CRE, WS, BS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == false, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_bad_2_2" << std::endl;
}

void expr_elf_bs_layouts_good_3() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_good_3" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, OR, ELF, AND, BS, IO_LAYOUTS, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 4, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");

    std::unordered_set<uint16_t> supported = {CRE, ELF, WS, BS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == true, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_good_3" << std::endl;
}

void expr_elf_bs_layouts_bad_3() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_bad_3" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, OR, ELF, AND, BS, IO_LAYOUTS, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 4, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");

    std::unordered_set<uint16_t> supported = {CRE, WS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == false, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_bad_3" << std::endl;
}

void expr_elf_bs_layouts_good_3_2() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_good_3_2" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, OR, ELF, AND, BS, IO_LAYOUTS, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 4, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");

    std::unordered_set<uint16_t> supported = {CRE, WS, BS, IO_LAYOUTS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == true, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_good_3_2" << std::endl;
}

void expr_elf_bs_layouts_ws_good() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_good" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, OR, WS, OPEN, AND, BS, IO_LAYOUTS, OR, ELF, WS, CLOSE, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::vector<std::shared_ptr<ELFSection>> ws_sub_sections;
    std::vector<uint8_t> init1 = {'I' , 'N', 'I', 'T' , '1'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init1));
    std::vector<uint8_t> init2 = {'I' , 'N', 'I', 'T', '2'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init2));
    std::vector<uint8_t> main0 = {'M' , 'A', 'I', 'N'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(main0));
    WSSection ws_section(ws_sub_sections);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);
    ws_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 5, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");
    test_assert(sections[4]->header.type == SectionType::WS, "fifth section is not WS");

    std::unordered_set<uint16_t> supported = {CRE, ELF, BS, IO_LAYOUTS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == true, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_ws_good" << std::endl;
}

void expr_elf_bs_layouts_ws_bad() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_bad" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, OR, WS, OPEN, AND, BS, IO_LAYOUTS, OR, ELF, WS, CLOSE, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::vector<std::shared_ptr<ELFSection>> ws_sub_sections;
    std::vector<uint8_t> init1 = {'I' , 'N', 'I', 'T' , '1'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init1));
    std::vector<uint8_t> init2 = {'I' , 'N', 'I', 'T', '2'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init2));
    std::vector<uint8_t> main0 = {'M' , 'A', 'I', 'N'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(main0));
    WSSection ws_section(ws_sub_sections);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);
    ws_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 5, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");
    test_assert(sections[4]->header.type == SectionType::WS, "fifth section is not WS");

    std::unordered_set<uint16_t> supported = {CRE, BS, IO_LAYOUTS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == false, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_ws_bad" << std::endl;
}

void expr_elf_bs_layouts_ws_bad_2() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_bad_2" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, OR, WS, OPEN, AND, BS, IO_LAYOUTS, OR, ELF, WS, CLOSE, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::vector<std::shared_ptr<ELFSection>> ws_sub_sections;
    std::vector<uint8_t> init1 = {'I' , 'N', 'I', 'T' , '1'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init1));
    std::vector<uint8_t> init2 = {'I' , 'N', 'I', 'T', '2'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init2));
    std::vector<uint8_t> main0 = {'M' , 'A', 'I', 'N'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(main0));
    WSSection ws_section(ws_sub_sections);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);
    ws_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 5, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");
    test_assert(sections[4]->header.type == SectionType::WS, "fifth section is not WS");

    std::unordered_set<uint16_t> supported = {CRE, BS, ELF};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == false, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_ws_bad_2" << std::endl;
}

void expr_elf_bs_layouts_ws_good_2() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_good_2" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OPEN, OR, BS, IO_LAYOUTS, OR, WS, BS, CLOSE, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::vector<std::shared_ptr<ELFSection>> ws_sub_sections;
    std::vector<uint8_t> init1 = {'I' , 'N', 'I', 'T' , '1'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init1));
    std::vector<uint8_t> init2 = {'I' , 'N', 'I', 'T', '2'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init2));
    std::vector<uint8_t> main0 = {'M' , 'A', 'I', 'N'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(main0));
    WSSection ws_section(ws_sub_sections);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);
    ws_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 5, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");
    test_assert(sections[4]->header.type == SectionType::WS, "fifth section is not WS");

    std::unordered_set<uint16_t> supported = {CRE, ELF, BS, IO_LAYOUTS, WS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == true, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_ws_good_2" << std::endl;
}

void expr_elf_bs_layouts_ws_good_2_() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_good_2_" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OPEN, OR, BS, IO_LAYOUTS, OR, WS, BS, CLOSE, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::vector<std::shared_ptr<ELFSection>> ws_sub_sections;
    std::vector<uint8_t> init1 = {'I' , 'N', 'I', 'T' , '1'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init1));
    std::vector<uint8_t> init2 = {'I' , 'N', 'I', 'T', '2'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init2));
    std::vector<uint8_t> main0 = {'M' , 'A', 'I', 'N'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(main0));
    WSSection ws_section(ws_sub_sections);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);
    ws_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 5, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");
    test_assert(sections[4]->header.type == SectionType::WS, "fifth section is not WS");

    std::unordered_set<uint16_t> supported = {CRE, ELF, IO_LAYOUTS, WS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == true, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_ws_good_2_" << std::endl;
}

void expr_elf_bs_layouts_ws_good_2_2() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_good_2_2" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OPEN, OR, BS, IO_LAYOUTS, OR, WS, BS, CLOSE, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::vector<std::shared_ptr<ELFSection>> ws_sub_sections;
    std::vector<uint8_t> init1 = {'I' , 'N', 'I', 'T' , '1'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init1));
    std::vector<uint8_t> init2 = {'I' , 'N', 'I', 'T', '2'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init2));
    std::vector<uint8_t> main0 = {'M' , 'A', 'I', 'N'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(main0));
    WSSection ws_section(ws_sub_sections);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);
    ws_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 5, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");
    test_assert(sections[4]->header.type == SectionType::WS, "fifth section is not WS");

    std::unordered_set<uint16_t> supported = {CRE, ELF, IO_LAYOUTS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == true, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_ws_good_2_2" << std::endl;
}

void expr_elf_bs_layouts_ws_bad_2_() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_bad_2_" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OPEN, OR, BS, IO_LAYOUTS, OR, WS, BS, CLOSE, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::vector<std::shared_ptr<ELFSection>> ws_sub_sections;
    std::vector<uint8_t> init1 = {'I' , 'N', 'I', 'T' , '1'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init1));
    std::vector<uint8_t> init2 = {'I' , 'N', 'I', 'T', '2'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init2));
    std::vector<uint8_t> main0 = {'M' , 'A', 'I', 'N'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(main0));
    WSSection ws_section(ws_sub_sections);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);
    ws_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 5, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");
    test_assert(sections[4]->header.type == SectionType::WS, "fifth section is not WS");

    std::unordered_set<uint16_t> supported = {CRE, BS, IO_LAYOUTS, WS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == false, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_ws_bad_2_" << std::endl;
}

// TODO: fix expression
void expr_elf_bs_layouts_ws_bad_2_2() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_bad_2_2" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OPEN, OR, BS, IO_LAYOUTS, CLOSE, OPEN, WS, BS, CLOSE, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::vector<std::shared_ptr<ELFSection>> ws_sub_sections;
    std::vector<uint8_t> init1 = {'I' , 'N', 'I', 'T' , '1'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init1));
    std::vector<uint8_t> init2 = {'I' , 'N', 'I', 'T', '2'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init2));
    std::vector<uint8_t> main0 = {'M' , 'A', 'I', 'N'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(main0));
    WSSection ws_section(ws_sub_sections);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);
    ws_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 5, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");
    test_assert(sections[4]->header.type == SectionType::WS, "fifth section is not WS");

    std::unordered_set<uint16_t> supported = {CRE, ELF, IO_LAYOUTS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == false, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_ws_bad_2_2" << std::endl;
}

void expr_elf_bs_layouts_ws_bad_2_3() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_bad_2_3" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OPEN, OR, BS, IO_LAYOUTS, OR, WS, BS, CLOSE, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::vector<std::shared_ptr<ELFSection>> ws_sub_sections;
    std::vector<uint8_t> init1 = {'I' , 'N', 'I', 'T' , '1'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init1));
    std::vector<uint8_t> init2 = {'I' , 'N', 'I', 'T', '2'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init2));
    std::vector<uint8_t> main0 = {'M' , 'A', 'I', 'N'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(main0));
    WSSection ws_section(ws_sub_sections);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);
    ws_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 5, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");
    test_assert(sections[4]->header.type == SectionType::WS, "fifth section is not WS");

    std::unordered_set<uint16_t> supported = {CRE, ELF};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == false, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_ws_bad_2_3" << std::endl;
}

void expr_elf_bs_layouts_ws_good_3() {
    std::cout << std::endl << "RUN: expr_elf_bs_layouts_ws_good_3" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {OPEN, AND, ELF, OPEN, AND, BS, OPEN, AND, OPEN, AND, IO_LAYOUTS, CLOSE, CLOSE, WS, CLOSE, CLOSE};
    CRESection exp_cre_section(expression);
    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::vector<std::shared_ptr<ELFSection>> ws_sub_sections;
    std::vector<uint8_t> init1 = {'I' , 'N', 'I', 'T' , '1'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init1));
    std::vector<uint8_t> init2 = {'I' , 'N', 'I', 'T', '2'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init2));
    std::vector<uint8_t> main0 = {'M' , 'A', 'I', 'N'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(main0));
    WSSection ws_section(ws_sub_sections);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    elf_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);
    ws_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 5, "sections found != 4");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == SectionType::ELF, "second section is not ELF");
    test_assert(sections[2]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[3]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");
    test_assert(sections[4]->header.type == SectionType::WS, "fifth section is not WS");

    std::unordered_set<uint16_t> supported = {CRE, ELF, IO_LAYOUTS, WS, BS};

    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(Evaluator::instance().evaluate(imp_cre_section->expression) == true, "expression returns false");

    std::cout << "PASSED: expr_elf_bs_layouts_ws_good_3" << std::endl;
}

void run_expression_tests() {
    expr_elf_good();
    expr_elf_bs_good();
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
