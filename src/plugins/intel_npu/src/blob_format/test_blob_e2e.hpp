#ifndef _TEST_BLOB_E2E_HPP_
 #define _TEST_BLOB_E2E_HPP_

#include <iostream>
#include <ostream>
#include <istream>
#include <sstream>
#include <vector>

#include "header.hpp"
#include "isection.hpp"

#include "sections/batch_size.hpp"
#include "sections/compat_req_expr.hpp"
#include "sections/elf.hpp"
#include "sections/layouts.hpp"
#include "sections/unknown.hpp"
#include "sections/ws.hpp"

#include "parser.hpp"
#include "test_utils.hpp"

void test_blob_with_header_but_no_sections()
{
    std::cout << std::endl << "RUN: test_blob_with_header_but_no_sections" << std::endl;
    Header header;

    std::stringstream ss;
    header.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 0, "sections found != 0");

    std::cout << "PASSED: test_blob_with_header_but_no_sections" << std::endl;
}

void test_blob_cre_elf()
{
    std::cout << std::endl << "RUN: test_blob_cre_elf" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {AND, ELF, BS, IO_LAYOUTS};
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

    std::cout << "PASSED: test_blob_cre_elf" << std::endl;
}

void test_blob_cre_unknown_ws_bs_layouts()
{
    std::cout << std::endl << "RUN: test_blob_cre_ws_bs_layouts" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {AND, ELF, BS, IO_LAYOUTS};
    CRESection exp_cre_section(expression);

    std::vector<std::shared_ptr<ELFSection>> ws_sub_sections;
    std::vector<uint8_t> init1 = {'I' , 'N', 'I', 'T' , '1'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init1));
    std::vector<uint8_t> init2 = {'I' , 'N', 'I', 'T', '2'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init2));
    std::vector<uint8_t> main0 = {'M' , 'A', 'I', 'N'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(main0));
    WSSection ws_section(ws_sub_sections);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    // Artificial unknown section
    SectionHeader unknown_header;
    unknown_header.type = static_cast<SectionType>(0xFFFF);
    unknown_header.length = 64;
    std::vector<uint8_t> dummy(unknown_header.length);
    unknown_header.serialize(ss);
    ss.write(reinterpret_cast<const char*>(&dummy[0]), unknown_header.length);
    // unknown section serialziation completed
    ws_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    read_blob(ss, sections);

    test_assert(sections.size() == 5, "sections found != 5");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == static_cast<SectionType>(0xFFFF), "unknown section type was not preserved");
    test_assert(sections[2]->header.type == SectionType::WS, "second section is not WS");
    test_assert(sections[3]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[4]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");

    std::cout << "PASSED: test_blob_cre_ws_bs_layouts" << std::endl;
}

void test_blob_cre_unknown_ws_bs_layouts_non_owning()
{
    std::cout << std::endl << "RUN: test_blob_cre_ws_bs_layouts_non_owning" << std::endl;
    Header header;
    
    std::vector<uint16_t> expression = {AND, ELF, BS, IO_LAYOUTS};
    CRESection exp_cre_section(expression);

    std::vector<std::shared_ptr<ELFSection>> ws_sub_sections;
    std::vector<uint8_t> init1 = {'I' , 'N', 'I', 'T' , '1'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init1));
    std::vector<uint8_t> init2 = {'I' , 'N', 'I', 'T', '2'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init2));
    std::vector<uint8_t> main0 = {'M' , 'A', 'I', 'N'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(main0));
    WSSection ws_section(ws_sub_sections);

    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));
    IOLayoutsSection layout_section(input_layouts, output_layouts);

    std::stringstream ss;
    header.serialize(ss);
    exp_cre_section.serialize(ss);
    // Artificial unknown section
    SectionHeader unknown_header;
    unknown_header.type = static_cast<SectionType>(0xFFFF);
    unknown_header.length = 64;
    std::vector<uint8_t> dummy(unknown_header.length);
    unknown_header.serialize(ss);
    ss.write(reinterpret_cast<const char*>(&dummy[0]), unknown_header.length);
    // unknown section serialziation completed
    ws_section.serialize(ss);
    bs_section.serialize(ss);
    layout_section.serialize(ss);

    std::vector<std::shared_ptr<ISection>> sections;
    std::string buffer = ss.str();   // makes a copy
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(buffer.data());
    std::size_t size = buffer.size();
    read_blob_from_data(ptr, size, sections);

    test_assert(sections.size() == 5, "sections found != 5");
    test_assert(sections[0]->header.type == SectionType::CRE, "first section is not CRE");
    test_assert(sections[1]->header.type == static_cast<SectionType>(0xFFFF), "unknown section type was not preserved");
    test_assert(sections[2]->header.type == SectionType::WS, "second section is not WS");
    test_assert(sections[3]->header.type == SectionType::BS, "third section is not BS");
    test_assert(sections[4]->header.type == SectionType::IO_LAYOUTS, "forth section is not IO_LAYOUTS");

    std::cout << "PASSED: test_blob_cre_ws_bs_layouts_non_owning" << std::endl;
}

#endif // _TEST_BLOB_E2E_HPP_
