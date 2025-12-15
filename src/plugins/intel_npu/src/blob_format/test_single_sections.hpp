#ifndef _TEST_SINGLE_SECTIONS_
 #define _TEST_SINGLE_SECTIONS_

#include "isection.hpp"

#include "sections/batch_size.hpp"
#include "sections/compat_req_expr.hpp"
#include "sections/elf.hpp"
#include "sections/layouts.hpp"
#include "sections/unknown.hpp"
#include "sections/ws.hpp"

#include "parser.hpp"

inline void test_assert(bool condition, const char* msg = "")
{
    if (!condition)
        {
            std::cout << "Condition failed with msg: " << msg << std::endl;
            throw std::runtime_error(msg);
        }
}


void test_simple_cre_section() {
    std::cout << std::endl << "RUN: test_simple_cre_section" << std::endl;

    std::vector<uint16_t> expression = {AND, ELF, BS, IO_LAYOUTS};
    CRESection exp_cre_section(expression);

    std::stringstream ss;
    exp_cre_section.serialize(ss);
    // std::cout << "serialized section to stream" << std::endl;

    std::vector<std::shared_ptr<ISection>> sections;
    read_sections(ss, sections);
    // std::cout << "read sections from the stream" << std::endl;

    test_assert(sections.size() == 1, "sections found != 1");
    test_assert(sections[0]->header.type = SectionType::CRE), "found section is not an CRE section";
    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(imp_cre_section != nullptr, "failed to cast to an CRESection");
    test_assert(imp_cre_section->expression == expression, "vectors are not identical");

    std::cout << "PASSED: test_simple_cre_section" << std::endl;
}

void test_simple_cre_section_non_owning() {
    std::cout << std::endl << "RUN: test_simple_cre_section_non_owning" << std::endl;

    std::vector<uint16_t> expression = {AND, ELF, BS, IO_LAYOUTS};
    CRESection exp_cre_section(expression);

    std::stringstream ss;
    exp_cre_section.serialize(ss);
    // std::cout << "serialized section to stream" << std::endl;

    std::vector<std::shared_ptr<ISection>> sections;
    std::string buffer = ss.str();   // makes a copy
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(buffer.data());
    std::size_t size = buffer.size();
    read_sections_from_data(ptr, size, sections);
    // std::cout << "read sections from uint8_t* data" << std::endl;

    test_assert(sections.size() == 1, "sections found != 1");
    test_assert(sections[0]->header.type = SectionType::CRE), "found section is not an CRE section";
    auto imp_cre_section = std::dynamic_pointer_cast<CRESection>(sections[0]);
    test_assert(imp_cre_section != nullptr, "failed to cast to an CRESection");

    std::vector<uint16_t> copy(imp_cre_section->expression_view.begin(), imp_cre_section->expression_view.end());
    test_assert( copy == expression, "vectors are not identical");

    std::cout << "PASSED: test_simple_cre_section_non_owning" << std::endl;
}

void test_simple_elf_section() {
    std::cout << std::endl << "RUN: test_simple_elf_section" << std::endl;

    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::stringstream ss;
    elf_section.serialize(ss);
    // std::cout << "serialized section to stream" << std::endl;

    std::vector<std::shared_ptr<ISection>> sections;
    read_sections(ss, sections);
    // std::cout << "read sections from the stream" << std::endl;

    test_assert(sections.size() == 1, "sections found != 1");
    test_assert(sections[0]->header.type = SectionType::ELF), "found section is not an elf section";
    auto elf = std::dynamic_pointer_cast<ELFSection>(sections[0]);
    test_assert(elf != nullptr, "failed to cast to an ELFSection");
    // std::cout << "elf->blob.size(): " << elf->blob.size() << std::endl;
    // for (auto elem : elf->blob)
    //     std::cout << (int)elem << " ";
    // std::cout << std::endl; 
    test_assert(elf->blob == blob, "vectors are not identical");

    std::cout << "PASSED: test_simple_elf_section" << std::endl;
}

void test_simple_elf_section_non_owning() {
    std::cout << std::endl << "RUN: test_simple_elf_section_non_owning" << std::endl;

    std::vector<uint8_t> blob = {1, 2, 3, 4, 5, 6};
    ELFSection elf_section(blob);

    std::stringstream ss;
    elf_section.serialize(ss);
    // std::cout << "serialized section to stream" << std::endl;

    std::vector<std::shared_ptr<ISection>> sections;
    std::string buffer = ss.str();   // makes a copy
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(buffer.data());
    std::size_t size = buffer.size();
    read_sections_from_data(ptr, size, sections);
    // std::cout << "read sections from the stream" << std::endl;

    test_assert(sections.size() == 1, "sections found != 1");
    test_assert(sections[0]->header.type = SectionType::ELF), "found section is not an elf section";
    auto elf = std::dynamic_pointer_cast<ELFSection>(sections[0]);
    test_assert(elf != nullptr, "failed to cast to an ELFSection");
    std::vector<uint8_t> copy(elf->blob_view.begin(), elf->blob_view.end());
    test_assert(copy == blob, "vectors are not identical");

    std::cout << "PASSED: test_simple_elf_section_non_owning" << std::endl;
}

void test_simple_ws_section() {
    std::cout << std::endl << "RUN: test_simple_ws_section" << std::endl;
    std::vector<std::shared_ptr<ELFSection>> ws_sub_sections;

    std::vector<uint8_t> init1 = {'I' , 'N', 'I', 'T' , '1'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init1));
    std::vector<uint8_t> init2 = {'I' , 'N', 'I', 'T', '2'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init2));
    std::vector<uint8_t> main0 = {'M' , 'A', 'I', 'N'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(main0));

    // std::cout << "ws_sub_sections.size() " << ws_sub_sections.size() << std::endl;
    WSSection ws_section(ws_sub_sections);

    std::stringstream ss;
    ws_section.serialize(ss);
    // std::cout << "serialzed section to stream" << std::endl;

    std::vector<std::shared_ptr<ISection>> blob_sections;
    read_sections(ss, blob_sections);
    // std::cout << "read sections from the stream" << std::endl;

    test_assert(blob_sections.size() == 1);
    test_assert(blob_sections[0]->header.type = SectionType::WS);
    auto ws = std::dynamic_pointer_cast<WSSection>(blob_sections[0]);
    test_assert(ws->num_subgraphs == 3);
    test_assert(ws->elf_blobs[0]->blob == init1 , "Mismatch on init1");
    test_assert(ws->elf_blobs[1]->blob == init2 , "Mismatch on init2");
    test_assert(ws->elf_blobs[2]->blob == main0 , "Mismatch on main");

    std::cout << "PASSED: test_simple_ws_section" << std::endl;
}

void test_simple_ws_section_non_owning() {
    std::cout << std::endl << "RUN: test_simple_ws_section_non_owning" << std::endl;
    std::vector<std::shared_ptr<ELFSection>> ws_sub_sections;

    std::vector<uint8_t> init1 = {'I' , 'N', 'I', 'T' , '1'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init1));
    std::vector<uint8_t> init2 = {'I' , 'N', 'I', 'T', '2'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(init2));
    std::vector<uint8_t> main0 = {'M' , 'A', 'I', 'N'};
    ws_sub_sections.push_back(std::make_shared<ELFSection>(main0));

    std::cout << "ws_sub_sections.size() " << ws_sub_sections.size() << std::endl;
    WSSection ws_section(ws_sub_sections);

    std::stringstream ss;
    ws_section.serialize(ss);
    // std::cout << "serialized section to stream" << std::endl;

    std::vector<std::shared_ptr<ISection>> blob_sections;
    std::string buffer = ss.str();   // makes a copy
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(buffer.data());
    std::size_t size = buffer.size();
    read_sections_from_data(ptr, size, blob_sections);
    // std::cout << "read sections from the stream" << std::endl;

    test_assert(blob_sections.size() == 1);
    test_assert(blob_sections[0]->header.type = SectionType::WS);
    auto ws = std::dynamic_pointer_cast<WSSection>(blob_sections[0]);
    test_assert(ws->num_subgraphs == 3);
    {
        std::vector<uint8_t> copy(ws->elf_blobs[0]->blob_view.begin(), ws->elf_blobs[0]->blob_view.end());
        test_assert(copy == init1, "Mismatch on init1");
    }
    {
        std::vector<uint8_t> copy(ws->elf_blobs[1]->blob_view.begin(), ws->elf_blobs[1]->blob_view.end());
        test_assert(copy == init2, "Mismatch on init1");
    }
    {
        std::vector<uint8_t> copy(ws->elf_blobs[2]->blob_view.begin(), ws->elf_blobs[2]->blob_view.end());
        test_assert(copy == main0, "Mismatch on main");
    }

    std::cout << "PASSED: test_simple_ws_section_non_owning" << std::endl;
}

void test_simple_bs_section()
{
    std::cout << std::endl << "RUN: test_simple_bs_section" << std::endl;
    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::stringstream ss;
    bs_section.serialize(ss);
    // std::cout << "serialzed batch size to stream" << std::endl;

    std::vector<std::shared_ptr<ISection>> sections;
    read_sections(ss, sections);
    // std::cout << "read sections from the stream" << std::endl;

    test_assert(sections.size() == 1);
    test_assert(sections[0]->header.type = SectionType::BS);
    auto bs = std::dynamic_pointer_cast<BatchSizeSection>(sections[0]);
    test_assert(bs->batchSize == batchSize);

    std::cout << "PASSED: test_simple_bs_section" << std::endl;
}

void test_simple_bs_section_non_owning()
{
    std::cout << std::endl << "RUN: test_simple_bs_section_non_owning" << std::endl;
    uint64_t batchSize = 3;
    BatchSizeSection bs_section(batchSize);

    std::stringstream ss;
    bs_section.serialize(ss);
    // std::cout << "serialized batch size to stream" << std::endl;

    std::vector<std::shared_ptr<ISection>> sections;
    std::string buffer = ss.str();   // makes a copy
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(buffer.data());
    std::size_t size = buffer.size();
    read_sections_from_data(ptr, size, sections);
    // std::cout << "read sections from the stream" << std::endl;

    test_assert(sections.size() == 1);
    test_assert(sections[0]->header.type = SectionType::BS);
    auto bs = std::dynamic_pointer_cast<BatchSizeSection>(sections[0]);
    test_assert(bs->batchSize == batchSize);

    std::cout << "PASSED: test_simple_bs_section_non_owning" << std::endl;
}

void test_simple_unknown_section()
{
    std::cout << std::endl << "RUN: test_simple_unknown_section" << std::endl;

    SectionHeader header;
    header.type = static_cast<SectionType>(0xFFFF);
    header.length = 64;
    std::vector<uint8_t> dummy(header.length);

    std::stringstream ss;
    header.serialize(ss);
    ss.write(reinterpret_cast<const char*>(&dummy[0]), header.length);
    // std::cout << "serialzed unknown section to stream" << std::endl;

    std::vector<std::shared_ptr<ISection>> sections;
    read_sections(ss, sections);
    // std::cout << "read sections from the stream" << std::endl;

    test_assert(sections.size() == 1);
    test_assert(sections[0]->header.type = static_cast<SectionType>(0xFFFF)); // Type should be preserved
    auto unknown = std::dynamic_pointer_cast<UnknownSection>(sections[0]);
    test_assert(unknown->header.length == 64); // Length should be preserved

    std::cout << "PASSED: test_simple_unknown_section" << std::endl;
}

void test_simple_layouts_section()
{
    std::cout << std::endl << "RUN: test_simple_layouts_section" << std::endl;

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));

    IOLayoutsSection layout_section(input_layouts, output_layouts);
    std::stringstream ss;
    layout_section.serialize(ss);
    // std::cout << "serialzed layouts section to stream" << std::endl;

    std::vector<std::shared_ptr<ISection>> sections;
    read_sections(ss, sections);
    // std::cout << "read sections from the stream" << std::endl;

    test_assert(sections.size() == 1, "Sections found != 1");
    test_assert(sections[0]->header.type = SectionType::IO_LAYOUTS ,  "Section type is not IO_LAYOUTS");
    auto layouts = std::dynamic_pointer_cast<IOLayoutsSection>(sections[0]);
    test_assert(layouts != nullptr, "Failed to cast to IOLayoutsSection");
    test_assert(layouts->input_layouts.size() == 2 , "Mismatch num_input_layouts");
    test_assert(layouts->output_layouts.size() == 1 , "Mismatch num_output_layouts");
    // std::cout << "layouts->output_layouts[0].to_string(): " << layouts->output_layouts[0].to_string() << std::endl;
    test_assert(layouts->input_layouts[0].to_string() == "NCHW" , "Mismatch on input layout 0");
    test_assert(layouts->input_layouts[1].to_string() == "HW" , "Mismatch on input layout 1");
    test_assert(layouts->output_layouts[0].to_string() == "NHWC" , "Mismatch on output layout");
    

    std::cout << "PASSED: test_simple_layouts_section" << std::endl;
}


void test_simple_layouts_section_non_owning()
{
    std::cout << std::endl << "RUN: test_simple_layouts_section_non_owning" << std::endl;

    std::vector<ov::Layout> input_layouts;
    std::vector<ov::Layout> output_layouts;
    input_layouts.push_back(ov::Layout("NCHW"));
    input_layouts.push_back(ov::Layout("HW"));
    output_layouts.push_back(ov::Layout("NHWC"));

    IOLayoutsSection layout_section(input_layouts, output_layouts);
    std::stringstream ss;
    layout_section.serialize(ss);
    // std::cout << "serialzed layouts section to stream" << std::endl;

    std::vector<std::shared_ptr<ISection>> sections;
    std::string buffer = ss.str();   // makes a copy
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(buffer.data());
    std::size_t size = buffer.size();
    read_sections_from_data(ptr, size, sections);
    // std::cout << "read sections from the stream" << std::endl;

    test_assert(sections.size() == 1, "Sections found != 1");
    test_assert(sections[0]->header.type = SectionType::IO_LAYOUTS ,  "Section type is not IO_LAYOUTS");
    auto layouts = std::dynamic_pointer_cast<IOLayoutsSection>(sections[0]);
    test_assert(layouts != nullptr, "Failed to cast to IOLayoutsSection");
    test_assert(layouts->input_layouts.size() == 2 , "Mismatch num_input_layouts");
    test_assert(layouts->output_layouts.size() == 1 , "Mismatch num_output_layouts");
    // std::cout << "layouts->output_layouts[0].to_string(): " << layouts->output_layouts[0].to_string() << std::endl;
    test_assert(layouts->input_layouts[0].to_string() == "NCHW" , "Mismatch on input layout 0");
    test_assert(layouts->input_layouts[1].to_string() == "HW" , "Mismatch on input layout 1");
    test_assert(layouts->output_layouts[0].to_string() == "NHWC" , "Mismatch on output layout");
    

    std::cout << "PASSED: test_simple_layouts_section_non_owning" << std::endl;
}

#endif // _TEST_SINGLE_SECTIONS_