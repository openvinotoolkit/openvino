// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_sections.hpp"

MockSection_1::MockSection_1(double value) : ISection(MockTypes::MOCK_1), value(value) {}

void MockSection_1::write(const std::unique_ptr<BlobWriterInterface>& writer) {
    writer->write(&value, sizeof(value));
}

double MockSection_1::get_value() const {
    return value;
}

std::shared_ptr<ISection> MockSection_1::read(BlobReader* blob_reader, const size_t section_length) {
    double value;
    blob_reader->copy_data_from_source(reinterpret_cast<char*>(&value), sizeof(value));
    return std::make_shared<MockSection_1>(value);
}

MockSection_2::MockSection_2(const std::vector<double>& values) : ISection(MockTypes::MOCK_2), values(values) {}

void MockSection_2::write(const std::unique_ptr<BlobWriterInterface>& writer) {
    uint64_t size = values.size();
    writer->write(&size, sizeof(size));
    for (const double& value : values) {
        const uint64_t pos = static_cast<uint64_t>(writer->get_offset_relative_to_npu_region());
        const uint64_t padding_size = (ALIGNMENT - (pos % ALIGNMENT)) % ALIGNMENT;
        writer->add_padding(padding_size);
        writer->write(&value, sizeof(value));
    }
}

std::vector<double> MockSection_2::get_values() const {
    return values;
}

uint64_t MockSection_2::write_size(size_t n_values, uint64_t start_offset) {
    uint64_t pos = start_offset + sizeof(uint64_t);
    for (size_t i = 0; i < n_values; ++i) {
        const uint64_t padding = (ALIGNMENT - (pos % ALIGNMENT)) % ALIGNMENT;
        pos += padding + sizeof(double);
    }
    return pos - start_offset;
}

std::shared_ptr<ISection> MockSection_2::read(BlobReader* blob_reader, const size_t section_length) {
    uint64_t size;
    blob_reader->copy_data_from_source(reinterpret_cast<char*>(&size), sizeof(size));
    std::vector<double> values(size);
    for (double& value : values) {
        const size_t pos = blob_reader->get_cursor_relative_position();
        const size_t padding = (ALIGNMENT - (pos % ALIGNMENT)) % ALIGNMENT;
        blob_reader->move_cursor_to_relative_position(pos + padding);
        blob_reader->copy_data_from_source(reinterpret_cast<char*>(&value), sizeof(value));
    }
    return std::make_shared<MockSection_2>(values);
}

MockSection_3::MockSection_3(const std::shared_ptr<MockSection_1>& section_1,
                             const std::shared_ptr<MockSection_2>& section_2)
    : ISection(MockTypes::MOCK_3),
      section_1(section_1),
      section_2(section_2) {}

void MockSection_3::write(const std::unique_ptr<BlobWriterInterface>& writer) {
    section_1->write(writer);
    section_2->write(writer);
}

std::pair<double, std::vector<double>> MockSection_3::get_values() const {
    return {section_1->get_value(), section_2->get_values()};
}

std::shared_ptr<ISection> MockSection_3::read(BlobReader* blob_reader, const size_t section_length) {
    auto section_1 = std::dynamic_pointer_cast<MockSection_1>(MockSection_1::read(blob_reader, section_length));
    auto section_2 = std::dynamic_pointer_cast<MockSection_2>(MockSection_2::read(blob_reader, section_length));
    return std::make_shared<MockSection_3>(section_1, section_2);
}

MockSectionWithTable::MockSectionWithTable(std::shared_ptr<MockSection_1> section_1,
                                           std::vector<std::shared_ptr<ISection>> reachables)
    : ISection(MockTypes::MOCK_WITH_TABLE),
      section_1(std::move(section_1)),
      reachables(std::move(reachables)) {}

MockSectionWithTable::MockSectionWithTable(std::shared_ptr<MockSection_1> section_1,
                                           std::unordered_map<SectionID, std::shared_ptr<ISection>> parsed_reachables,
                                           OffsetsTable embedded_table)
    : ISection(MockTypes::MOCK_WITH_TABLE),
      section_1(std::move(section_1)),
      parsed_reachables(std::move(parsed_reachables)),
      embedded_table(std::move(embedded_table)) {}

void MockSectionWithTable::write(const std::unique_ptr<BlobWriterInterface>& writer) {
    const uint64_t number_of_entries = static_cast<uint64_t>(reachables.size());
    const uint64_t payload_start = static_cast<uint64_t>(writer->get_offset_relative_to_current_section());
    // stream position + size of table location field + size of member field section_1
    const uint64_t table_location = payload_start + sizeof(uint64_t) + sizeof(double);

    writer->write(&table_location, sizeof(table_location));
    section_1->write(writer);

    // reserve entries payload
    writer->write(&number_of_entries, sizeof(number_of_entries));
    const auto table_entries_pos = writer->get_offset_relative_to_current_section();
    const uint64_t total_entry_bytes = number_of_entries * OffsetsTable::get_entry_size();
    writer->add_padding(total_entry_bytes);

    // write the reachable sections
    std::vector<Entry> entries;
    std::unordered_map<SectionType, SectionTypeInstance> counters;
    for (const auto& r : reachables) {
        const SectionType type = r->get_section_type();
        const SectionTypeInstance instance = counters[type]++;
        const uint64_t offset = static_cast<uint64_t>(writer->get_offset_relative_to_current_section());
        r->write(writer);
        writer->seek_to_the_end();
        const uint64_t length = static_cast<uint64_t>(writer->get_offset_relative_to_current_section()) - offset;
        entries.push_back({SectionID(type, instance), offset, length});
    }

    // go back to entries payload and write the actual entries
    writer->move_cursor_relative_to_current_section(table_entries_pos);
    for (const auto& e : entries) {
        writer->write(&e.id.type, sizeof(e.id.type));
        writer->write(&e.id.type_instance, sizeof(e.id.type_instance));
        writer->write(&e.offset, sizeof(e.offset));
        writer->write(&e.length, sizeof(e.length));
    }
    writer->seek_to_the_end();
}

std::shared_ptr<MockSection_1> MockSectionWithTable::get_section_1() const {
    return section_1;
}

const std::unordered_map<SectionID, std::shared_ptr<ISection>>& MockSectionWithTable::get_reachables() const {
    return parsed_reachables;
}

OffsetsTable MockSectionWithTable::get_embedded_table() const {
    return embedded_table;
}

std::shared_ptr<ISection> MockSectionWithTable::read_embedded(BlobReader* blob_reader,
                                                              const SectionType type,
                                                              const size_t length) {
    switch (type) {
    case MockTypes::MOCK_2:
        return MockSection_2::read(blob_reader, length);
    case MockTypes::MOCK_3:
        return MockSection_3::read(blob_reader, length);
    default:
        return nullptr;
    }
}

std::shared_ptr<ISection> MockSectionWithTable::read(BlobReader* blob_reader, const size_t section_length) {
    uint64_t table_location;
    blob_reader->copy_data_from_source(reinterpret_cast<char*>(&table_location), sizeof(table_location));

    auto s1 = std::dynamic_pointer_cast<MockSection_1>(MockSection_1::read(blob_reader, sizeof(double)));

    blob_reader->move_cursor_to_relative_position(table_location);

    uint64_t number_of_entries;
    blob_reader->copy_data_from_source(reinterpret_cast<char*>(&number_of_entries), sizeof(number_of_entries));

    OffsetsTable embedded;
    std::vector<Entry> entries;
    for (uint64_t i = 0; i < number_of_entries; i++) {
        SectionType type;
        SectionTypeInstance instance;
        uint64_t offset, length;
        blob_reader->copy_data_from_source(reinterpret_cast<char*>(&type), sizeof(type));
        blob_reader->copy_data_from_source(reinterpret_cast<char*>(&instance), sizeof(instance));
        blob_reader->copy_data_from_source(reinterpret_cast<char*>(&offset), sizeof(offset));
        blob_reader->copy_data_from_source(reinterpret_cast<char*>(&length), sizeof(length));
        const SectionID id(type, instance);
        embedded.add_entry(id, offset, length);
        entries.push_back({id, offset, length});
    }

    std::unordered_map<SectionID, std::shared_ptr<ISection>> parsed;
    for (const auto& e : entries) {
        blob_reader->move_cursor_to_relative_position(e.offset);
        parsed[e.id] = read_embedded(blob_reader, e.id.type, e.length);
    }

    return std::shared_ptr<MockSectionWithTable>(
        new MockSectionWithTable(std::move(s1), std::move(parsed), std::move(embedded)));
}
