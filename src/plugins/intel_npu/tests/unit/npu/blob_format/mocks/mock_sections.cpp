// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_sections.hpp"

#include <algorithm>

MockSection_1::MockSection_1(double value) : ISection(MockTypes::MOCK_1), value(value) {}

void MockSection_1::write(std::ostream& stream, BlobWriter* writer) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

double MockSection_1::get_value() const {
    return value;
}

bool MockSection_1::lazy_check() const {
    return value < VALID_THRESHOLD;
}

std::shared_ptr<ISection> MockSection_1::read(BlobReader* blob_reader, const size_t section_length) {
    double value;
    blob_reader->copy_data_from_source(reinterpret_cast<char*>(&value), sizeof(value));
    return std::make_shared<MockSection_1>(value);
}

MockSection_2::MockSection_2(const std::vector<double>& values) : ISection(MockTypes::MOCK_2), values(values) {}

void MockSection_2::write(std::ostream& stream, BlobWriter* writer) {
    static constexpr char zeros[ALIGNMENT] = {};
    uint64_t size = values.size();
    stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const double& value : values) {
        const uint64_t pos = static_cast<uint64_t>(writer->get_stream_relative_position(stream));
        const uint64_t padding = (ALIGNMENT - (pos % ALIGNMENT)) % ALIGNMENT;
        stream.write(zeros, static_cast<std::streamsize>(padding));
        stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
}

std::vector<double> MockSection_2::get_values() const {
    return values;
}

bool MockSection_2::lazy_check() const {
    return std::all_of(values.begin(), values.end(), [](double v) {
        return v < VALID_THRESHOLD;
    });
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

void MockSection_3::write(std::ostream& stream, BlobWriter* writer) {
    section_1->write(stream, writer);
    section_2->write(stream, writer);
}

std::pair<double, std::vector<double>> MockSection_3::get_values() const {
    return {section_1->get_value(), section_2->get_values()};
}

bool MockSection_3::lazy_check() const {
    return section_1->lazy_check() && section_2->lazy_check();
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

void MockSectionWithTable::write(std::ostream& stream, BlobWriter* writer) {
    const uint64_t number_of_entries = static_cast<uint64_t>(reachables.size());
    const uint64_t payload_start = static_cast<uint64_t>(writer->get_stream_relative_position(stream));
    // stream position + size of table location field + size of member field section_1
    const uint64_t table_location = payload_start + sizeof(uint64_t) + sizeof(double);

    stream.write(reinterpret_cast<const char*>(&table_location), sizeof(table_location));
    section_1->write(stream, writer);

    // reserve entries payload
    stream.write(reinterpret_cast<const char*>(&number_of_entries), sizeof(number_of_entries));
    const auto table_entries_pos = stream.tellp();
    const uint64_t total_entry_bytes = number_of_entries * OffsetsTable::get_entry_size();
    std::vector<char> zeros(total_entry_bytes, 0);
    stream.write(zeros.data(), static_cast<std::streamsize>(total_entry_bytes));

    // write the reachable sections
    std::vector<Entry> entries;
    std::unordered_map<SectionType, SectionTypeInstance> counters;
    for (const auto& r : reachables) {
        const SectionType type = r->get_section_type();
        const SectionTypeInstance instance = counters[type]++;
        const uint64_t offset = static_cast<uint64_t>(writer->get_stream_relative_position(stream));
        r->write(stream, writer);
        stream.seekp(0, std::ios_base::end);
        const uint64_t length = static_cast<uint64_t>(writer->get_stream_relative_position(stream)) - offset;
        entries.push_back({SectionID(type, instance), offset, length});
    }

    // go back to entries payload and write the actual entries
    stream.seekp(table_entries_pos);
    for (const auto& e : entries) {
        stream.write(reinterpret_cast<const char*>(&e.id.type), sizeof(e.id.type));
        stream.write(reinterpret_cast<const char*>(&e.id.type_instance), sizeof(e.id.type_instance));
        stream.write(reinterpret_cast<const char*>(&e.offset), sizeof(e.offset));
        stream.write(reinterpret_cast<const char*>(&e.length), sizeof(e.length));
    }
    stream.seekp(0, std::ios_base::end);
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

void MockSectionWithTable::set_driver_probe(DriverProbe probe) {
    query_driver = std::move(probe);
}

bool MockSectionWithTable::lazy_check() const {
    if (!section_1->lazy_check()) {
        return false;
    }

    // query the driver for each section type
    std::unordered_set<SectionType> probed;
    for (const auto& [id, section] : parsed_reachables) {
        const SectionType t = id.type;
        if (probed.insert(t).second && query_driver && !query_driver(t)) {
            return false;
        }
    }

    return true;
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
