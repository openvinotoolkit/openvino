// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/isection.hpp"
#include "intel_npu/common/offsets_table.hpp"

using namespace intel_npu;

enum MockTypes {
    MOCK_1 = 4000,
    MOCK_2,
    MOCK_3,
    MOCK_WITH_TABLE,
};

class MockSection_1 : public ISection {
public:
    MockSection_1(double value);

    void write(const std::unique_ptr<BlobWriterInterface>& writer) override;

    double get_value() const;

    static std::shared_ptr<ISection> read(BlobReader* blob_reader, const size_t section_length);

private:
    double value;
};

class MockSection_2 : public intel_npu::ISection {
public:
    MockSection_2(const std::vector<double>& values);

    void write(const std::unique_ptr<BlobWriterInterface>& writer) override;

    std::vector<double> get_values() const;

    static std::shared_ptr<ISection> read(BlobReader* blob_reader, const size_t section_length);

    static uint64_t write_size(size_t n_values, uint64_t start_offset);

    static constexpr uint64_t ALIGNMENT = 64;

private:
    std::vector<double> values;
};

class MockSection_3 : public intel_npu::ISection {
public:
    MockSection_3(const std::shared_ptr<MockSection_1>& section_1, const std::shared_ptr<MockSection_2>& section_2);

    void write(const std::unique_ptr<BlobWriterInterface>& writer) override;

    std::pair<double, std::vector<double>> get_values() const;

    static std::shared_ptr<ISection> read(BlobReader* blob_reader, const size_t section_length);

private:
    std::shared_ptr<MockSection_1> section_1;
    std::shared_ptr<MockSection_2> section_2;
};

// behaves like it supports its own list of supported sections
// would it be overkill for it to include a CRE?
class MockSectionWithTable : public intel_npu::ISection {
public:
    // called only in case of write method
    MockSectionWithTable(std::shared_ptr<MockSection_1> section_1, std::vector<std::shared_ptr<ISection>> reachables);

    void write(const std::unique_ptr<BlobWriterInterface>& writer) override;

    std::shared_ptr<MockSection_1> get_section_1() const;

    const std::unordered_map<SectionID, std::shared_ptr<ISection>>& get_reachables() const;

    OffsetsTable get_embedded_table() const;

    static std::shared_ptr<ISection> read(BlobReader* blob_reader, const size_t section_length);

private:
    // Dispatches to the concrete read() for the given embedded section type.
    static std::shared_ptr<ISection> read_embedded(BlobReader* blob_reader, SectionType type, size_t length);

    // called only in case of read method
    MockSectionWithTable(std::shared_ptr<MockSection_1> section_1,
                         std::unordered_map<SectionID, std::shared_ptr<ISection>> parsed_reachables,
                         OffsetsTable embedded_table);

    struct Entry {
        SectionID id;
        uint64_t offset;
        uint64_t length;
    };

    std::shared_ptr<MockSection_1> section_1;
    std::vector<std::shared_ptr<ISection>> reachables;
    std::unordered_map<SectionID, std::shared_ptr<ISection>> parsed_reachables;
    OffsetsTable embedded_table;
};
