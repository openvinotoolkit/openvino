#ifndef _BATCH_SIZE_HPP
 #define _BATCH_SIZE_HPP

#include "isection.hpp"
#include "factory.hpp"

struct BatchSizeSection : ISection {
    int64_t batchSize;

    explicit BatchSizeSection(uint64_t batchSize) : batchSize(batchSize) {
        header.type = SectionType::BS;
        header.length = sizeof(batchSize);
    }

    explicit BatchSizeSection(SectionHeader header) : ISection(header) { };

    void serialize(std::ostream& stream) override {
        header.serialize(stream);
        stream.write(reinterpret_cast<const char*>(&batchSize), sizeof(batchSize));
    }

    void read_value(std::istream& stream) override {
        stream.read(reinterpret_cast<char*>(&batchSize), sizeof(batchSize));
    }

    void read_value(const uint8_t* data) override {
        batchSize = *(const int64_t*)data;
    }
};

// Self-registration
namespace sections::batch_size
{
    const bool registered = []()
    {
        SectionFactory::instance().registerSection(
            SectionType::BS,
            [](SectionHeader& header)
            {
                // std::cout << "returning a shared ptr of BatchSizeSection" << std::endl;
                return std::make_shared<BatchSizeSection>(header);
            }
        );
        return true;
    }();
}

#endif // _BATCH_SIZE_HPP