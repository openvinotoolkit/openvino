#ifndef _COMPAT_REQ_EXPR_HPP_
 #define _COMPAT_REQ_EXPR_HPP_

#include <vector>
#include <span>

#include "isection.hpp"
#include "factory.hpp"

struct CRESection : ISection {
    std::vector<uint16_t> expression; // can be empty
    std::span<const uint16_t> expression_view;

    explicit CRESection(std::vector<uint16_t>& expression) : expression(expression), expression_view(expression) {
        header.type = SectionType::CRE;
        header.length = expression.size() * sizeof(uint16_t);
        std::cout << "CRESection::ctor(): header.length: " << header.length << std::endl; 
    }

    explicit CRESection(SectionHeader& header) : ISection(header) { };

    void serialize(std::ostream& stream) override {
        header.serialize(stream);
        stream.write(reinterpret_cast<const char*>(expression.data()), header.length);
    }

    void read_value(std::istream& stream) override{
        std::cout << "CRESection::read_value()" << std::endl;
        expression.resize(header.length / sizeof(expression[0]));
        stream.read(reinterpret_cast<char*>(expression.data()), expression.size() * sizeof(expression[0]));
        expression_view = expression;
    }

    void read_value(const uint8_t* data) override{
        expression_view = std::span<const uint16_t>{reinterpret_cast<const uint16_t*>(data), static_cast<size_t>(header.length / sizeof(expression[0]))};
    }
};

// Self-registration
namespace sections::compat_req_expr
{
    const bool registered = []()
    {
        SectionFactory::instance().registerSection(
            SectionType::CRE,
            [](SectionHeader& header)
            {
                std::cout << "returning a shared ptr of CRESection" << std::endl;
                return std::make_shared<CRESection>(header);
            }
        );
        return true;
    }();
}

#endif // _COMPAT_REQ_EXPR_HPP_