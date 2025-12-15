#ifndef _FACTORY_HPP
 #define _FACTORY_HPP

#include <functional>
#include <unordered_map>
#include "isection.hpp"
#include "sections/unknown.hpp"

class SectionFactory
{
public:
    using Creator = std::function<std::shared_ptr<ISection>(SectionHeader&)>;

    static SectionFactory& instance()
    {
        static SectionFactory factory;
        return factory;
    }

    void registerSection(const SectionType type, Creator creator)
    {
        // std::cout << "Register function for " << type << std::endl;
        _creators[type] = std::move(creator);
    }

    std::shared_ptr<ISection> create(SectionHeader& header) const
    {
        auto it = _creators.find(header.type);
        if (it == _creators.end())
        {
            // throw std::runtime_error("Unknown section: " + header.type);
            return std::make_shared<UnknownSection>(header);
        }

        // std::cout << "Calling create function for " << header.type << std::endl;
        return it->second(header);
    }

private:
    std::unordered_map<SectionType, Creator> _creators;
};

#endif // _FACTORY_HPP