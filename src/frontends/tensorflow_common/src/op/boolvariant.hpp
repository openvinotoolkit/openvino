#pragma once

#include <memory>

namespace ov {

class BoolVariant {
public:
    BoolVariant(bool value) : value_(value) {}

    bool get() const {
        return value_;
    }

    static std::shared_ptr<BoolVariant> make(bool value) {
        return std::make_shared<BoolVariant>(value);
    }

private:
    bool value_;
};

}  // namespace ov