// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include "fake_quantize_on_data.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class DequantizationOperations {
public:
    class Convert {
    public:
        Convert();
        Convert(const ngraph::element::Type outPrecision, const bool addDeqAttr = true);
        bool empty() const noexcept;
        bool equal(const DequantizationOperations::Convert& value) const noexcept;
        bool operator==(const Convert& value) const noexcept {
            return equal(value);
        }

        ngraph::element::Type outPrecision = element::undefined;
        bool addDequantizationAttribute = true;
    private:
        bool isEmpty;
    };

    class Subtract {
    public:
        Subtract();
        Subtract(const float value, const bool addDeqAttr = true);
        Subtract(const std::vector<float>& values, const bool addDeqAttr = true);
        Subtract(const std::vector<float>& values, const ngraph::element::Type outPrecision, const bool addDeqAttr = true);
        Subtract(
            const std::vector<float>& values,
            const ngraph::element::Type outPrecision,
            const ngraph::Shape& constantShape,
            const bool addDequantizationAttribute = true,
            const size_t constantIndex = 1ul,
            const ngraph::element::Type constantPrecision = ngraph::element::undefined,
            const bool addConvert = false,
            const std::vector<std::string>& attributes = {},
            const std::vector<std::string>& convertAttributes = {});
        bool empty() const noexcept;
        bool equal(const DequantizationOperations::Subtract& value) const noexcept;
        bool operator==(const Subtract& value) const noexcept {
            return equal(value);
        }
        void erase() {
            isEmpty = true;
        }
        Subtract& setConstantPrecision(const ngraph::element::Type& precision);

        std::vector<float> values;
        ngraph::element::Type outPrecision = ngraph::element::undefined;
        ngraph::Shape constantShape;
        bool constantShapeIsDefined = false;
        bool addDequantizationAttribute = true;
        size_t constantIndex = 1ul;
        ngraph::element::Type constantPrecision = ngraph::element::undefined;
        bool addConvert = false;
        std::vector<std::string> attributes;
        std::vector<std::string> convertAttributes;

    private:
        bool isEmpty;
    };

    class Multiply {
    public:
        Multiply();
        Multiply(const float value);
        Multiply(const std::vector<float>& values);
        Multiply(const std::vector<float>& values, const ngraph::element::Type outPrecision);
        Multiply(
            const std::vector<float>& values,
            const ngraph::element::Type outPrecision,
            const ngraph::Shape& constantShape,
            const bool addDequantizationAttribute = true,
            const size_t constantIndex = 1ul,
            const ngraph::element::Type constantPrecision = ngraph::element::undefined);
        bool empty() const noexcept;
        bool equal(const DequantizationOperations::Multiply& value) const noexcept;
        bool operator==(const Multiply& value) const noexcept {
            return equal(value);
        }
        Multiply& setConstantPrecision(const ngraph::element::Type& precision);

        std::vector<float> values;
        ngraph::element::Type outPrecision = ngraph::element::undefined;
        ngraph::Shape constantShape;
        bool constantShapeIsDefined = false;
        bool addDequantizationAttribute = true;
        size_t constantIndex = 1ul;
        ngraph::element::Type constantPrecision = ngraph::element::undefined;

    private:
        bool isEmpty;
    };

    DequantizationOperations();

    DequantizationOperations(const Convert& convert, const Subtract& subtract, const Multiply& multiply);

    bool empty() const noexcept;
    bool equal(const DequantizationOperations& value) const noexcept;
    bool operator==(const DequantizationOperations& value) const noexcept {
        return equal(value);
    }
    void setPrecision(const ngraph::element::Type& type) noexcept;

    Convert convert;
    Subtract subtract;
    Multiply multiply;
};

inline std::ostream& operator<<(std::ostream& out, const DequantizationOperations::Convert& convert) {
    return out << "_" << (convert.outPrecision != element::undefined ? convert.outPrecision.get_type_name() : "");
}

inline std::ostream& operator<<(std::ostream& out, const DequantizationOperations::Subtract& subtract) {
    return out << "_" <<
        subtract.values << "_" <<
        subtract.outPrecision << "_" <<
        subtract.constantShape << "_" <<
        subtract.constantShapeIsDefined << "_" <<
        subtract.addDequantizationAttribute << "_" <<
        subtract.constantIndex << "_" <<
        subtract.constantPrecision << "_" <<
        subtract.addConvert;
}

inline std::ostream& operator<<(std::ostream& out, const DequantizationOperations::Multiply& multiply) {
    return out << "_" <<
        multiply.values << "_" <<
        multiply.outPrecision << "_" <<
        multiply.constantShape << "_" <<
        multiply.constantShapeIsDefined << "_" <<
        multiply.addDequantizationAttribute << "_" <<
        multiply.constantIndex << "_" <<
        multiply.constantPrecision;
}

inline std::ostream& operator<<(std::ostream& out, const DequantizationOperations& data) {
    return out << "_" << data.convert << "_" << data.subtract << "_" << data.multiply;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
