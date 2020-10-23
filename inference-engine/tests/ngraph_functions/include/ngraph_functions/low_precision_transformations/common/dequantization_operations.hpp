// Copyright (C) 2020 Intel Corporation
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
        Convert(const ngraph::element::Type outPrecision);
        bool empty() const noexcept;

        ngraph::element::Type outPrecision;
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
            const bool addDequantizationAttribute = true);
        bool empty() const noexcept;
        Subtract& setConstantPrecision(const ngraph::element::Type& precision);

        std::vector<float> values;
        ngraph::element::Type outPrecision;
        ngraph::Shape constantShape;
        bool constantShapeIsDefined;
        bool addDequantizationAttribute;
        ngraph::element::Type constantPrecision = ngraph::element::undefined;
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
            const size_t constantIndex = 1ul);
        bool empty() const noexcept;
        Multiply& setConstantPrecision(const ngraph::element::Type& precision);

        std::vector<float> values;
        ngraph::element::Type outPrecision;
        ngraph::Shape constantShape;
        bool constantShapeIsDefined;
        bool addDequantizationAttribute;
        size_t constantIndex = 1ul;
        ngraph::element::Type constantPrecision = ngraph::element::undefined;
    private:
        bool isEmpty;
    };

    DequantizationOperations();

    DequantizationOperations(const Convert& convert, const Subtract& subtract, const Multiply& multiply);

    bool empty() const;

    Convert convert;
    Subtract subtract;
    Multiply multiply;
};

inline std::ostream& operator<<(std::ostream& out, const DequantizationOperations& data) {
    return out << "_" <<
        (data.convert.outPrecision != element::undefined ? data.convert.outPrecision.get_type_name() : "") << "_" <<
        data.subtract.values << "_" <<
        data.subtract.constantShape << "_" <<
        data.subtract.outPrecision << "_" <<
        data.multiply.values << "_" <<
        data.multiply.constantShape << "_" <<
        data.multiply.outPrecision;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
