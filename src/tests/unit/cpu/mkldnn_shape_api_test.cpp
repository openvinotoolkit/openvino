// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "mkldnn_memory.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

TEST(ShapeAPITest, isStatic) {
    auto isStatic = [] (const Shape& shape, const bool expected) {
        ASSERT_EQ(shape.isStatic(), expected);
    };

    std::pair<Shape, bool> payload[] {
            {Shape(SizeVector{1, 1, 10, 10}), true},
            {Shape(SizeVector{4, 2, 10, 7 }), true},
            {Shape(SizeVector{4, 2, 10, 7 }), true},
            {Shape(SizeVector{1, 1, 10, 10}), true},
            {Shape(ngraph::PartialShape{{16}, {1, 16}, {-1, -1}, {-1, -1}}), false},
            {Shape(ngraph::PartialShape{{8, 16}, {16}, {8, 24}, {10, 20}}), false},
            {Shape(ngraph::PartialShape{{8, 16}, {8, 16}, {8, 24}, {10, 20}}), false},
            {Shape(ngraph::PartialShape{{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}), false}
    };

    for (auto& p : payload) {
        isStatic(p.first, p.second);
    }
}

TEST(ShapeAPITest, isDynamic) {
    auto isDynamic = [] (const Shape& shape, const bool expected) {
        ASSERT_EQ(shape.isDynamic(), expected);
    };

    std::pair<Shape, bool> payload[] {
            {Shape(SizeVector{1, 1, 10, 10}), false},
            {Shape(SizeVector{4, 2, 10, 7 }), false},
            {Shape(SizeVector{4, 2, 10, 7 }), false},
            {Shape(SizeVector{1, 1, 10, 10}), false},
            {Shape(ngraph::PartialShape{{16}, {1, 16}, {-1, -1}, {-1, -1}}), true},
            {Shape(ngraph::PartialShape{{8, 16}, {16}, {8, 24}, {10, 20}}), true},
            {Shape(ngraph::PartialShape{{8, 16}, {8, 16}, {8, 24}, {10, 20}}), true},
            {Shape(ngraph::PartialShape{{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}), true}
    };

    for (auto& p : payload) {
        isDynamic(p.first, p.second);
    }
}

TEST(ShapeAPITest, isCompatible) {
    auto isCompatible = [] (const Shape& shape, const VectorDims& expected, const bool result) {
        ASSERT_EQ(result, shape.isCompatible(expected));
    };

    std::pair<std::pair<Shape, VectorDims>, bool> payload[] {
            {{Shape(SizeVector{1, 1, 10, 10}), VectorDims{1, 1, 10, 10}}, true},
            {{Shape(SizeVector{4, 2, 10, 7 }), VectorDims{1, 1, 10, 10}}, false},
            {{Shape(ngraph::PartialShape{{16}, {1, 16}, {-1, -1}, {-1, -1}}), VectorDims{16, 8, 10, 10}}, true},
            {{Shape(ngraph::PartialShape{{8, 16}, {16}, {8, 24}, {10, 20}}), VectorDims{10, 16, 10, 10}}, true},
            {{Shape(ngraph::PartialShape{{8, 16}, {8, 16}, {8, 24}, {10, 20}}), VectorDims{24, 4, 8, 8}}, false},
            {{Shape(ngraph::PartialShape{{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}), VectorDims{1, 1, 10, 10}}, true},
    };

    for (auto& p : payload) {
        isCompatible(p.first.first, p.first.second, p.second);
    }
}

TEST(ShapeAPITest, getDims) {
    auto getDims = [] (const Shape& shape, const VectorDims& expected) {
        ASSERT_EQ(expected, shape.getDims());
    };

    std::pair<Shape, VectorDims> payload[] {
            {Shape(SizeVector{1, 1, 10, 10}), VectorDims{1, 1, 10, 10}},
            {Shape(SizeVector{4, 2, 10, 7 }), VectorDims{4, 2, 10, 7 }},
            {Shape(SizeVector{4, 2, 10, 7 }), VectorDims{4, 2, 10, 7 }},
            {Shape(SizeVector{1, 1, 10, 10}), VectorDims{1, 1, 10, 10}},
            {Shape(ngraph::PartialShape{{16}, {1, 16}, {-1, -1}, {-1, -1}}),
             VectorDims{16, Shape::UNDEFINED_DIM, Shape::UNDEFINED_DIM, Shape::UNDEFINED_DIM}},
            {Shape(ngraph::PartialShape{{8, 16}, {16}, {8, 24}, {10, 20}}),
             VectorDims{Shape::UNDEFINED_DIM, 16, Shape::UNDEFINED_DIM, Shape::UNDEFINED_DIM}},
            {Shape(ngraph::PartialShape{{8, 16}, {8, 16}, {8, 24}, {10, 20}}),
             VectorDims{Shape::UNDEFINED_DIM, Shape::UNDEFINED_DIM, Shape::UNDEFINED_DIM, Shape::UNDEFINED_DIM}},
            {Shape(ngraph::PartialShape{{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}),
             VectorDims{Shape::UNDEFINED_DIM, Shape::UNDEFINED_DIM, Shape::UNDEFINED_DIM, Shape::UNDEFINED_DIM}}
    };

    for (auto& p : payload) {
        getDims(p.first, p.second);
    }
}

TEST(ShapeAPITest, getElementsCount) {
    auto getElementsCount = [] (const Shape& shape, const size_t expected) {
        return shape.getElementsCount() == expected;
    };

    std::pair<Shape, size_t> payload[] {
            {Shape(SizeVector{1, 1, 10, 10}), 100},
            {Shape(SizeVector{4, 2, 10, 7 }), 560},
            {Shape(SizeVector{4, 2, 10, 7 }), 560},
            {Shape(SizeVector{1, 1, 20, 10}), 200},
    };

    for (auto& p : payload) {
        ASSERT_TRUE(getElementsCount(p.first, p.second));
        ASSERT_ANY_THROW(getElementsCount(Shape(ngraph::PartialShape{{16}, {1, 16}, {-1, -1}, {-1, -1}}), 0));
    }
}

TEST(ShapeAPITest, getMaxDims) {
    auto getMaxDims = [] (const Shape& shape, const VectorDims& expected) {
        ASSERT_EQ(expected, shape.getMaxDims());
    };

    std::pair<Shape, VectorDims> payload[] {
            {Shape(SizeVector{1, 1, 10, 10}), VectorDims{1, 1, 10, 10}},
            {Shape(SizeVector{4, 2, 10, 7 }), VectorDims{4, 2, 10, 7 }},
            {Shape(SizeVector{4, 2, 10, 7 }), VectorDims{4, 2, 10, 7 }},
            {Shape(SizeVector{1, 1, 10, 10}), VectorDims{1, 1, 10, 10}},
            {Shape(ngraph::PartialShape{{16}, {1, 16}, {-1, -1}, {-1, -1}}),
                                              VectorDims{16, 16, Shape::UNDEFINED_DIM, Shape::UNDEFINED_DIM}},
            {Shape(ngraph::PartialShape{{8, 16}, {16}, {8, 24}, {10, 20}}),
                                              VectorDims{16, 16, 24, 20}},
            {Shape(ngraph::PartialShape{{8, 16}, {8, 16}, {-1, -1}, {10, 20}}),
                                              VectorDims{16, 16, Shape::UNDEFINED_DIM, 20}},
            {Shape(ngraph::PartialShape{{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}),
                                              VectorDims{Shape::UNDEFINED_DIM, Shape::UNDEFINED_DIM, Shape::UNDEFINED_DIM, Shape::UNDEFINED_DIM}}
    };

    for (auto& p : payload) {
        getMaxDims(p.first, p.second);
    }
}

TEST(ShapeAPITest, getMinDims) {
    auto getMinDims = [] (const Shape& shape, const VectorDims& expected) {
        ASSERT_EQ(expected, shape.getMinDims());
    };

    std::pair<Shape, VectorDims> payload[] {
            {Shape(SizeVector{1, 1, 10, 10}), VectorDims{1, 1, 10, 10}},
            {Shape(SizeVector{4, 2, 10, 7 }), VectorDims{4, 2, 10, 7 }},
            {Shape(SizeVector{4, 2, 10, 7 }), VectorDims{4, 2, 10, 7 }},
            {Shape(SizeVector{1, 1, 10, 10}), VectorDims{1, 1, 10, 10}},
            {Shape(ngraph::PartialShape{{16}, {1, 16}, {-1, -1}, {-1, -1}}),
                                              VectorDims{16, 1, 0, 0}},
            {Shape(ngraph::PartialShape{{8, 16}, {16}, {8, 24}, {10, 20}}),
                                              VectorDims{8, 16, 8, 10}},
            {Shape(ngraph::PartialShape{{8, 16}, {8, 16}, {-1, -1}, {10, 20}}),
                                              VectorDims{8, 8, 0, 10}},
            {Shape(ngraph::PartialShape{{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}),
                                              VectorDims{0, 0, 0, 0}}
    };

    for (auto& p : payload) {
        getMinDims(p.first, p.second);
    }
}

TEST(ShapeAPITest, getStaticDims) {
    auto getStaticDims = [] (const Shape& shape, const VectorDims& expected) {
        return shape.getStaticDims() == expected;
    };

    std::pair<Shape, VectorDims> payload[] {
            {Shape(SizeVector{1, 1, 10, 10}), VectorDims{1, 1, 10, 10}},
            {Shape(SizeVector{4, 2, 10, 7 }), VectorDims{4, 2, 10, 7 }},
            {Shape(SizeVector{4, 2, 10, 7 }), VectorDims{4, 2, 10, 7 }},
            {Shape(SizeVector{1, 1, 20, 10}), VectorDims{1, 1, 20, 10}},
    };

    for (auto& p : payload) {
        ASSERT_TRUE(getStaticDims(p.first, p.second));
        ASSERT_ANY_THROW(getStaticDims(Shape(ngraph::PartialShape{{16}, {1, 16}, {-1, -1}, {-1, -1}}), VectorDims{16, 1, 0, 0}));
    }
}

TEST(ShapeAPITest, getRank) {
    auto getRank = [] (const Shape& shape, const size_t expected) {
        ASSERT_EQ(expected, shape.getRank());
    };

    std::pair<Shape, size_t> payload[] {
            {Shape(SizeVector{1, 1, 10, 10, 10}), 5},
            {Shape(SizeVector{4, 2, 10, 7 }), 4},
            {Shape(SizeVector{4, 2, 10 }), 3},
            {Shape(SizeVector{1, 1}), 2},
            {Shape(SizeVector{8}), 1},
            {Shape(ngraph::PartialShape{{16}, {1, 16}, {-1, -1}, {-1, -1}, {-1, -1}}), 5},
            {Shape(ngraph::PartialShape{{8, 16}, {16}, {8, 24}, {10, 20}}), 4},
            {Shape(ngraph::PartialShape{{8, 16}, {8, 16}, {8, 24}}), 3},
            {Shape(ngraph::PartialShape{{8, 16}, {8, 16}}), 2},
            {Shape(ngraph::PartialShape{{-1, -1}}), 1},
    };

    for (auto& p : payload) {
        getRank(p.first, p.second);
    }
}

TEST(ShapeAPITest, hasDefinedUpperBounds) {
    auto hasDefinedUpperBounds = [] (const Shape& shape, const bool expected) {
        ASSERT_EQ(shape.hasDefinedUpperBounds(), expected);
    };

    std::pair<Shape, bool> payload[] {
            {Shape(SizeVector{1, 1, 10, 10}), true},
            {Shape(SizeVector{4, 2, 10, 7 }), true},
            {Shape(SizeVector{4, 2, 10, 7 }), true},
            {Shape(SizeVector{1, 1, 10, 10}), true},
            {Shape(ngraph::PartialShape{{16}, {1, 16}, {-1, -1}, {-1, -1}}), false},
            {Shape(ngraph::PartialShape{{8, 16}, {16}, {8, 24}, {10, 20}}), true},
            {Shape(ngraph::PartialShape{{8, 16}, {8, 16}, {8, 24}, {10, 20}}), true},
            {Shape(ngraph::PartialShape{{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}), false}
    };

    for (auto& p : payload) {
        hasDefinedUpperBounds(p.first, p.second);
    }
}

TEST(ShapeAPITest, toPartialShape) {
    auto toPartialShape = [] (const Shape& shape, const ngraph::PartialShape& expected) {
        ASSERT_EQ(shape.toPartialShape(), expected);
    };

    std::pair<Shape, ngraph::PartialShape> payload[] {
            {Shape(SizeVector{1, 1, 10, 10}), ngraph::PartialShape{1, 1, 10, 10}},
            {Shape(SizeVector{4, 2, 10, 7 }), ngraph::PartialShape{4, 2, 10, 7 }},
            {Shape(SizeVector{4, 2, 10, 7 }), ngraph::PartialShape{4, 2, 10, 7 }},
            {Shape(SizeVector{1, 1, 20, 10}), ngraph::PartialShape{1, 1, 20, 10}},
            {Shape(ngraph::PartialShape{{16}, {1, 16}, {-1, -1}, {-1, -1}}),
                    ngraph::PartialShape{{16}, {1, 16}, {-1, -1}, {-1, -1}}},
            {Shape(ngraph::PartialShape{{8, 16}, {16}, {8, 24}, {10, 20}}),
                    ngraph::PartialShape{{8, 16}, {16}, {8, 24}, {10, 20}}},
            {Shape(ngraph::PartialShape{{8, 16}, {8, 16}, {-1, -1}, {10, 20}}),
                    ngraph::PartialShape{{8, 16}, {8, 16}, {-1, -1}, {10, 20}}},
            {Shape(ngraph::PartialShape{{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}),
                    ngraph::PartialShape{{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}}
    };

    for (auto& p : payload) {
        toPartialShape(p.first, p.second);
    }
}
