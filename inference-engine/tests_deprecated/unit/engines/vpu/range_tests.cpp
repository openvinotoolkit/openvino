// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <set>
#include <list>
#include <array>

#include <gtest/gtest.h>

#include <vpu/utils/range.hpp>
#include <vpu/utils/small_vector.hpp>
#include <vpu/utils/intrusive_handle_list.hpp>

using namespace testing;

//
// VPU_ContRangeTests
//

class VPU_ContRangeTests: public ::testing::Test {
protected:
    const int count = 10;
    std::list<int> list;

    void SetUp() override {
        for (int i = 0; i < count; ++i) {
            list.push_back(i);
        }
    }
};

TEST_F(VPU_ContRangeTests, PreservesIterationOrder) {
    auto contRange = vpu::containerRange(list);

    int gold = 0;
    auto innerIt = list.cbegin();
    for (const auto& val : contRange) {
        ASSERT_EQ(val, *innerIt++) << "Values given by owner and inner containers differ";
        gold++;
    }
    ASSERT_EQ(gold, count) << "Owner and inner ranges differ in length";
}

TEST_F(VPU_ContRangeTests, RespectsInnerPushBacksWhileIteration) {
    auto contRange = vpu::containerRange(list);

    int gold = 0;
    auto innerIt = list.begin();

    for (auto val : contRange) {
        if (gold < 5) {
            // duplicate first 5 elements of the head, inserting them after the tail
            list.push_back(*innerIt);
        }
        ASSERT_EQ(val, *innerIt++) << "Values given by owner and inner containers differ";
        gold++;
    }

    ASSERT_EQ(gold, count + 5) << "Initial inner container size was not preserved";
}

TEST_F(VPU_ContRangeTests, RespectsInnerRemovalsWhileIteration) {
    auto contRange = vpu::containerRange(list);

    int gold = 0;
    auto innerIt = list.begin();
    auto innerRevIt = list.end();

    for (auto val : contRange) {
        if (gold < 5) {
            // removing elements from the end
            innerRevIt = list.erase(--innerRevIt);
        }
        ASSERT_EQ(val, *innerIt++) << "Values given by owner and inner containers differ";
        gold++;
    }

    ASSERT_EQ(gold, count - 5) << "Removals were ignored";
}

TEST_F(VPU_ContRangeTests, SurvivesInnerInsertionsWhileIteration) {
    auto contRange = vpu::containerRange(list);

    int gold = 0;
    auto innerIt = list.begin();

    for (auto it = contRange.begin(); it != contRange.end(); ++it) {
        ASSERT_EQ(*it, *innerIt) << "Values given by owner and inner containers differ";

        if (gold < 10) {
            // duplicate head elements of inner, inserting them just before the current iterator
            list.insert(innerIt, *innerIt);
        }
        gold++;
        innerIt++;
    }

    ASSERT_EQ(gold, count) << "Insertions at the head influenced iteration";
}

//
// VPU_MapRangeTests
//

namespace {

const std::function<int(int)> incFunc = [](int val) { return val + 1; };

}

class VPU_MapRangeTests: public ::testing::Test {
protected:
    struct InnerStruct final : public vpu::EnableHandle {
        int val = 0;
        vpu::IntrusiveHandleListNode<InnerStruct> node;
        explicit InnerStruct(int val) : val(val), node(this) {}
    };

    const int count = 10;
    std::list<int> list;
    std::vector<int> vec;

    const static std::function<double(int)> incAndConvertFunc;

    void SetUp() override {
        for (int i = 0; i < count; ++i) {
            list.push_back(i);
            vec.push_back(i);
        }
    }
};

const std::function<double(int)> VPU_MapRangeTests::incAndConvertFunc = [](int val)
        { return static_cast<double>(val + 1); };

TEST_F(VPU_MapRangeTests, PreservesIterationOrder) {
    auto mapRange = vpu::mapRange(vpu::containerRange(vec), incFunc);

    int gold = 0;
    auto innerIt = vec.cbegin();
    for (auto cit = mapRange.begin(); cit != mapRange.end(); ++cit) {
        int mappedExpectation = incFunc(*innerIt);
        ASSERT_EQ(*cit, mappedExpectation) << "Values given by map and inner containers differ";
        gold++;
        innerIt++;
    }
    ASSERT_EQ(gold, count) << "Owner and inner ranges differ in length";
}

TEST_F(VPU_MapRangeTests, MapToAnotherType) {
    auto mapRange = vpu::mapRange(vpu::containerRange(vec), incAndConvertFunc);

    int gold = 0;
    auto innerIt = vec.cbegin();
    for (auto cit = mapRange.begin(); cit != mapRange.end(); ++cit) {
        const int base = *innerIt;
        const double mappedExpectation = incAndConvertFunc(base);
        ASSERT_EQ(*cit, mappedExpectation) << "Values given by map and inner containers differ";
        gold++;
        innerIt++;
    }
    ASSERT_EQ(gold, count) << "Owner and inner ranges differ in length";
}

TEST_F(VPU_MapRangeTests, CountSharedPointers) {
    std::vector<std::shared_ptr<InnerStruct>> nodesExternalVector;
    vpu::IntrusiveHandleList<InnerStruct> list(&InnerStruct::node);

    for (int i = 0; i < count; ++i) {
        auto innerStructSPtr = std::make_shared<InnerStruct>(i);
        ASSERT_EQ(1, innerStructSPtr.use_count()) << "single instance of shared pointer ";
        nodesExternalVector.push_back(innerStructSPtr);
        ASSERT_EQ(2, innerStructSPtr.use_count()) << "stack instance of shared pointer plus copy in vector";
        list.push_back(innerStructSPtr);
        ASSERT_EQ(2, innerStructSPtr.use_count()) << "intrusive list keeps weak pointer only";
    }

    auto mapRange = vpu::mapRange(
            vpu::containerRange(list),
            [](const vpu::Handle<InnerStruct>& innerPtr) {
                return incFunc(innerPtr->val);
            });

    for (int i = 0; i < count; ++i) {
        ASSERT_EQ(1, nodesExternalVector[i].use_count()) << "intrusive list keeps weak pointer only";
    }
}

TEST_F(VPU_MapRangeTests, IterationOverIntrusiveListSurvivesElementRemoval) {
    std::vector<std::shared_ptr<InnerStruct>> nodesExternalVector;
    vpu::IntrusiveHandleList<InnerStruct> list(&InnerStruct::node);

    for (int i = 0; i < count; ++i) {
        auto innerStructSPtr = std::make_shared<InnerStruct>(i);
        nodesExternalVector.push_back(innerStructSPtr);
        list.push_back(innerStructSPtr);
    }

    auto mapRange = vpu::mapRange(
            vpu::containerRange(list),
            [](const vpu::Handle<InnerStruct>& innerPtr) {
                return incFunc(innerPtr->val);
            });

    int i = 0;
    for (auto mprit = mapRange.begin(); mprit != mapRange.end(); ++mprit, ++i) {
        ASSERT_EQ(*mprit, incFunc(i)) << "mapped value must conform to increment function";
        list.pop_front();
    }
}

//
// VPU_FilterRangeTests
//

namespace {

const std::function<bool(int)> evenFunc = [](int val) { return val % 2 == 0; };

}

class VPU_FilterRangeTests: public ::testing::Test {
protected:
    struct InnerStruct final : public vpu::EnableHandle {
        int val = 0;
        vpu::IntrusiveHandleListNode<InnerStruct> node;
        explicit InnerStruct(int val) : val(val), node(this) {}
    };

    const int count = 10;
    std::vector<int> vec;

    void SetUp() override {
        for (int i = 0; i < count; ++i) {
            vec.push_back(i);
        }
    }
};

TEST_F(VPU_FilterRangeTests, FilteringOnlyEvenNumbers) {
    auto filterRange = vpu::filterRange(vpu::containerRange(vec), evenFunc);

    int i = 0;
    for (auto val : filterRange) {
        ASSERT_EQ(val, i);
        i += 2;
    }
    ASSERT_EQ(i, count);
}

TEST_F(VPU_FilterRangeTests, FilteringOutFirst) {
    auto filterRange = vpu::filterRange(
        vpu::containerRange(vec),
        [](int val) {
            return val != 0;
        });

    int gold = 1;
    for (auto val : filterRange) {
        ASSERT_EQ(val, gold);
        gold++;
    }
    ASSERT_EQ(gold, count);
}

TEST_F(VPU_FilterRangeTests, FilteringOutLast) {
    auto filterRange = vpu::filterRange(
        vpu::containerRange(vec),
        [&](int val) {
            return val != count - 1;
        });

    int gold = 0;
    for (auto val : filterRange) {
        ASSERT_EQ(val, gold);
        gold++;
    }
    ASSERT_EQ(gold, count - 1);
}

TEST_F(VPU_FilterRangeTests, CountSharedPointers) {
    std::list<std::shared_ptr<InnerStruct>> nodesExternalList;
    vpu::IntrusiveHandleList<InnerStruct> list(&InnerStruct::node);

    for (int i = 0; i < count; ++i) {
        auto inner = std::make_shared<InnerStruct>(i);
        ASSERT_EQ(1, inner.use_count()) << "single instance of shared pointer";
        nodesExternalList.push_back(inner);
        ASSERT_EQ(2, inner.use_count()) << "stack instance of shared pointer plus copy in vector";
        list.push_back(inner);
        ASSERT_EQ(2, inner.use_count()) << "intrusive list keeps weak pointer only";
    }

    for (auto cit = nodesExternalList.cbegin(); cit != nodesExternalList.end(); ++cit) {
        ASSERT_EQ(1, cit->use_count()) << "intrusive list keeps weak pointer only";
    }

    auto filterRange = vpu::filterRange<vpu::NonNull>(vpu::containerRange(list));

    for (auto cit = nodesExternalList.cbegin(); cit != nodesExternalList.end(); ++cit) {
        ASSERT_EQ(1, cit->use_count()) << "intrusive list keeps weak pointer only";
    }
}

TEST_F(VPU_FilterRangeTests, IterationOverIntrusiveListSurvivesElementRemoval) {
    std::list<std::shared_ptr<InnerStruct>> nodesExternalList;
    vpu::IntrusiveHandleList<InnerStruct> list(&InnerStruct::node);

    for (int i = 0; i < count; ++i) {
        auto inner = std::make_shared<InnerStruct>(i);
        nodesExternalList.push_back(inner);
        list.push_back(inner);
    }

    auto filterRange = vpu::filterRange<vpu::NonNull>(vpu::containerRange(list));

    int gold = 0;
    for (const auto& ptr : filterRange) {
        ASSERT_EQ(ptr->val, gold);
        list.pop_front();
        gold++;
    }
    ASSERT_EQ(gold, count);

    for (auto cit = nodesExternalList.cbegin(); cit != nodesExternalList.end(); ++cit) {
        ASSERT_EQ(1, cit->use_count()) << "intrusive list keeps weak pointer only";
    }
}

TEST_F(VPU_FilterRangeTests, IterationOverIntrusiveListWhileElementsBeingRemoved) {
    std::list<std::shared_ptr<InnerStruct>> nodesExternalList;
    vpu::IntrusiveHandleList<InnerStruct> list(&InnerStruct::node);

    for (int i = 0; i < count; ++i) {
        auto inner = std::make_shared<InnerStruct>(i);
        nodesExternalList.push_back(inner);
        list.push_back(inner);
    }

    auto filterRange = vpu::filterRange(
        vpu::containerRange(list),
        [](const vpu::Handle<InnerStruct>& innerPtr) {
            return evenFunc(innerPtr->val);
        });

    int gold = 0;
    for (const auto& ptr : filterRange) {
        ASSERT_EQ(ptr->val, gold);
        // remove even & odd front elems
        list.pop_front();
        list.pop_front();
        gold += 2;
    }
    ASSERT_EQ(gold, count);

    for (auto cit = nodesExternalList.cbegin(); cit != nodesExternalList.end(); ++cit) {
        ASSERT_EQ(1, cit->use_count()) << "intrusive list keeps weak pointer only";
    }
}

TEST_F(VPU_FilterRangeTests, IterationOverEmptyIntrusiveListWhereAllElementsFilteredOut) {
    std::list<std::shared_ptr<InnerStruct>> nodesExternalList;
    vpu::IntrusiveHandleList<InnerStruct> list(&InnerStruct::node);

    for (int i = 0; i < count; ++i) {
        auto inner = std::make_shared<InnerStruct>(i);
        nodesExternalList.push_back(inner);
        list.push_back(inner);
    }

    auto filterRange = vpu::filterRange(
        vpu::containerRange(list),
        [](const vpu::Handle<InnerStruct>& innerPtr) {
            return (innerPtr->val < 0);
        });

    for (const auto& ptr : filterRange) {
        ASSERT_TRUE(false) << "Must not see any item in filtered list";
    }

    for (auto cit = list.cbegin(); cit != list.cend(); ++cit) {
        if (evenFunc((*cit)->val)) {
            list.erase(cit);
        }
    }

    for (const auto& ptr : filterRange) {
        ASSERT_TRUE(false) << "Must not see any item in filtered list";
    }

    for (auto cit = nodesExternalList.cbegin(); cit != nodesExternalList.end(); ++cit) {
        ASSERT_EQ(1, cit->use_count()) << "intrusive list keeps weak pointer only";
    }
}

//
// VPU_ReverseRangeTests
//

class VPU_ReverseRangeTests: public ::testing::Test {
protected:
    struct InnerStruct final : public vpu::EnableHandle {
        int val = 0;
        vpu::IntrusiveHandleListNode<InnerStruct> node;
        explicit InnerStruct(int val) : val(val), node(this) {}
    };
};

TEST_F(VPU_ReverseRangeTests, Vector) {
    const int count = 4;

    std::vector<int> vec(count);
    for (int i = 0; i < count; ++i) {
        vec[i] = i;
    }

    int goldVal = count - 1;
    for (auto val : vpu::reverseRange(vpu::containerRange(vec))) {
        ASSERT_EQ(goldVal, val);
        --goldVal;
    }
}

TEST_F(VPU_ReverseRangeTests, IntrusiveHandleList) {
    const int count = 4;

    std::list<std::shared_ptr<InnerStruct>> nodesExternalList;
    vpu::IntrusiveHandleList<InnerStruct> list(&InnerStruct::node);

    for (int i = 0; i < count; ++i) {
        auto inner = std::make_shared<InnerStruct>(i);
        nodesExternalList.push_back(inner);
        list.push_back(inner);
    }

    int goldVal = count - 1;
    for (const auto& item : vpu::reverseRange(vpu::containerRange(list))) {
        ASSERT_EQ(goldVal, item->val);
        --goldVal;
    }
}

TEST_F(VPU_ReverseRangeTests, Filter) {
    const int count = 10;

    std::vector<int> vec(count);
    for (int i = 0; i < count; ++i) {
        vec[i] = i;
    }

    int goldVal = count - 1;
    if (!evenFunc(goldVal)) {
        --goldVal;
    }

    for (auto val : vpu::reverseRange(vpu::filterRange(vpu::containerRange(vec), evenFunc))) {
        ASSERT_EQ(goldVal, val);
        goldVal -= 2;
    }
}

TEST_F(VPU_ReverseRangeTests, Map) {
    const int count = 4;

    std::vector<int> vec(count);
    for (int i = 0; i < count; ++i) {
        vec[i] = i;
    }

    int goldVal = count;
    for (auto val : vpu::reverseRange(vpu::mapRange(vpu::containerRange(vec), incFunc))) {
        ASSERT_EQ(goldVal, val);
        --goldVal;
    }
}

//
// VPU_FlattenRangeTests
//

class VPU_FlattenRangeTests: public ::testing::Test {
protected:
    struct InnerStruct final : public vpu::EnableHandle {
        int val = 0;
        vpu::IntrusiveHandleListNode<InnerStruct> node;
        explicit InnerStruct(int val) : val(val), node(this) {}
    };

    struct OuterStruct final : public vpu::EnableHandle {
        vpu::IntrusiveHandleList<InnerStruct> inner;
        vpu::IntrusiveHandleListNode<OuterStruct> node;
        OuterStruct() : inner(&InnerStruct::node), node(this) {}
    };
};

TEST_F(VPU_FlattenRangeTests, VectorWithCopy) {
    const int outerCount = 4;

    int outerVal = 0;
    int innerVal = 0;
    int actualCount = 0;
    int goldCount = 0;

    std::vector<int> outerVec(outerCount);

    for (int i = 0; i < outerCount; ++i) {
        outerVec[i] = i + 1;
        goldCount += i + 1;
    }

    auto range =
        outerVec |
        vpu::asRange() |
        vpu::map([](int innerCount) {
            std::vector<int> innerVec(innerCount);
            for (int i = 0; i < innerCount; ++i) {
                innerVec[i] = i;
            }
            return std::move(innerVec) | vpu::asRange();
        }) |
        vpu::flatten();

    outerVal = 1;
    innerVal = 0;
    actualCount = 0;
    for (const auto& item : range) {
        EXPECT_EQ(item, innerVal) << "forward";
        ++actualCount;

        ++innerVal;
        if (innerVal == outerVal) {
            innerVal = 0;
            ++outerVal;
        }
    }
    EXPECT_EQ(actualCount, goldCount) << "forward";
    EXPECT_EQ(range.size(), goldCount);
    EXPECT_EQ(range.front(), 0);
    EXPECT_EQ(range.back(), outerCount - 1);

    auto revRange = range | vpu::reverse();

    outerVal = outerCount;
    innerVal = outerVal - 1;
    actualCount = 0;
    for (const auto& item : revRange) {
        EXPECT_EQ(item, innerVal) << "reverse";
        ++actualCount;

        --innerVal;
        if (innerVal < 0) {
            --outerVal;
            innerVal = outerVal - 1;
        }
    }
    EXPECT_EQ(actualCount, goldCount) << "reverse";
    EXPECT_EQ(revRange.size(), goldCount);
    EXPECT_EQ(revRange.front(), outerCount - 1);
    EXPECT_EQ(revRange.back(), 0);
}

TEST_F(VPU_FlattenRangeTests, IntrusiveHandleList) {
    const int outerCount = 4;
    const int innerCount = 8;
    const int innerTotalCount = innerCount * outerCount;

    int goldVal = 0;

    std::vector<std::shared_ptr<InnerStruct>> innerExternalList;

    for (int i = 0; i < innerTotalCount; ++i) {
        auto inner = std::make_shared<InnerStruct>(i);
        innerExternalList.push_back(inner);
    }

    std::list<std::shared_ptr<OuterStruct>> outerExternalList;
    vpu::IntrusiveHandleList<OuterStruct> list(&OuterStruct::node);

    for (int i = 0; i < outerCount; ++i) {
        auto outer = std::make_shared<OuterStruct>();
        for (int j = 0; j < innerCount; ++j) {
            outer->inner.push_back(innerExternalList[i * innerCount + j]);
        }
        outerExternalList.push_back(outer);
        list.push_back(outer);
    }

    auto range =
            vpu::containerRange(list) |
            vpu::map([](const vpu::Handle<OuterStruct>& o) { return vpu::containerRange(o->inner); }) |
            vpu::flatten();

    goldVal = 0;
    for (const auto& item : range) {
        ASSERT_EQ(goldVal, item->val);
        ++goldVal;
    }
    ASSERT_EQ(goldVal, innerTotalCount);

    goldVal = innerTotalCount - 1;
    for (const auto& item : range | vpu::reverse()) {
        ASSERT_EQ(goldVal, item->val);
        --goldVal;
    }
    ASSERT_EQ(goldVal, -1);

    goldVal = 0;
    for (const auto& item : range | vpu::filter([](const vpu::Handle<InnerStruct>& i) { return i->val % 2 == 0; })) {
        ASSERT_EQ(goldVal, item->val);
        goldVal += 2;
    }
    ASSERT_EQ(goldVal, innerTotalCount);
}

TEST_F(VPU_FlattenRangeTests, WithEmptySubRanges) {
    const int count = 8;
    int goldVal = 0;

    std::vector<std::shared_ptr<InnerStruct>> vec(count);

    for (int i = 0; i < count; ++i) {
        vec[i] = std::make_shared<InnerStruct>(i);
    }

    auto range =
            vec |
            vpu::asRange() |
            vpu::map([](const std::shared_ptr<InnerStruct>& val) {
                return evenFunc(val->val) ? vpu::Handle<InnerStruct>(val) : vpu::Handle<InnerStruct>();
            }) |
            vpu::map([](const vpu::Handle<InnerStruct>& val) {
                return val | vpu::asSingleElementRange() | vpu::filter<vpu::NonNull>();
            }) |
            vpu::flatten() |
            vpu::map([](const vpu::Handle<InnerStruct>& val) {
                return val->val;
            });

    goldVal = 0;
    for (auto item : range) {
        ASSERT_EQ(goldVal, item);
        goldVal += 2;
    }
    ASSERT_EQ(goldVal, count);
}
