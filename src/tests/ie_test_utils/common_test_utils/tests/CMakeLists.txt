set(TARGET_NAME commonUtilsTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
        commonTestUtils
        LINK_LIBRARIES
        commonTestUtils
        ADD_CPPLINT
)