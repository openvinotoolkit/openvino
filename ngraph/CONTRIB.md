Contributor Guidelines
======================

The latest version of this file can be found at: 

https://www.ngraph.ai/documentation/contributing/guide

License
-------

All contributed code must be compatible with the [Apache
2](https://www.apache.org/licenses/LICENSE-2.0) license, preferably by
being contributed under the Apache 2 license. Code contributed with
another license will need the license reviewed by Intel before it can be
accepted.

Code formatting
---------------

All C/C++ source code in the repository, including the test code, must
adhere to the source-code formatting and style guidelines described
here. The coding style described here applies to the nGraph repository.
Related repositories may make adjustements to better match the coding
styles of libraries they are using.

### Adding ops to nGraph Core

Our design philosophy is that the graph is not a script for running
optimized kernels; rather, the graph is a specification for a
computation composed of basic building blocks which we call `ops`.
Compilation should match groups of `ops` to appropriate optimal and
semantically-equivalent groups of kernels for the backend(s) in use.
Thus, we expect that adding of new Core ops should be infrequent and
that most functionality instead gets added with new functions that build
sub-graphs from existing core ops.

### Coding style

We have a coding standard to help us to get development done. If part of
the standard is impeding progress, we either adjust that part or remove
it. To this end, we employ coding standards that facilitate
understanding of *what nGraph components are doing*. Programs are
easiest to understand when they can be understood locally; if most local
changes have local impact, you do not need to dig through multiple files
to understand what something does and if it is safe to modify.

#### Names

Names should *briefly* describe the thing being named and follow these
casing standards:

-   Define C++ class or type names with `CamelCase`.
-   Assign template parameters with `UPPER_SNAKE_CASE`.
-   Case variable and function names with `lower_snake_case`.

Method names for basic accessors are prefixed by `get_`, `is_`, or
`set_` and should have simple $\mathcal{O}(1)$ implementations:

-   A `get_` method should be externally idempotent. It may perform some
    simple initialization and cache the result for later use. Trivial
    `get_` methods can be defined in a header file. If a method is
    non-trivial, that is often a sign that it is not a basic accessor.
-   An `is_` may be used instead of `get_` for boolean accessors.
-   A `set_` method should change the value returned by the
    corresponding `get_` method.
    -   Use `set_is_` if using `is_` to get a value.
    -   Trivial `set_` methods may be defined in a header file.
-   Names of variables should indicate the use of the variable.
    -   Member variables should be prefixed with `m_`.
    -   Static member variables should be rare and be prefixed with
        `s_`.
-   Do not use `using` to define a type alias at top-level in header
    file. If the abstraction is useful, give it a class.
    -   C++ does not enforce the abstraction. For example if `X` and `Y`
        are aliases for the same type, you can pass an `X` to something
        expecting a `Y`.
    -   If one of the aliases were later changed, or turned into a real
        type, many callers could require changes.

#### Namespaces

-   `ngraph` is for the public API, although this is not
    currently enforced.
    -   Use a nested namespace for implementation classes.
    -   Use an unnamed namespace or `static` for file-local names. This
        helps prevent unintended name collisions during linking and when
        using shared and dynamically-loaded libraries.
    -   Never use `using` at top-level in a header file.

        -   Doing so leaks the alias into users of the header, including
            headers that follow.

        - It is okay to use `using` with local scope, such as inside a class 
          definiton.

    -   Be careful of C++'s implicit namespace inclusions. For example,
        if a parameter's type is from another namespace, that namespace
        can be visible in the body.
    -   Only use `using std` and/or `using ngraph` in `.cpp` files.
        `using` a nested namespace has can result in
        unexpected behavior.

#### File Names

-   Do not use the same file name in multiple directories. At least one
    IDE/debugger ignores the directory name when setting breakpoints.
-   Use `.hpp` for headers and `.cpp` for implementation.
-   Reflect the namespace nesting in the directory hierarchy.
-   Unit test files are in the `tests` directory.
    -   Transformer-dependent tests are tests running on the default
        transformer or specifying a transformer. For these, use the form

        ``` 
        TEST(file_name, test_name)
        ```

    -   Transformer-independent tests:
        -   File name is `file_name.in.cpp`
        -   Add `#include "test_control.hpp"` to the file's includes
        -   Add the line
            `static std::string s_manifest = "${MANIFEST}";` to the top
            of the file.
        -   Use

            ``` 
            NGRAPH_TEST(${BACKEND_NAME}, test_name)
            ```

            for each test. Files are generated for each transformer and
            the `${BACKEND_NAME}` is replaced with the transformer name.

            Individual unit tests may be disabled by adding the name of
            the test to the `unit_test.manifest` file found in the
            transformer's source file directory.

#### Formatting

Things that look different should look different because they are
different. We use **clang format** to enforce certain formatting.
Although not always ideal, it is automatically enforced and reduces
merge conflicts.

-   The .clang-format file located in the root of the project specifies
    our format.
    -   The script maint/apply-code-format.sh enforces that formatting
        at the C/C++ syntactic level.
    -   The script at maint/check-code-format.sh verifies that the
        formatting rules are met by all C/C++ code (again, at the
        syntax level). The script has an exit code of `0` when code
        meets the standard and non-zero otherwise. This script does
        *not* modify the source code.
-   Formatting with `#include` files:
    -   Put headers in groups separated by a blank line. Logically order
        the groups downward from system-level to 3rd-party to `ngraph`.
    -   Formatting will keep the files in each group in
        alphabetic order.
    -   Use this syntax for files that **do not change during nGraph
        development**; they will not be checked for changes
        during builds. Normally this will be everything but the ngraph
        files:

        ``` 
        #include <file>
        ```

    -   Use this syntax for files that **are changing during nGraph
        development**; they will be checked for changes during builds.
        Normally this will be ngraph headers:

        ``` 
        #include "file"
        ```

    -   Use this syntax for system C headers with C++ wrappers:

        ``` 
        #include <c...>
        ```

-   To guard against multiple inclusion, use:

    ``` 
    #pragma once
    ```

    -   The syntax is a compiler extension that has been adopted by all
        supported compilers.
-   The initialization

    ``` 
    Foo x{4, 5};
    ```

    is preferred over

    ``` 
    Foo x(4, 5);
    ```

-   Indentation should be accompanied by braces; this includes
    single-line bodies for conditionals and loops.
-   Exception checking:
    -   Throw an exception to report a problem.
    -   Nothing that calls `abort`, `exit` or `terminate` should
        be used. Remember that ngraph is a guest of the framework.
    -   Do not use exclamation points in messages!
    -   Be as specific as practical. Keep in mind that the person who
        sees the error is likely to be on the other side of the
        framework and the message might be the only information they see
        about the problem.
-   If you use `auto`, know what you are doing. `auto` uses the same
    type-stripping rules as template parameters. If something returns a
    reference, `auto` will strip the reference unless you use `auto&`:
    -   Don't do things like

        ``` 
        auto s = Shape{2,3};
        ```

        Instead, use

        ``` 
        Shape s{2, 3};
        ```

    -   Indicate the type in the variable name.

-   One variable declaration/definition per line
    -   Don't use the C-style

        ``` 
        int x, y, *z;
        ```

        Instead, use:

        ``` 
        int x;
        int y;
        int* z;
        ```
