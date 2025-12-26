import textwrap

from jinja2 import Template


def gen_cmakelists(language="CXX", verify=True, project="project", libname="mylibrary",
                   libsources=None, appname="myapp", appsources=None, cmake_version="3.15",
                   install=False, find_package=None, libtype="", deps=None, public_header=None,
                   custom_content=None):
    """
    language: C, C++, C/C++
    project: the project name
    """
    cmake = textwrap.dedent("""\
        {% if verify %}
        set(CMAKE_CXX_COMPILER_WORKS 1)
        set(CMAKE_CXX_ABI_COMPILED 1)
        set(CMAKE_C_COMPILER_WORKS 1)
        set(CMAKE_C_ABI_COMPILED 1)
        {% endif %}

        cmake_minimum_required(VERSION {{cmake_version}})
        project({{project}} {{language}})

        {% if find_package is mapping %}
        {% for s, c in find_package.items() %}
        find_package({{s}} COMPONENTS {{c}} )
        {% endfor %}
        {% else %}
        {% for s in find_package %}
        find_package({{s}})
        {% endfor %}
        {% endif %}

        {% if libsources %}
        add_library({{libname}} {{libtype}} {% for s in libsources %} {{s}} {% endfor %})
        target_include_directories({{libname}} PUBLIC "include")
        {% endif %}

        {% if libsources and find_package %}
        {% if find_package is mapping %}
        target_link_libraries({{libname}} {% for s, c in find_package.items() %} {{s}}::{{c}} {% endfor %})
        {% else %}
        target_link_libraries({{libname}} {% for s in find_package %} {{s}}::{{s}} {% endfor %})
        {% endif %}
        {% endif %}

        {% if libsources and deps %}
        target_link_libraries({{libname}} {% for s in deps %} {{s}} {% endfor %})
        {% endif %}

        {% if appsources %}
        add_executable({{appname}} {% for s in appsources %} {{s}} {% endfor %})
        target_include_directories({{appname}} PUBLIC "include")
        {% endif %}

        {% if appsources and libsources %}
        target_link_libraries({{appname}} {{libname}})
        {% endif %}

        {% if appsources and not libsources and find_package %}
        {% if find_package is mapping %}
         target_link_libraries({{appname}} {% for s, c in find_package.items() %} {{s}}::{{c}} {% endfor %})
        {% else %}
        target_link_libraries({{appname}} {% for s in find_package %} {{s}}::{{s}} {% endfor %})
        {% endif %}
        {% endif %}

        {% if appsources and deps %}
        target_link_libraries({{appname}} {% for s in deps %} {{s}} {% endfor %})
        {% endif %}

        {% if libsources and public_header %}
        set_target_properties({{libname}} PROPERTIES PUBLIC_HEADER "{{public_header}}")
        {% endif %}

        {% if install %}
        {% if appsources %}
        install(TARGETS {{appname}})
        {% endif %}
        {% if libsources %}
        install(TARGETS {{libname}})
        {% endif %}
        {% endif %}

        {% if custom_content %}
        {{custom_content}}
        {% endif %}

        """)

    t = Template(cmake, trim_blocks=True, lstrip_blocks=True)
    return t.render({"verify": verify,
                     "language": language,
                     "project": project,
                     "libname": libname,
                     "libsources": libsources,
                     "appname": appname,
                     "appsources": appsources,
                     "cmake_version": cmake_version,
                     "install": install,
                     "find_package": find_package or [],
                     "libtype": libtype,
                     "public_header": public_header,
                     "deps": deps,
                     "custom_content": custom_content
                     })
