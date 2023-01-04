--------------------------------------------------------------------------------
--
--  This file is part of the Doxyrest toolkit.
--
--  Doxyrest is distributed under the MIT license.
--  For details see accompanying license.txt file,
--  the public copy of which is also available at:
--  http://tibbo.com/downloads/archive/doxyrest/license.txt
--
--------------------------------------------------------------------------------

--!
--! \defgroup frame-config
--! \grouporder 2
--! \title Frame Settings
--!
--! This section describes frame settings controlling input and output paths,
--! titles, force-includes, declaration coding style, etc.
--!
--! A default ``doxyrest-config.lua`` file for standard frames can be found at
--! ``$DOXYREST_FRAME_DIR/doxyrest-config.lua``. Copy it to your project
--! directory and then adjust all the necessary parameters.
--!
--! @{
--!

--!
--! Table containing a list of frame directories. All frame files will be
--! searched in directories -- and in the sequence -- specified here.
--!
--
FRAME_DIR_LIST = { "@DOXYREST_OUT@/frame/cfamily", "@DOXYREST_OUT@/frame/common"}

--!
--! The output master (index) reStructuredText file. Usually, the index frame
--! also generates auxillary files -- they will be placed next to the master
--! file. The command line option ``-f`` *overrides* this value.
--! If neither ``FRAME_FILE`` nor ``-f`` is specified, ``index.rst.in`` will be
--! used as the default frame file.
--!

FRAME_FILE = '@DOXYREST_OUT@/frame/cfamily/index.rst.in'

--!
--! The input master (index) XML file. Specifying it here allows calling
--! ``doxyrest`` without parameters; otherwise, the master XML *must* be passed
--! via the command line. If both ``INPUT_FILE`` and command line parameter are
--! specified, the command line takes precedence.
--!

INPUT_FILE = '@XML_OUTPUT@/index.xml'

--!
--! The output master (index) reStructuredText file. Usually, the index frame
--! also generates auxillary files -- they will be placed next to the master
--! file. The command line option ``-o`` *overrides* this value. If neither
--! ``OUTPUT_FILE`` nor ``-o`` is specified, ``index.rst`` will be used as
--! the default output master file.
--!

OUTPUT_FILE = "@RST_OUTPUT@/index.rst"

--!
--! File with project-specific reStructuredText definitions. When non``nil``,
--! this file will be included at the top of every generated ``.rst`` file.
--!

FORCE_INCLUDE_FILE = nil

--!
--! If you want to add extra reStructuredText documentation pages, do so
--! by adding them to this list.
--!

EXTRA_PAGE_LIST = {}

--!
--! The title of the main (index) page. This only is used when ``INTRO_FILE``
--! is not set (otherwise, the title of intro file will be used).
--!

INDEX_TITLE = "OpenVINO Runtime C++ API Reference"

--!
--! File with project introduction (reStructuredText). When non-nil, this file
--! will be included into ``index.rst`` file and NOT added to the list of other
--! documentation pages.
--!

INTRO_FILE = nil

--!
--! Specify whether to sort groups lexicographically (by ``title``) or
--! logically (by ``id``). To maintain the original order (in which a group has
--! been seen in the XML database), use ``originalIdx``.
--!
--! Omitting ``SORT_GROUPS_BY`` (or setting it to ``nil``) results in groups
--! being sorted by ``title``.
--!

SORT_GROUPS_BY = nil


--[[!
	By default, the page for the global namespace page will be called
	"Global Namespace" and will contain no description except that for the
	global compounds and members.

	It's possible to override this behaviour by defining an auxillary compound
	(page or group) with a special ``id``; this page/group may contain a
	user-defined title,  a brief description and a detailed description. Use
	``GLOBAL_AUX_COMPOUND_ID`` to define this special id.

	.. note::

		To make sure you use the correct **Doxygen** XML ID of the group/page,
		find the definition of the group in one of ``.xml`` files and copy
		the value of ``id`` attribute.

		For example, if the group was declared as ``\defgroup global`` then
		the its ``id`` will probably be either ``group_<your-group-name>`` or
		``group__<your-group-name>``.
]]

GLOBAL_AUX_COMPOUND_ID = "group_global"

--[[!
	Doxyrest offers a workaround for the lack of footnotes in Doxygen by
	getting documentation blocks for specially named pseudo-members and
	converting those into footnotes.

	``FOOTNOTE_MEMBER_PREFIX`` specifies the name prefix for such
	pseudo-members. If it is set to ``nil`` or an empty string, Doxyrest
	will not attempt to convert any members to footnotes.

	\sa :ref:`footnotes`
]]

FOOTNOTE_MEMBER_PREFIX = nil

--!
--! Specify the main language of your project; this string will be used for
--! the reStructuredText ``.. code-block::`` sections and for conditional
--! formatting of module item declarations.
--!

LANGUAGE = cpp

--!
--! Convert ``\verbatim`` sections in doxy-comments to ``.. code-block::``
--! sections in the output reStructuredText. The string value of
--! ``VERBATIM_TO_CODE_BLOCK`` will be used as the language of
--! ``.. code-block::`` section. By default, it's ``"none"`` which results in
--! no syntax highlighting. To disable conversion at all, use ``nil``.
--!

VERBATIM_TO_CODE_BLOCK = "cpp"

--!
--! If the original doxy comments contain asterisks, they have to be escaped in
--! reStructuredText (asterisks are used to mark **bold** or *italic* blocks).
--!

ESCAPE_ASTERISKS = false

--!
--! If the original doxy comments contain pipe characters ``|``, they have to be
--! escaped in reStructuredText (pipes are used for substitution references).
--!

ESCAPE_PIPES = false

--!
--! If the original doxy comments contain trailingasterisks, they have to be
--! escaped in reStructuredText (trailing underscores are used for internal
--! links).
--!

ESCAPE_TRAILING_UNDERSCORES = false

--!
--! Exclude items declared in specific locations. Use a regular expression to
--! define a mask of directories/source files to completely exclude from the
--! final documentation. For example, ``.*/impl/.*lua$`` will exclude all
--! ``.lua`` files located in ``impl/`` directory.
--!

EXCLUDE_LOCATION_PATTERN = nil

--!
--! Exclude variables and functions without any documentation (no doxy-comments).
--!

EXCLUDE_UNDOCUMENTED_ITEMS = true

--!
--! \subgroup
--!
--! By default, Doxyrest tries to translate Doxygen ``\section``,
--! ``\subsection``, and ``\subsubsection`` commands (as well as the
--! ``<heading level="n">`` blocks generated by Doxygen from Markdown titles
--! inside comments) into reStructuredText titles.
--!
--! This sometimes leads to problems because Doxygen headings may appear outside
--! of the global scope (e.g. inside lists) while reStructuredText titles are
--! only allowed at the global scope. Another issue is Doxygen headings may
--! yield inconsistent title structure (e.g. a title level 1 followed by level
--! 3).
--!
--! If you run into these issues, use ``SECTION_TO_RUBRIC`` or
--! ``HEADING_TO_RUBRIC``  to always convert Doxygen sections or ``<heading>``
--! blocks or into reStructuredText ``.. rubric::`` directives. This yields
--! uni-level headings, but solves both aforementioned problems.
--!

SECTION_TO_RUBRIC = false
HEADING_TO_RUBRIC = false

--[[!
	By default, Doxyrest frames build a Python dictionary to be used later on by
	the ``:cref:`` (code-reference) role. This database maps language-specific
	qualified names of items to their IDs.

	This, together with setting the Sphinx ``default_role`` to ``cref``, allows
	to conveniently reference items from Doxy-comments or regular ``.rst`` files
	as such:

	.. code-block:: none

		The `ui::Dialog` class is a base to the `ui::FileDialog` class.

	However, if this facility is not used, building (and loading) of the
	cref-database can be omitted to save both space and time.
]]

CREF_DB = false

--[[!
	Exclude items with higher protection level than ``PROTECTION_FILTER``:

		1. ``public``
		2. ``protected``
		3. ``private``
		4. ``package``

	By default, only public items are included into documentation.
]]

PROTECTION_FILTER = "public"

--!
--! In many projects empty defines are *only* used as include-guards (and as
--! such, should be excluded from the documentation). If this is not the case
--! and empty defines should be kept in the final documentation, change this
--! setting to ``false``.
--!

EXCLUDE_EMPTY_DEFINES = true

--!
--! If non-``nil``, each define will be checked using this regular expression
--! and if its name matches, this define will be excluded from the documentation.
--!

EXCLUDE_DEFINE_PATTERN = nil

--!
--! Usually providing documentation blocks for default constructors is
--! not necessary (as to avoid redundant meaningless "Constructs a new object"
--! paragraphs). Change this to ``false`` if default constructors *should* be
--! included.
--!

EXCLUDE_DEFAULT_CONSTRUCTORS = true

--!
--! Usually providing documentation blocks for a destructors is
--! not necessary (as to avoid redundant meaningless "Destructs an object"
--! paragraphs). Change this to ``false`` if destructors *should* be
--! included.
--!

EXCLUDE_DESTRUCTORS = true

--[[!
	Usually providing documentation blocks for primitive C typedefs such as:

	.. code-block:: C

		typedef struct S S;

	is not necessary. Change this to ``false`` if such typedefs *should* be
	included.
]]

EXCLUDE_PRIMITIVE_TYPEDEFS = false

--!
--! For a base class/struct, show all the types directly derived from it.
--!

SHOW_DIRECT_DESCENDANTS = true

--[[!
	\subgroup

	Insert space between function name and parameter list like this:

	.. code-block:: C

		void foo ();

	By default, ``PRE_PARAM_LIST_SPACE`` is ``false`` which yields:

	.. code-block:: C

		void foo();
]]

PRE_PARAM_LIST_SPACE = false
PRE_OPERATOR_NAME_SPACE = true
PRE_OPERATOR_PARAM_LIST_SPACE = true

--[[!
	Insert a new line before the body of a enum, struct, class, etc.

	When ``PRE_BODY_NL`` is ``false``:

	.. code-block:: cpp

		class MyClass {
			...
		}

	When ``PRE_BODY_NL`` is ``true``:

	.. code-block:: cpp

		class MyClass
		{
			...
		}
]]

PRE_BODY_NL = true

--[[!
	Use multi-line parameter lists in function declarations if parameter count is
	greater than this threshold. ``nil`` means don't use parameter count
	threshold.

	For example, when ``ML_PARAM_LIST_COUNT_THRESHOLD`` is ``2``, the function
	declarations will look as such:

	.. code-block:: C

		void fn0();
		void fn1(int a);
		void fn2(int a, int b);

		void fn3(
			int a,
			int b
			int c
			);
]]

ML_PARAM_LIST_COUNT_THRESHOLD = nil

--[[!
	Use multi-line parameter lists in function declarations if single-line
	declaration length parameter count is greater than this threshold.
	``nil`` means don't use length threshold.

	Similar to ``ML_PARAM_LIST_COUNT_THRESHOLD``, but the threshold parameter
	here is *declaration length* and not *declaration parameter count*.

	Example:

	.. code-block:: C

		void foo(int a, int b, int c);

		void bar(
			int veryLongParameterName,
			int anotherVeryLongParameterName
			);

]]

ML_PARAM_LIST_LENGTH_THRESHOLD = 80

--[[!
	Use multi-line specifier-modifier lists in function declarations, i.e
	allocate a dedicated line for each type specifier/morifier.

	For example, when ``ML_SPECIFIER_MODIFIER_LIST`` is ``true``, the function
	declarations will look as such:

	.. code-block:: C

		void
		foo();

		static
		bool
		__cdecl
		bar(int a);
]]

ML_SPECIFIER_MODIFIER_LIST = false

--[[!
	Use the new C++11 syntax for ``typedef``-s:

	When ``TYPEDEF_TO_USING`` is ``false``:

	.. code-block:: cpp

		typedef unsigned int uint_t;
		typedef void Foo(int);
		typedef void (*FooPtr)(int);

	When ``TYPEDEF_TO_USING`` is ``true``:

	.. code-block:: cpp

		using uint_t = unsigned int;
		using Foo typedef void (int);
		using FooPtr typedef void (*)(int);
]]

TYPEDEF_TO_USING = false

--[[!
	Sometimes, it's required to redirect a Doxygen link to some external location.
	In this case, add an entry to ``EXTERNAL_CREF_DB`` with the target URL, e.g.:

	.. code-block:: lua

		EXTERNAL_CREF_DB =
		{
			[ "jnc.Scheduler" ] = "https://vovkos.github.io/jancy/stdlib/class_jnc_Scheduler.html",
			[ "strlen"] = "https://vovkos.github.io/jancy/stdlib/group_crt.html#doxid-function-strlen"

			-- ...
		}

	The key of the map is an ``importid`` attribute. This is a non-standard Doxygen
	attribute; Jancy compiler generates is when a referenced item is contained in an
	imported extensions library (``.jncx``)
]]

EXTERNAL_CREF_DB = {}

--[[!
	Usually providing documentation blocks for Lua ``local`` variables or
	functions is not desired. Change this to ``false`` if local items
	*should* be included in the final documentation.
]]

EXCLUDE_LUA_LOCALS = true

--! @}
