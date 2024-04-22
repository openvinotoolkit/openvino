# OpenVINO Documentation Guide

## Table of contents

* [Basic article structure](#basic-article-structure)
  * [White spaces](#white-spaces)
  * [Headings (including the article title)](#headings-including-the-article-title-)
  * [Line length](#line-length)
  * [Tables](#tables)
* [Cross-linking](#cross-linking)
* [File naming and structure](#file-naming-and-structure)
* [Sphinx directives](#sphinx-directives)
  * [Code block formatting](#code-block-formatting)
  * [Images](#images)
  * [Tabs](#tabs-sphinx-design-documentation)
  * [Panels and drop-downs](#panels-and-drop-downs-sphinx-design-documentation)
  * [Html includes](#html-includes)
  * [Escaping functional characters (e.g. double hyphens)](#escaping-functional-characters-eg-double-hyphens)
  * [Tables](#tables-1)
  * [ScrollBox](#scrollbox)
  * [Admonitions / disclaimers](#admonitions--disclaimers)
* [Writing Tips](#writing-tips)
* [Screenshots and videos](#screenshots-and-videos)
  * [Screenshots](#screenshots)
  * [Videos](#videos)

## Basic article structure

OpenVINO documentation is built using [Sphinx](https://www.sphinx-doc.org/) and the [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html) formatting.
That means the basic formatting rules need to be used:

### White spaces

OpenVINO documentation is developed to be easily readable in both html and
reStructuredText. Here are some suggestions on how to make it render nicely
and improve document clarity.

### Headings (including the article title)

They are made by "underscoring" text with punctuation marks (at least as
many marks as letters in the underscored header). Use the following convention:

```
   H1
   ====================

   H2
   ####################

   H3
   ++++++++++++++++++++

   H4
   --------------------

   H5
   ....................
```

### Line length

In programming, a limit of 80 characters per line is a common BKM. It may also apply
to reading natural languages fairly well. For this reason, the aim are lines of around
70 to 100 characters long. The limit is not a strict rule but rather a guideline to
follow in most cases. The breaks will not translate to html, and rightly so, but will
make reading and editing documents in GitHub or an editor much easier.

### Tables

Tables may be difficult to implement well in websites. For example, longer portions
of text, like descriptions, may render them difficult to read (e.g. improper cell
widths or heights). Complex tables may also be difficult to read in source files.
To prevent that, check the [table directive documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#table-directives)
and see our [custom directives](#tables-1). Use the following guidelines for easier editing:

* For very big and complex data sets: use a list instead of a table or remove
  the problematic content from the table and implement it differently.
* For very big and complex data sets that need to use tables: use an external
  file (e.g. PDF) and link to it.
* For medium tables that look bad in source (e.g. due to long lines of text),
  use the reStructuredText list table format.
* For medium and small tables, use the reStructuredText grid or simple table formats.

## Cross-linking

There are several directives Sphinx uses for linking, each has its purpose and format.
Follow these guidelines for consistent results:

* Avoid absolute references to internal documents as much as possible (link to source, not html).
* Note that sphinx uses the "back-tick" character and not the "inverted-comma" => ` vs. '
* When a file path starts at the same directory is used, put "./" at its beginning.
* Always add a space before the opening angle bracket ("<") for target files.

Use the following formatting:

* Links

  * Link to an external page / file
    * `` :doc:`a docs page <relative file path>` ``

      Directs to an rst or md file within our
      documentation, so that it renders properly in html.

  * Link to a heading on the same page
    * `` 'a heading in the same article <#this-is-section-heading-title>`__ ``

      Anchors are created automatically for all existing headings. The link to a heading
      includes the text of a heading with minor adjustments:
      * prefixed with `` # ``
      * all letters are lower case,
      * all special glyphs, like brackets are removed,
      * spaces are replaced with hyphens.

* Anchors

  * Anchor in an article
     * `` .. _anchor_in_the_target_article:: ``

        Placed before the target heading, every anchor:
         * starts with an underscore,
         * is globally unique, based on the article title,
         * is written in snake_case.

  * Link to an anchor on a different page in our documentation
     * `` :ref:`the created anchor <anchor_in_the_target_article>` ``

       Directs to a unique anchor across the documentation.

## File naming and structure

These conventions facilitate consistency and clarity of our articles, but also allow
Sphinx to build documentation pages without additional issues.

* File names (HTML file name)
  * kebab-case (all lower-case)
  * title-related, but not necessarily the same
  * descriptive, including prefixes grouped in a non-disruptive manner
  * prefixed with `` '[legacy]-' `` if an article is deprecated but cannot be removed

* Article titles
  * organic, meaningful, not overly long (around 35 characters is a good limit)
  * not necessarily unique, but should be if possible
  * noun-based, where present participle as the first word is avoided,
  * the imperative is preferred
  * title case - with every word capitalized
  * no full stop at the end
  * prefixed with `` '[LEGACY]-' `` if an article is deprecated but cannot be removed

* Headings
  * levels H2 and lower in sentence case (only the first word capitalized),
  * lower-level headings include more characters than the top ones
  * other rules as per article titles

* Files (assets)
  * named in a similar vein as article files
  * meaningful and specific names

* Additional assets (directory structure)
  * images, css files, JS scripts

    Placed in a corresponding subfolder of the `` sphinx_setup/_static `` directory, which is always moved to the server.
    Files not referred to explicitly in rst directives are not moved to the server, therefore they should be placed in `` sphinx_setup/_static ``.

  * code snippets

    Put in the `` docs/snippets `` folder, one file only for a given programming language (one file for C++ and one file for Python).

* Parent articles (directory structure)
  * articles including a TOC (child articles)

    Placed at the same level as the folder containing their child articles

  * the `` docs/articles_en `` directory

    Its directory structure should be approximately consistent with what is shown on the https://docs.openvino.ai

## Sphinx directives

Sphinx allows to format content using a basic set of common options.
Here is a list of styling options together with descriptions of when and how to use them.
Some directives may be specific to our framework configuration and may not be covered by
external Sphinx resources.

> **NOTE**: Sphinx directives are sensitive to white spaces! Every directive should retain
a line break before it starts and three-space for every line of its potential content.

### Code block formatting

Literal code formatting is achieved in several ways:

* inline - put the text between double backticks: \`` `` text formatted as literal text `` \``

* as a block - use the ``.. code-block::`` directive and add the name for the linter (``py``, ``c``, ``cpp``, ``sh``, ``console``, etc.)

  Example:
  ```
  .. code-block:: sh

     text formatted as literal text
  ```

* from a code snippet - use the ``.. doxygensnippet::`` directive.

  Always use the brackets in the ``fragment`` parameter, for more build reliability:

  Example:
  ```
  .. doxygensnippet:: the path to the snippet file
     :language: language name for the linter, e.g, py, c, or cpp
     :fragment: [the name of the snippet fragment to render]
  ```

Below is a list of the most commonly used linters in OpenVINO documentation:

<details><summary>Linters</summary>

| Name | Linter |
|:---:|:---:|
| Bash | sh |
| Bash Session | console |
| Batch file | bat |
| CMake | cmake |
| C | c |
| C++ | cpp |
| Python | py |

</details>

### Images

It is essential that a common directory structure is used. Images should be located in
[``docs/sphinx_setup/_static/images``](https://github.com/openvinotoolkit/openvino/tree/master/docs/sphinx_setup/_static/images)
so ``_static/images/image.png`` can be used in links.

Example:
```
.. image:: _static/images/image_name.png
   :width: 300px
   :align: center
```

### Tabs ([sphinx-design documentation](https://sphinx-design.readthedocs.io/en/latest/tabs.html))

Different versions of content are displayed one at a time, as per reader's choice,
to make them unobtrusive.

* Tabs should be applied to small portions of pages, making most of page content unaffected.
  * should not be used for big blocks of text and definitely not for entire pages
  * should not be used for very small content blocks, when it feels distracting and
    unnecessary - instead all text versions can be visible at once

  Tabs are created from the sphinx-design addon, which allows synchronizing tabs with the
  same label. For more build reliability and consistency, follow these ``:sync:``
  label guidelines:
    * if the ``.. tab-item::`` name is a language, use the appropriate linter for ``:sync:``
      (see linters in the code block formatting section above)
    * if the ``.. tab-item::`` name is not a language, use the same name for ``:sync:``
      (all lower-case letters)
    * if the ``.. tab-item::`` name consists of two or more words, you may decide not to use
      ``:synch:`` if it is likely to be the only instance of this tab name.
      If you still want to use ``:sync:``,  separate the words with ``-`` instead of spaces.

    Example:
    ```
    .. tab-set

       .. tab-item:: Python
          :sync: py

          .. code-block:: py

             content displayed within a code block and visible only when the Python tab is active - the Python linter name used for synch

        .. tab-item:: Windows
           :sync: windows

           content visible only when the Windows tab is active, this time without a code block

       .. tab-item:: Get started with Python
          :sync: get-started-with-python

          content visible only when the Get started with Python tab is active
    ```

### Panels and drop-downs ([sphinx-design documentation](https://sphinx-design.readthedocs.io/en/latest/dropdowns.html))

Content placed in responsive text frames, which can be arranged in different grid layouts
and styled easily. Drop-down panels hide content for readers to expand by clicking the
panel, if necessary.

Example 1:
```
.. dropdown:: Dropdown title

   Dropdown content
```
Example 2:

A dropdown with explicit content as default.
```
.. dropdown:: Open dropdown
   :open:

   Dropdown content
```

### Html includes

You can add HTML content and CSS styling in an article directly, but it should be used
scarcely and only if necessary.

Example:
```
.. raw:: html

   <link rel="stylesheet" type="text/css" href="_static/css/style_sheet_name.css">

   <p>paragraph</p>
   <div id="container">
   <a href="link" >link</a>
   </div>
```

### Escaping functional characters (e.g. double hyphens)

In Sphinx, you can use a backslash (``\``) to escape the character following it.
For example, double hyphens (``--``) may be transformed into a dash (``–``) in the HTML
output. This can happen,  in a code piece within the (``.. note::``) admonition. To output
the correct format, escape the double-hyphen.

Example:
```
.. note::

   To get additional information on all parameters that can be used, check up the help option:
   `\--help`. Among others, you can find there `-s` option which offers silent mode, which
   together with `\--eula approve` enables you to run the whole installation with default
   values without any user interference.
```

### Tables

Tables can be made in several different ways. The preferred ones are the grid table and the
list table. The grid table does not require any additional directives, it just needs
to adhere to the table formatting rules. The list directive works well for longer tables with
longer content, for example long links. More complicated options are also available,
such as csv import and LaTex-compatible class styling. For help in creating tables,
see [tablesgenerator.com](https://www.tablesgenerator.com/text_tables)

You can use a custom class of ``sort-table`` to make tables sortable, if necessary.
Add the class via the ``.. container::``  directive.

Examples:
```
.. container:: sort-table

   +-------+-------------+
   | Col 1 | Col 2       |
   +=======+=============+
   | 1     | Lorem Ipsum |
   +-------+-------------+
   | 2     | Lorem Ipsum |
   +-------+-------------+


.. list-table:: Title
   :widths: 25 25 50
   :header-rows: 1

   * - Heading row 1, column 1
     - Heading row 1, column 2
   * - Row 1, column 1
     - Row 1, column 3


.. tabularcolumns:: |p{1cm}|p{7cm}|

.. csv-table:: Lorem Ipsum
   :file: _files/lorem-tab.csv
   :header-rows: 1
   :class: longtable
   :widths: 1 1
```

### ScrollBox
Our custom directive creating a block of any content enclosed in a self-scrollable frame.
It can be used for a long table, so that readers can scroll them without scrolling
the entire page.

It accepts the following parameters:

* ``name`` – assigns id for the element in HTML
* ``height`` – numeric values, with px as units (default: ``300px``)
* ``width`` – numeric values, with ``%`` or ``px`` as units (default: ``100%``)
* ``delimiter`` – adds a left border to the element, accepts numeric values, with ``px`` as
  units (for example: ``1px``)
* ``delimiter-color`` – assigns color to the set delimiter, takes hexadecimal values
 (``#368ce2``) or string values (``blue``)

Example:
```
.. scrollbox::
   :name: thisismyid
   :height: 500px
   :width: 100%
   :delimiter: 2px
   :delimiter-color: #368ce2

   +-------+-------------+
   | Col 1 | Col 2       |
   +=======+=============+
   | 1     | Lorem Ipsum |
   +-------+-------------+
   | 2     | Lorem Ipsum |
   +-------+-------------+
```

### Admonitions / disclaimers

The⠀following⠀admonition types are used in the documentation:

* [``note``](https://docutils.sourceforge.io/docs/ref/rst/directives.html#note) -
  for general disclaimers (rendered in blue)
* [``tip``](https://docutils.sourceforge.io/docs/ref/rst/directives.html#tip) and
  [``important``](https://docutils.sourceforge.io/docs/ref/rst/directives.html#important) -
  for additional suggestions (rendered in green)
* [``warning``](https://docutils.sourceforge.io/docs/ref/rst/directives.html#warning) -
  for adding highly important and possibly problematic detail (rendered in yellow)
* [``danger``](https://docutils.sourceforge.io/docs/ref/rst/directives.html#danger) -
  for warning against critical problems (rendered in red)

Example:
```
.. note::

   Text within admonitions will be styled as a disclaimer, enclosed in a frame, with a proper
   header and color.
```

## Writing Tips

* Trademarks

  We used to put asterisks (*) next to third-party brands to indicate trademarks. Not any more.
  Since the 2021.4 we have been allowed by the legal team to drop the notation.
  Still, we need to annotate Intel and OV products so:

  ☒ TensorFlow* and Keras* models are supported by OpenVINO and optimized for Intel® solutions

  ☑ TensorFlow and Keras models are supported by OpenVINO™ and optimized for Intel® solutions

  Importantly, the mark needs to be included only at the first mention of each brand. Any
  following mention may skip the mark.

* Tool names and "THE"

  Names of our tools and components, such as Model Optimizer, are used extensively throughout
  the documentation. It may sometimes be problematic to decide if they require   articles
  or not. A good guideline is to never put articles (a, an, the) in front of anything that
  can be written with capital letters and abbreviated, so:

  ☒ the Model Optimizer and the OpenVINO™

  ☑ Model Optimizer and OpenVINO™

* Code elements, e.g., methods, parameters, commands

  Code elements behave like names and values, so are not preceded by articles (a or the),
  except when used as a descriptor of a noun. They should retain the original format, which
  means we do not capitalize the first letter, even if the element starts a new sentence.
  Importantly, always put code elements in "back-ticks" (in RST we use two backticks in
  a row - \`` ``code_element`` \``) to format them as an in-line code block.

  |    |    |
  |:---|:---|
  | ☒ The ov::InferRequest provides methods for model inference. | ☑ ``ov::InferRequest``  provides methods for model inference. |
  | ☒ \`` ``ov::InferRequest`` \`` class provides methods for model inference. | ☑ The \`` ``ov::InferRequest`` \`` class provides methods for model inference. |

* Oxford comma

  For lists of two elements, no comma before "and." With three or more elements, put a comma
  before "and." When nested in a larger list, "and" gets no comma.

  ☒ I ate a hot-dog, a meatloaf and a tortilla. (hot-dog is not a meatloaf and a tortilla put
  together, as it might be understood)

  ☑ I ate a hot-dog, a meatloaf, and a tortilla. (we have three separate food articles)

  ☑ We ate a hot-dog and Cola, pizza and Fanta, a burger and beer, and nachos and Sprite.

* active vs. passive

  Most documentation guides discourage passive voice. Yet it is still a valid and
  stylistically proper way of writing documentation. Instead of avoiding it, you should
  simply not overuse it. If you can word the sentence in either voice with equal results,
  choose the active one. However, it should be at your discretion to decide when the
  passive fits better, for example, improving message clarity, avoiding repetitions,
  or avoiding subject changes.

  ☒ If your files haven't been backed up, they should be immediately.

  ☒ The person or service responsible for backups should back up configuration files regularly.

  ☑ If you haven't backed up your files, you should do it immediately.

  ☑ Configuration files should be backed up regularly.

* person (you... one, user, I, we)

  Choosing the active voice may require you to refer to a hypothetical subject.
  Contrary to the academic style of writing, technical documentation favors referring
  to the reader directly and often applies the imperative voice. It is especially
  pronounced in instructions. Therefore, avoid using "one" and "the user." Use "you"
  instead. Self-references, using the first person, should be avoided. If it is
  imperative to the meaning of the sentence, a collective noun should be used instead
  of the "I" and "we" pronouns.

  ☒ One should always remember to back up their files.

  ☒ We can always help you by reviewing your documentation contributions.

  ☑ You should always remember to back up your files.

  ☑ OpenVINO technical writers can always help you by reviewing your documentation contributions.


* Common words and phrases
  * i.e., / e.g.,

    For consistency, assume the American English style of putting a comma after the two
    abbreviations. Both options are grammatically correct. Usage of these abbreviations is
    considered highly formal.

    ☒ Dynamic shapes, i.e. with dynamic dimensions ...

    ☑ Dynamic shapes, i.e., with dynamic dimensions ...

  * inference

    In most cases, "inference" will be uncountable, thus, not require an article.
    In specific cases, it can be used as countable, accepting "the."

    ☒ an inference

    ☑ inference

    ☑ The inference you have performed can be used as a benchmark.

  * page

    Use "on this page" ... "in this page" is not correct in most cases

  * please

    It should be avoided as we are not asking the user, we are instructing them.


    ☒ To start the operation, please use the 'launch' command.

    ☑ To start the operation, use the 'launch' command.

  * reshapable

    Alternative spelling: reshapeable. A neologism used widely in IT and design.
    The alternative spelling seems more popular but less correct.

  * hyphenated compounds

    The choice between hyphenated and non-hyphenated compound nouns is often a matter of
    preference. When in doubt, use the general rule of: hyphens for adjectives before nouns.
    Hyphenating prefixes, like "pre," is typically discouraged but with some words it is so
    widely adopted that it should be accepted as the default. What's important, sometimes it
    is better to use a hyphen against a grammatical convention, if it benefits clarity
    and ease of understanding. For example, co-operation is easier to read and pronounce
    than cooperation.

    For consistency, use the following compound nouns and adjectives:

    <details><summary>Examples:</summary>

    | adjective       | noun           | additional info                                                                   |
    |-----------------|----------------|-----------------------------------------------------------------------------------|
    | command-line    | command line   |                                                                                   |
    | device-specific |                |                                                                                   |
    | end-to-end      |                | end to end may be used as an adjective                                            |
    | frontend        | frontend       |  never front-end                                                                  |
    | open-source     | open source    | never open-sourced (even with the meaning of 'released as open source')           |
    | pre-processing  | pre-processing | due to the popularity criterion                                                   |
    | pre-release     |                | the noun "prerelease" should be avoided.                                          |
    | pre-trained     |                |  should be avoided in meanings other than the adjectival. click here for details. |

    </details>

## Screenshots and videos

### Screenshots

Screenshots have always been a very common way of presenting content in articles.
Unfortunately, they are also commonly misused - too many with too little consistency,
producing content which is clumsy and hard to read.

To see how images are treated in Sphinx, check out
[Sphinx directives and formatting](https://docutils.sourceforge.io/docs/ref/rst/directives.html#image).
To make documents clear, informative, and easy to follow, keep to several rules below.

* Use the same app to get consistent results and make documents look like made by members
  of the same team.

  Greenshot ([click to download](https://github.com/greenshot/greenshot/releases/download/Greenshot-RELEASE-1.2.10.6/Greenshot-INSTALLER-1.2.10.6-RELEASE.exe))
  is a simple but versatile screen capturing tool. The default values provided by Greenshot
  are fine, but if you want to tweak how it works, here is its
  [F.A.Q.](https://getgreenshot.org/faq)

* **Annotate screenshots with numbers** instead of writing on them or leaving them blank.

  * Refer to the numbers within textual content of the article.
    Put numbers inside the square brackets within text - ``[ 1 ]``.

    (The spaces here are not grammatically correct, but they help with Confluence's
    auto-correction mechanism and facilitate clarity.)

  * Annotate the screenshot with the number from the bracket.
  * To add numbers, use the default settings provided by Greenshot ("``Add Counter``"
    with the "``i``" hotkey).
  * If you describe many elements of the same or two similar screens, put multiple
    numbers in one screenshot, instead of producing many images.

  <details><summary>Examples:</summary>

  ☒ BAD

  Greenshot works well right after installation, but you may choose to customize it using
  the Preferences panel. Avoid extensive customization, as it is recommended to produce
  consistent screenshot annotations. The settings you should consider to adjust are
  Hotkeys for Capturing a window and a region.

  ![](docs/articles_en/assets/images/bad-annotated-screenshot-1.png)

  ![](docs/articles_en/assets/images/bad-annotated-screenshot-2.png)

  ☑ GOOD

  Greenshot works well right after installation, but you may choose to customize it using
  the Preferences panel [ 1 ]. Avoid extensive customization, as it is recommended to produce
  consistent screenshot annotations. The settings you should consider to adjust are
  Hotkeys for Capturing a window [ 2 ] and a region [ 3 ].

  ![](docs/articles_en/assets/images/good-annotated-screenshot.png)

  </details>

* **Do not put the picture inside a paragraph or a list.** Instead, put the annotated
  screenshot at the bottom of the paragraph, or the whole article.

  This may seem quite controversial, as mixing text with images is a common practice.
  However, it is very easy to create a mess with this approach. Instead, you should try not
  to break the flow of text too much.

* **Do not make the picture very big**, both visually and byte-wise, unless it is the
  main source of information (as opposed to being an addition to the text).
  To do so you can:

  * crop the image so that it shows only the relevant part of the original screen
    (for example, only the described window)
  * downsize the image - if you capture the image in 4K or 8K, the page will load slowly
    and the image will occupy too much space when clicked. In this case, less is more.
  * choose its size when putting the image in the article, maybe even consider formatting
    it as a clickable thumbnail.

* **Underline only the elements which need to be highlighted**, such as areas, very
  specific elements (hard to mark with a number), multiple elements assigned to one
  number, etc.

  Use ONLY three types of highlights and use their default settings
  (red, 2pt. line, drop shadow):

  * a rectangle, e.g., for an area,
  * a line, e.g., for a specific text
  * an arrow, e.g., for a specific element

### Videos

Videos have become a very popular way of providing instruction, yet they are not always
the best choice for the documentation. Regardless of their popularity, videos are not the
easiest nor the quickest way of learning new things. They are difficult to maintain
because it is "impossible" to edit them quickly, which means they usually stay outdated.

In the typical documentation environment, they should be used exclusively for very short
tutorials, like presenting small procedures or the clicking order for some interfaces.
If you know what you are doing making a video, here is a short description of a
relatively straightforward procedure:


* Recording

  Install [ShareX](https://getsharex.com/), an open source screen recording application.
  It allows you to adjust a lot of settings for the application and the recording task itself
  but it works well out of the box. The only thing required before recording is downloading
  the ffmpeg.exe video codec file and placing it in ``Documents\sharex\tools`` on your computer.

  To record a video:

  1. BKM - launch the application with DPI compatibility adjustments. This prevents problems
     with high DPI of new, big screens and Windows scaling, resulting in the mouse cursor
     disappearing in your videos.
  2. Launch Sharex and in the panel on the left choose "Capture => Screen recording."
  3. With the new cursor, mark the area to immediately start capturing it.
  4. When ready, click stop and the mp4 video will be saved in your documents instantly.

* Editing

  Install ShotCut (https://shotcut.org/), an open source non-linear video editor. If you
  need a more robust tool and can accept a more restrictive license, you can
  try the professional DaVinci Resolve suite.

  To perform edits:

  1. Drag and drop the recorded file on the Playlist panel on the left.
  2. From the playlist, move the clip to the timeline at the bottom - make sure it is
     aligned to the left border, that is the beginning of the video.
  3. Clicking and dragging the play-head (the white triangle above the timeline clip with
     a vertical line) browse through the video and find the parts that are unnecessary.
  4. Mark the beginning of the part to remove and press the "s" key to cut the clip at that spot.
  5. Cut the clip one more time at the end of the piece to remove.
  6. Click the newly cut part and press "delete" to remove it.
  7. Drag the remaining clip until it touches the piece on its left.
  8. Continue cutting out all the unnecessary parts of the main clip until you find it
     satisfactory. Remember, you can always correct the cut by grabbing the clip's edge
     and moving it to either side.
  9. When ready, choose "export" in the top menu, choose the default preset, open the "advanced"
     panel to change frames/sec to 24, and possibly disable audio if no sound is used.
  10. Click "export file" to save it.

