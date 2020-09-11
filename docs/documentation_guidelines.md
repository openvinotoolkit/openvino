# Documentation Guidelines

## Authoring Guidelines

We want our documentation to be easy to read and understand. This section describes guidelines for writing documentation that is clear, concise, confident, and courteous.

For details on organizing content and how we use Markdown, refer to [Structure and Formatting](#structure-and-formatting-guidelines).

###	Use Simple English

* **Be brief.** Use short sentences and paragraphs. Stick to the principle of one main idea per sentence, plus one additional point if needed. Each paragraph should address one main idea. Remember the basic structure of a paragraph: Introduction, body, and conclusion.
* **Use simple words.** Use simple words to increase reader comprehension and reduce ambiguity. Follow our tips for making good word choices:
   * **Avoid jargon:** Write for your audience, using everyday language where possible, and technical terms where appropriate. Avoid clichés, idioms, and metaphors.
   * **Be consistent:** Use one term for each concept or action and use it consistently.
   * **Avoid “fancy” words and phrases:** If there is a simpler word or phrase, use it.

###	Make Content Scannable

Organize your content to make it scannable for the reader, which helps them find what they need quickly, and to understand the information more efficiently.

* **Put the most important content first.** Make sure your introduction clearly communicates what the reader can find on the page. Present the point of the document first, and organize supporting information towards the end of the page.
* **Write scannable headings.** Expect readers of documentation to skim and scan the content, and to leave if they don’t find what they need quickly. Good headings add organization to your content and help the reader to find and understand content more effectively. Follow our guidelines for writing effective Headings.
* **Write great link text.** Great link text tells the reader what they can expect when they click on a link. It also helps make the page more scannable. Follow our guidelines for writing Link text.

###	Use Strong Verbs

* **Passive verbs make writing stuffy and formal**. Use strong verbs to get to the point and avoid unnecessary words and phrases.
* **Commands**, also called imperatives, are the fastest and most direct way of giving someone instructions.
* **Use simple present tense** instead of future tense for most text. Search for the words “will” or “shall” to find future tense instances. Future tense is acceptable for conditional statements, such as in a caution or a warning.

###	Parallelism

Parallelism refers to the practice of using similar patterns of grammar, and sometimes length, to coordinate words, phrases, and clauses.  
Use parallel construction in lists. The table below shows some unparallel structures and how they can be made parallel with a little rewording.

| Parallel (do)           | Unparallel (don’t)        |
|-------------------------|---------------------------|
| 1. Mount the panel.     | 1. Mount the panel.       |
| 2. Install the battery. | 2. Battery installation.  |
| 3. Wire the keypad.     | 3. Wiring the keypad.     |

## Structure Guidelines

Content should be organized to support scanning. Consistent organization, formatting, and writing style helps readers quickly find what they need and to understand the content more effectively. This section describes our organization.

### Website Structure
The documentation website is organized this way:

1. **Getting Started**: installation guides, prerequisites, how to set working environment
2. **How To-s**: videos and guides, for some reason not included in **Guides** section
3. **Guides**: developer guides for all OpenVINO Toolkit components
4. **Resources**: samples, demos, pre-trained models and tools (Accuracy Checker, Post-Training Optimization etc)
5. **Performance Information**: benchmark tests information, ways to improve Performance
6. **API References**: in-built wiki with class structures

### Page Structures
Each page in the documentation should follow the basic format of:

* **Overview**: 1-2 sentences describing what this page shows and why it matters
* **Prerequisites**: Describe any pre-work necessary to the content (if appropriate)
* **Content**
* **Next steps**: List links to next steps (if appropriate)
* **Related topics**: List links to related content (if appropriate)

## Formatting Guidelines
We apply the formatting using Markdown.

### Inline Text Formatting
Use our quick reference for the most commonly used inline text elements:

| Element                 | Markdown Examples            |   Notes |
|---------------          |---------------------| -------- |
| Notes                   | `> **NOTE**: text`   | The space between `>` and `**NOTE**` is required  |
| Code elements (variables, methods, classes)      | ``` `text` ```      |              |
| Code snippets           | ` ```cpp`<br>text<br>` ``` `      |                         |
| Console commands, output| ` ```sh`<br>text<br>` ``` ` |                               |
| Product name	          | OpenVINO™,<br>Intel® Distribution of OpenVINO™ toolkit  |   |

### Headings
Use headings to section and organize your content for better readability and clarity.

* **Use strong verbs.** Strong, active verbs get to the point. Avoid **-ing** verbs, such as Running, Testing, etc.
* **Be concise and descriptive.** Use only the words necessary to describe the section.
* **Use sentence case.** Capitalize first letter every word except articles, prepositions: **Build Inference Engine on Linux**
* **Avoid punctuation.** Unless your heading is a question, don’t use sentence punctuation in headings.
* **Use parallel structure.** Headings at the same level should use the same grammatical pattern. This provides structure to the document and helps users find information more easily. See [Parallelism](#parallelism).

Use following rules to write headings in your documentation:
* All files must have a top level heading, which is the title for the page.
  ```
  # First Level Heading
  ```
* Up to three additional levels of headings are allowed under the title heading.
  ```
  ## Second Level Heading
  ### Third Level Heading
  #### Fourth Level Heading
  ```
* Each heading should be followed by at least one paragraph of content. Avoid two or more consecutive headings.
* To add a link to a heading in the current page, do the following:  
    * Add the `<a>` HTML tag and set a unique name to your heading with `name` argument:
    ```XML
    ## <a name="set-the-environment-variables"></a> Set the Environment Variables
    ```
    * Place the link to the heading using the set name:
    ```XML
        Refer <a href="#set-the-environment-variables">here</a>
    ```

### Code Snippets and Examples
You can create fenced code snippets by placing triple backticks ` ``` `  before and after the code block and add an optional language identifier to enable syntax highlighting in your fenced code block. For example, to syntax highlight C++ code:
```
```cpp
#include <inference_engine.hpp>
void main(int argc, char *argv[])
{
    InferenceEngine::Core ie;
}
```

```cpp
#include <inference_engine.hpp>
void main(int argc, char *argv[])
{
    InferenceEngine::Core ie;
}
```

For full list with supported languages refer [here](https://github.com/github/linguist/blob/master/lib/linguist/languages.yml).

### Lists and Instructions
This section provides information about formatting numbered and bulleted lists.

#### Numbered Lists
Use a numbered list when the order or priority of the items is important, such as step-by-step instructions:
```
1. **Initialize core** to..
2. **Prepare input** for..
3. ...
```
Numbered lists are most frequently used for procedures. Use numbered lists to show sequence for the items. Follow our guidelines for numbered lists:
* Make few first words describe whole step and bold it:  
  6) **Prepare input**. You can use one of the following options to prepare input
* Introduce a numbered list with a sentence. End the setup text with a colon. Example:
  "To configure the unit, perform the following steps:"
* Each item in the list should be [parallel](#parallelism).
* Treat numbered list items as full sentences with correct ending punctuation.
* You may interrupt numbered lists with other content, if relevant, e.g. explanatory text, commands, or code.

#### Bulleted lists
Use a bulleted list when the order of the items is not important:
```
* Option X
* Option Y
* Option Z
```
* Introduce a bulleted list with a sentence. End the setup text with a colon. Example:  
  “To repair the unit, you will need the following items:”
* Make few first words describe whole step and bold it:  
  * **Prepare input**. You can use one of the following options to prepare input
* Each item in the list should be parallel.
* Avoid interrupting bulleted lists with other paragraph styles.
* Second-level bullets are acceptable; avoid third-level bullets.

For both list types, keep all items in the list parallel. See [Parallelism](#parallelism):

### Links
All links in content should follow these guidelines:

* **Avoid generic text:** Don’t use generic, uninformative link text such as “click here” or “read more”.
* **Write descriptive link text:** Link text should describe where the link goes, without having to read the surrounding text.
* **Use unique link text:** Each link text on a page should be unique. If users see the same link text twice on a page, they’ll assume it goes to the same place.
* **Start link text with keywords:** Frontload the link text with the most important words to help users scan the text.

Use following examples to write links in your documentation:
* To add a cross-reference to another documentation page, use file path from [DoxygenLayout.xml](./openvino-documentation/blob/master/docs/doxygen/DoxygenLayout.xml) (don't use direct links to docs.openvinotoolkit.org):
```
[OpenVINO Installation on Linux](./docs/install_guides/installing-openvino-linux.md)
```
Currently links to headings in other markdown files are not supported. To refer to a specific section, you may use the following example:  
```
"Use **Set Environment Variables** section in the [OpenVINO Installation on Linux](./docs/install_guides/installing-openvino-linux.md) document.
```
* To add URL link to any software, link to the latest version, for example:
```
For more details, see [CMake page](https://cmake.org/cmake/help/latest/manual/cmake.1.html#manual:cmake(1))
```

## Graphics Guidelines
Use images or figures to convey information that may be difficult to explain using words alone. Well-planned graphics reduce the amount of text required to explain a topic or example.

Follow these guidelines when using graphics in support of your documentation:

* Keep it simple. Use images that serve a specific purpose in your document, and contain only the information the reader needs.
* Avoid graphics that will need frequent updating. Don’t include information in a graphic that might change with each release, such as product versions.
* Use either PNG or JPEG bitmap files for screenshots and SVG files for vector graphics.
* Place the image immediately after the text it helps clarify, or as close as possible.
* Use the Markdown directives to insert images and figures into the document. Include both alt text, and the title.
* Include at least one direct reference to an image from the main text, using the figure number.

Images should follow these naming and location conventions:

* Save the image files in a figures folder at the same level as the file that will reference the image.
* Name image files according to the following rules:
   * Use only lower case letters.
   * Separate multiple words in filenames using dashes.
   * Name images using the filename of the file they appear on and add a number to indicate their place in the file. For example, the third figure added to the `welcome.md` file must be named `welcome-3.png`.
