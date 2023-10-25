# OpenVINO Documentation Guide

## Basic article structure

OpenVINO documentation is built using Sphinx and the reStructuredText formatting. 
That means the basic formatting rules need to be used:


### White Spaces

OpenVINO documentation is developed to be easily readable in both html and 
reStructuredText. Here are some suggestions on how to make it render nicely 
and improve document clarity.

### Headings (including the article title)

They are made by "underscoring" text with punctuation marks (at least as 
many marks as letters in the underscored header). We use the following convention:

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
to reading natural languages fairly well. For this reason, we aim at lines of around 
70 to 100 characters long. The limit is not a strict rule but rather a guideline to 
follow in most cases. The breaks will not translate to html, and rightly so, but will 
make reading and editing documents in GitHub or an editor much easier.

### Tables 

Tables may be difficult to implement well in websites. For example, longer portions 
of text, like descriptions, may render them difficult to read (e.g. improper cell 
widths or heights). Complex tables may also be difficult to read in source files. 
To prevent that, check the [table directive documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#table-directives)
and see our custom directives. Use the following guidelines for easier editing:

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

Use the following formatting for different links:

* link to an external page / file
  * `` `text <url> `__ ``
  * use a double underscore for consistency

* link to an internal documentation page / file
  * `` :doc:`a docs page <relative file path>` ``
  * Link to an rst or md file within our documentation, so that it renders properly in html

* link to a header on the same page
  * `` 'a header in the same article <this-is-section-header-title>`__ ``
  * anchors are created automatically for all existing headers
  * such anchor looks like the header, with minor adjustments:
    * all letters are lower case,
    * remove all special glyphs, like brackets,
    * replace spaces with hyphens 

* Create an anchor in an article
   * `` .. _anchor-in-the target-article:: ``
   * put it before the header to which you want to link
   * See the rules for naming anchors / labels at the bottom of this article
   
* link to an anchor on a different page in our documentation
   * `` :ref:`the created anchor <anchor-in-the target-article>` ``
   * link to the anchor using just its name


* anchors / labels 

  Read about anchors 

  Sphinx uses labels to create html anchors, which can be linked to from anywhere in documentation. 
  Although they may be put at the top of any article to make linking to it very easy, we do not use 
  this approach. Every label definition starts with an underscore, the underscore is not used in links.

  Most importantly, every label needs to be globally unique. It means that it is always a good 
  practice to start their labels with a clear identifier of the article they reside in.


