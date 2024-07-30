Contribute to OpenVINO
========================

.. toctree::
   :maxdepth: 1
   :hidden:

   contributing/code-contribution-guide

OpenVINO™ is always looking for opportunities to improve and your contributions
play a big role in this process. Here are four ways you can make OpenVINO better:

- `Provide feedback <#provide-feedback>`__
- `Contribute code changes <#contribute-code-changes>`__
- `Improve documentation <#improve-documentation>`__
- `Promote and support OpenVINO <#promote-and-support-openvino>`__


:fas:`comments` Provide feedback
################################

.. rubric:: Report bugs / issues
   :name: report-bugs-issues

If you notice unexpected behavior in OpenVINO or its components, you can
`create a new issue <https://github.com/openvinotoolkit/openvino/issues>`__
in the GitHub issue tracker.

.. rubric:: Propose improvements
   :name: propose-improvements

If you want to share your ideas for improving OpenVINO:

- Open a new `GitHub Discussion <https://github.com/openvinotoolkit/openvino/discussions>`__.
- Create a `Feature Request Issue <https://github.com/openvinotoolkit/openvino/issues/new?assignees=octocat&labels=enhancement%2Cfeature&projects=&template=feature_request.yml&title=%5BFeature+Request%5D%3A+>`__
  if your idea is already well defined.

In both cases, provide a detailed description and list potential use cases,
benefits, and challenges. Keep in mind that even if your input is not immediately
prioritized, it may be used at a later or undertaken by the community.


:fas:`code-branch` Contribute code changes
##########################################

Always check if the change is still needed! Verify if
`the issue <https://github.com/openvinotoolkit/openvino/issues>`__ or
`request <https://github.com/openvinotoolkit/openvino/pulls>`__ is still open
and nobody has started working on it. If the ticket is already work in progress,
you can always ask if you can help.

**Address only the issues that affect the master or**
:doc:`LTS release branches <./release-notes-openvino/release-policy>`.

**Do not start work on contributions, if a proper issue/ request has not been created.**

.. tip::

   If you want to start with something simple, check out
   `first-time contributions <https://github.com/orgs/openvinotoolkit/projects/3>`__.


.. rubric:: Fix bugs
   :name: fix-bugs

Choose one of the issues reported in
`GitHub Issue Tracker <https://github.com/openvinotoolkit/openvino/issues>`__ and
`create a Pull Request <https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING_PR.md>`__
(PR) addressing it.

If you find a new bug and want to fix it, you should still
create a new issue before working on the PR. This way, it will be easier for other
developers to track changes.

.. rubric:: Develop new features
   :name: develop-new-features

If you find a `Feature Request <https://github.com/openvinotoolkit/openvino/issues/new?assignees=octocat&labels=enhancement%2Cfeature&projects=&template=feature_request.yml&title=%5BFeature+Request%5D%3A+>`__
you want to work on, make sure it is clearly defined. If you have any doubts,
or the change is complex, `discuss it <https://github.com/openvinotoolkit/openvino/discussions>`__
with OpenVINO developers first.

If you have an idea for a new feature and want
to develop it, you should still create a Feature Request  before working on the
PR. This way, it will be easier for other developers to track changes.

.. rubric:: Develop a new device plugin
   :name: develop-new-device-plugin

If you want to run inference on a device that is currently not supported, you
can see how to develop a new plugin for it in the
`Plugin Developer Guide <https://docs.openvino.ai/nightly/documentation/openvino-extensibility/openvino-plugin-library.html>`__.


:fas:`file-alt` Improve documentation
#####################################

OpenVINO user documentation is built from several sources, mainly the files in
the `docs/articles_en <https://github.com/openvinotoolkit/openvino/tree/master/docs/articles_en>`__
folder, using `Sphinx <https://www.sphinx-doc.org/>`__ and the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`__
markup language.

OpenVINO `developer documentation <https://github.com/openvinotoolkit/openvino/tree/master/docs/dev>`__
is available only in markdown in the `docs/dev <https://github.com/openvinotoolkit/openvino/tree/master/docs/dev>`__
folder.

To edit docs, consider using the Editor’s
`guide <https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING_DOCS.md>`__
and contacting `documentation maintainers <https://github.com/orgs/openvinotoolkit/teams/openvino-docs-maintainers>`__,
who will help you with information architecture and formatting, as well as
review, adjust, and merge the PR.

.. rubric:: Review user documentation
   :name: review-user-documentation

In most cases, creating a PR is enough to correct a documentation mistake, improve
the language, and update or extend the information. For your convenience, the
top-right panel of most pages includes the “Edit on GitHub” button that will
take you to the source file of the given article.

.. rubric:: Write new content
   :name: write-new-content

For more extensive changes in docs, reach out to any of the
`documentation maintainers <https://github.com/orgs/openvinotoolkit/teams/openvino-docs-maintainers>`__
to discuss the new content.


:fas:`bullhorn` Promote and support OpenVINO
############################################

.. rubric:: Popularize OpenVINO
   :name: popularize-openvino

Articles, tutorials, blog posts, demos, videos, and any other involvement in the
OpenVINO community is more than welcome. If you discuss or present OpenVINO on
various social platforms, you are raising awareness of the product among AI
enthusiasts and enabling other people to discover the toolkit.

Feel free to reach out to OpenVINO developers if you need help with making a
contribution. You can also contact
`documentation maintainers <https://github.com/orgs/openvinotoolkit/teams/openvino-docs-maintainers>`__
, if you need help with visuals, brand materials, or content creation in general.

.. rubric:: Help other community members
   :name: help-community

If you are an experienced OpenVINO user and want to help, you can share your
expertise with the community at any time. Check GitHub
`Discussions <https://github.com/openvinotoolkit/openvino/discussions>`__
and `Issues <https://github.com/openvinotoolkit/openvino/issues>`__ to see if
you can help someone.

.. note::

   By contributing to the OpenVINO project, you agree that your contributions
   will be licensed under `the terms of the OpenVINO repository <https://github.com/openvinotoolkit/openvino/blob/master/LICENSE>`__.


Additional Resources
#####################

- :doc:`Code Contribution Guide <./contributing/code-contribution-guide>`
- Choose a `"Good First Issue" <https://github.com/orgs/openvinotoolkit/projects/3>`__.
- Learn more about `OpenVINO architecture <https://github.com/openvinotoolkit/openvino/blob/master/src/docs/architecture.md>`__.
- Check out a `blog post on contributing to OpenVINO <https://medium.com/openvino-toolkit/how-to-contribute-to-an-ai-open-source-project-c741f48e009e>`__.
- Visit `Intel DevHub Discord server <https://discord.gg/7pVRxUwdWG>`__ to join
  discussions and talk to OpenVINO developers.