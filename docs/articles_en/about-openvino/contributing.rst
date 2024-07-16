Contribute to OpenVINO
========================

.. toctree::
   :maxdepth: 1
   :hidden:

   contributing/technical-guide

OpenVINO™ is always looking for opportunities to improve and your contributions
play a big role in this process. There are several ways you can make the product better.
Keep in mind that even if your input is not immediately prioritized, it may be used
at a later time or undertaken by the community, regardless of the official roadmap.

:fas:`comments` Provide feedback
################################

.. rubric:: Report bugs / issues
   :name: report-bugs-issues

If you notice unexpected behavior in OpenVINO or its components, you can
`create a new issue <https://github.com/openvinotoolkit/openvino/issues>`__
in the GitHub issue tracker.

.. rubric:: Propose new features / improvements
   :name: propose-new-features

If you want to share your ideas for improving OpenVINO, you can open a new
`GitHub Discussion <https://github.com/openvinotoolkit/openvino/discussions>`__.
If your idea is already well defined, you can also create a
`Feature Request Issue <https://github.com/openvinotoolkit/openvino/issues/new?assignees=octocat&labels=enhancement%2Cfeature&projects=&template=feature_request.yml&title=%5BFeature+Request%5D%3A+>`__
Remember to provide a detailed description, including use cases, benefits,
and potential challenges.


:fas:`code-branch` Contribute code changes
##########################################

Before you start a contribution, check if the change has not been implemented already.
If it happens to be in progress, align with the developer to work cooperatively.
Remember to build OpenVINO, using the latest master branch and verify that your
changes are still applicable. Do not address issues that only affect older non-LTS
releases, like 2022.2. For more information, refer to the
:doc:`Release Policy <./release-notes-openvino/release-policy>`.

.. rubric:: Fix bugs
   :name: fix-bugs

If you want to help improving OpenVINO, choose one of the issues reported in
`GitHub Issue Tracker <https://github.com/openvinotoolkit/openvino/issues>`__ and
`create a Pull Request <https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING_PR.md>`__
addressing it. Consider one of the tasks listed as
`first-time contributions <https://github.com/orgs/openvinotoolkit/projects/3>`__.

.. rubric:: Develop new features
   :name: develop-new-features

If the feature you want to develop is more complex or not well defined yet,
you may want to `discuss it <https://github.com/openvinotoolkit/openvino/discussions>`__
with OpenVINO developers first.

.. rubric:: Develop a new device plugin
   :name: develop-new-device-plugin

OpenVINO is always open to extending its support for new hardware. If you want to
run inference on a device that is currently not supported, you can see how to
develop a new plugin for it in the
`Plugin Developer Guide <https://docs.openvino.ai/nightly/documentation/openvino-extensibility/openvino-plugin-library.html>`__.

See :doc:`technical guide for contribution <./contributing/technical-guide>`
for further details on how to start contributing code changes.

:fas:`file-alt` Improve documentation
#####################################

You are welcome to review and provide feedback on the existing articles.
Whether you notice a mistake, see the possibility of improving the text,
or think more information should be added, you can reach out to any of the `documentation
maintainers <https://github.com/orgs/openvinotoolkit/teams/openvino-docs-maintainers>`__ to discuss the potential changes.
You can also edit the articles directly on GitHub or open them with
`Github.dev <https://github.dev/openvinotoolkit/openvino>`__. Do not forget to create
a **Pull Request**, following the `guidelines <https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING_DOCS.md>`__.

.. rubric:: Review user documentation
   :name: user-documentation

The documentation is built from several sources, mainly
`docs <https://github.com/openvinotoolkit/openvino/tree/master/docs>`__ folder,
using `Sphinx <https://www.sphinx-doc.org/>`__ and the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`__
markup language. It is published at `docs.openvino.ai <https://docs.openvino.ai/>`__,
which is recommended for reading the articles.

.. rubric:: Check OpenVINO developer documentation
   :name: openvino-developer-documentation

The documentation is contained entirely in the
`docs/dev <https://github.com/openvinotoolkit/openvino/tree/master/docs/dev>`__ folder.


:fas:`bullhorn` Promote and support OpenVINO
############################################

.. rubric:: Popularize OpenVINO
   :name: popularize-openvino

Articles, tutorials, blog posts, demos, videos, and any other involvement in the OpenVINO
community is more than welcome. If you discuss or present OpenVINO on various social platforms,
you are raising awareness of the product among A.I. enthusiasts and enabling other people
to discover the toolkit. Feel free to reach out to OpenVINO developers if you need help
with making a contribution.

.. rubric:: Help other community members
   :name: help-community

If you are an experienced OpenVINO user and want to help, you can share your expertise
with the community at any time. Check GitHub
`Discussions <https://github.com/openvinotoolkit/openvino/discussions>`__ and
`Issues <https://github.com/openvinotoolkit/openvino/issues>`__ to see if you can help someone.

Additional Resources
#####################

- :doc:`Technical guide for contribution <./contributing/technical-guide>`
- Choose a `"Good First Issue" <https://github.com/orgs/openvinotoolkit/projects/3>`__.
- Learn more about `OpenVINO architecture <https://github.com/openvinotoolkit/openvino/blob/master/src/docs/architecture.md>`__.
- Check out a `blog post on contributing to OpenVINO <https://medium.com/openvino-toolkit/how-to-contribute-to-an-ai-open-source-project-c741f48e009e>`__.
- Visit `Intel DevHub Discord server <https://discord.gg/7pVRxUwdWG>`__ to join discussions and talk to OpenVINO developers.

License
#####################

.. note::

   By making a contribution to the OpenVINO project, you agree to the terms of the `LICENSE <https://github.com/openvinotoolkit/openvino/blob/master/LICENSE>`__.