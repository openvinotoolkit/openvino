from conan.api.output import ConanOutput, Color, LEVEL_VERBOSE, LEVEL_DEBUG


def print_graph_basic(graph):
    # I am excluding the "download"-"cache" or remote information, that is not
    # the definition of the graph, but some history how it was computed
    # maybe we want to summarize that info after the "GraphBuilder" ends?
    # TODO: Should all of this be printed from a json representation of the graph? (the same json
    #   that would be in the json_formatter for the graph?)
    output = ConanOutput()
    requires = {}
    build_requires = {}
    test_requires = {}
    python_requires = {}
    deprecated = {}
    for node in graph.nodes:
        if hasattr(node.conanfile, "python_requires"):
            for _, r in node.conanfile.python_requires.items():
                python_requires[r.ref] = r.recipe, r.remote
        if node.recipe in ("Consumer", "Cli"):
            continue
        if node.context == "build":
            build_requires[node.ref] = node.recipe, node.remote
        else:
            if node.test:
                test_requires[node.ref] = node.recipe, node.remote
            else:
                requires[node.ref] = node.recipe, node.remote
        if node.conanfile.deprecated:
            deprecated[node.ref] = node.conanfile.deprecated

    output.info("Graph root", Color.BRIGHT_YELLOW)
    path = ": {}".format(graph.root.path) if graph.root.path else ""
    output.info("    {}{}".format(graph.root, path), Color.BRIGHT_CYAN)

    def _format_requires(title, reqs_to_print):
        if not reqs_to_print:
            return
        output.info(title, Color.BRIGHT_YELLOW)
        for ref_, (recipe, remote) in sorted(reqs_to_print.items()):
            if remote is not None:
                recipe = "{} ({})".format(recipe, remote.name)
            output.info("    {} - {}".format(ref_.repr_notime(), recipe), Color.BRIGHT_CYAN)

    _format_requires("Requirements", requires)
    _format_requires("Test requirements", test_requires)
    _format_requires("Build requirements", build_requires)
    _format_requires("Python requires", python_requires)

    def _format_resolved(title, reqs_to_print):
        if not reqs_to_print:
            return
        output.info(title, Color.BRIGHT_YELLOW)
        for k_, v_ in sorted(reqs_to_print.items()):
            output.info("    {}: {}".format(k_, v_), Color.BRIGHT_CYAN)

    if graph.replaced_requires:
        output.info("Replaced requires", Color.BRIGHT_YELLOW)
        for k, v in graph.replaced_requires.items():
            output.info("    {}: {}".format(k, v), Color.BRIGHT_CYAN)

    _format_resolved("Resolved alias", graph.aliased)
    if graph.aliased:
        output.warning("'alias' is a Conan 1.X legacy feature, no longer recommended and "
                       "it might be removed in 3.0.")
        output.warning("Consider using version-ranges instead.")
    _format_resolved("Resolved version ranges", graph.resolved_ranges)
    for req in graph.resolved_ranges:
        if str(req.version) == "[]":
            output.warning("Empty version range usage is discouraged. Use [*] instead",
                           warn_tag="deprecated")
            break

    overrides = graph.overrides()
    if overrides:
        output.info("Overrides", Color.BRIGHT_YELLOW)
        for req, override_info in overrides.serialize().items():
            output.info("    {}: {}".format(req, override_info), Color.BRIGHT_CYAN)

    if deprecated:
        output.info("Deprecated", Color.BRIGHT_YELLOW)
        for d, reason in deprecated.items():
            reason = f": {reason}" if reason else ""
            output.info("    {}{}".format(d, reason), Color.BRIGHT_CYAN)

    if graph.options_conflicts:
        output.info("Options conflicts", Color.BRIGHT_YELLOW)
        for ref, ref_conflicts in graph.options_conflicts.items():
            for option, conflict_info in ref_conflicts.items():
                prev_value = conflict_info['value']
                output.info(f"    {ref}:{option}={prev_value} (current value)", Color.BRIGHT_CYAN)
                for src_ref, conflict_value in conflict_info["conflicts"]:
                    output.info(f"        {src_ref}->{option}={conflict_value}", Color.BRIGHT_CYAN)
        output.info("    It is recommended to define options values in profiles, not in recipes",
                    Color.BRIGHT_CYAN)
        output.warning("There are options conflicts in the dependency graph", warn_tag="risk")

    if deprecated:
        output.warning("There are deprecated packages in the graph", warn_tag="risk")

    if graph.visibility_conflicts:
        msg = ["Packages required both with visible=True and visible=False"]
        for ref, consumers in graph.visibility_conflicts.items():
            msg.append(f"    {ref}: Required by {', '.join(str(c) for c in consumers)}")
        output.warning("\n".join(msg), warn_tag="risk")


def print_graph_packages(graph):
    # I am excluding the "download"-"cache" or remote information, that is not
    # the definition of the graph, but some history how it was computed
    # maybe we want to summarize that info after the "GraphBuilder" ends?
    output = ConanOutput()
    requires = {}
    build_requires = {}
    test_requires = {}
    skipped_requires = []
    tab = "    "
    for node in graph.nodes:
        if node.recipe in ("Consumer", "Cli"):
            continue
        node_info = node.conanfile.info
        if node.context == "build":
            existing = build_requires.setdefault(node.pref,
                                                 [node.binary, node.binary_remote, node_info])
        else:
            if node.test:
                existing = test_requires.setdefault(node.pref,
                                                    [node.binary, node.binary_remote, node_info])
            else:
                existing = requires.setdefault(node.pref,
                                               [node.binary, node.binary_remote, node_info])
        # TO avoid showing as "skip" something that is used in other node of the graph
        if existing[0] == "Skip":
            existing[0] = node.binary

    def _format_requires(title, reqs_to_print):
        if not reqs_to_print:
            return
        output.info(title, Color.BRIGHT_YELLOW)
        for pref, (status, remote, info) in sorted(reqs_to_print.items(), key=repr):
            name = pref.repr_notime() if status != "Platform" else str(pref.ref)
            msg = f"{tab}{name} - "
            if status == "Skip":
                skipped_requires.append(str(pref.ref))
                output.verbose(f"{msg}{status}", Color.BRIGHT_CYAN)
            elif status == "Missing" or status == "Invalid":
                output.write(msg, Color.BRIGHT_CYAN)
                output.writeln(status, Color.BRIGHT_RED)
            elif status == "Build":
                output.write(msg, Color.BRIGHT_CYAN)
                output.writeln(status, Color.BRIGHT_YELLOW)
            else:
                # Support python36
                msg += status
                if remote:
                    msg += f" ({remote.name})"
                output.info(msg, Color.BRIGHT_CYAN)
            if output.level_allowed(LEVEL_DEBUG):
                compact_dumps = info.summarize_compact()
                for line in compact_dumps:
                    output.debug(f"{tab}{tab}{line}", Color.BRIGHT_GREEN)

    _format_requires("Requirements", requires)
    _format_requires("Test requirements", test_requires)
    _format_requires("Build requirements", build_requires)

    if skipped_requires and not output.level_allowed(LEVEL_VERBOSE):
        output.info("Skipped binaries", Color.BRIGHT_YELLOW)
        output.info(f"{tab}{', '.join(skipped_requires)}", Color.BRIGHT_CYAN)
