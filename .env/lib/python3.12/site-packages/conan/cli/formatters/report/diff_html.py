diff_html = r"""
{% macro render_sidebar_folder(folder, folder_info) %}
    {%- for name, sub_folder_info in folder_info["folders"].items() %}
        {% set folder_name = folder + "/" + name %}
        <li>
            <details open class="folder">
                <summary>{{ name }}</summary>
                <ul>
                    {{ render_sidebar_folder(folder_name, sub_folder_info) }}
                </ul>
            </details>
        </li>
    {%- endfor %}
    {%- for name, file_info in folder_info["files"].items() %}
        {% set file_type = "renamed" if file_info["renamed_to"] else (
                           "deleted" if file_info["is_deleted"] else (
                           "new" if file_info["is_new"] else "old")) %}
        <li class="file file-{{ file_type }}"
            data-path="{{ file_info["relative_path"] }}"
            data-type="{{ file_type }}">
            <a href="#diff_{{- safe_filename(file_info["filename"]) -}}"
                onclick="setDataIsLinked(event)" draggable="false"
                class="side-link"
                title="{{ replace_cache_paths(file_info["relative_path"]) | replace("(old)/", "") | replace("(new)/", "") }}">
                {% if file_info["renamed_to"] %}
                    {{ file_info["renamed_to"].split("/")[1:][-1] }}
                {% else %}
                    {{ name }}
                {% endif %}
            </a>
        </li>
    {%- endfor %}
{% endmacro %}

{% macro render_diff_folder(folder_info) %}
    {%- for name, sub_folder_info in folder_info["folders"].items() %}
        {{ render_diff_folder(sub_folder_info) }}
    {%- endfor %}
    {%- for name, file_info in folder_info["files"].items() %}
        {% set filename = file_info["filename"] %}

        <div id="diff_{{ safe_filename(filename) }}" data-path="{{ filename }}" class="diff-container">
            <div class="diff-content">
                <details open class="diff-details">
                    <summary class="diff-summary">
                        <b id="diff_{{ safe_filename(filename) }}_filename" class="filename" data-replaced-paths="">
                            <span>{{ replace_cache_paths(filename) | replace("(old)/", "") | replace("(new)/", "") }}</span>
                            {% if file_info["renamed_to"] %}
                                &nbsp;&#x2192&nbsp;
                                <span>{{ replace_cache_paths(file_info["renamed_to"]) | replace("(old)/", "") | replace("(new)/", "") }}</span>
                            {% endif %}
                        </b>
                        <div class="changes-count-container"></div>
                    </summary>
                    <div class="diff-lines">
                    </div>
                </details>
            </div>
        </div>
    {%- endfor %}
{% endmacro %}
<html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Diff report for {{ old_reference }} - {{ new_reference }}</title>
        <style>
            /* --- Colors --- */
            :root {
                --body-bgColor: #f8f8f8;
                --sidebar-bgColor: #f4f4f466;
                --sidebar-borderColor: #ccc;
                --sidebar-contents-bgColor: #f4f4f4;
                --content-bgColor: #f8f8f8;
                --search-area-borderColor: #ccc;
                --search-field-borderColor: #ccc;
                --file-list-borderColor: #ddd;
                --folder-summary-hover-bgColor: #e0e0e033;
                --folder-ul-hover-borderColor: #00000066;
                --sidebar-li-a-hover-bgColor: #e0e0e0;
                --sidebar-button-hover-bgColor: var(--sidebar-li-a-hover-bgColor);
                --sidebar-link-color: black;
                --sidebar-link-hover-color: var(--sidebar-link-color);
                --sidebar-link-visited-color: var(--sidebar-link-color);
                --sidebar-file-new-color: green;
                --sidebar-file-old-color: gray;
                --sidebar-file-deleted-color: red;
                --diff-content-borderColor: black;
                --diff-content-bgColor: white;
                --diff-container-linked-borderColor: #0078d7;
                --diff-summary-borderColor: #ccc;
                --diff-summary-bgColor: #f8f8f8;
                --diff-summary-hover-bgColor: #f0f0f0;
                --new-lines-count-color: green;
                --old-lines-count-color: black;
                --context-line-color: #888;
                --context-chunk-header-bgColor: #cef8ff;
                --context-chunk-header-color: var(--context-line-color);
                --added-line-bgColor: #cbfcd9;
                --added-line-color: black;
                --deleted-line-bgColor: #ffebe9;
                --deleted-line-color: black;
                --line-number-added-bgColor: #76ffbb;
                --line-number-deleted-bgColor: #fdb9c1;
            }

            /* --- Global Styles --- */

            body {
                font-family: monospace;
                margin: 0px;
                background-color: var(--body-bgColor);
            }

            /* --- Main Layout --- */

            .container {
                display: flex;
                height: 100%;
                overflow: scroll;
            }

            .sidebar {
                width: 17%;
                min-width: 10%;
                max-width: 33%;
                padding: 10px;
                overflow: scroll;
                background: var(--sidebar-bgColor);
                border-right: 1px solid var(--sidebar-borderColor);
                resize: horizontal;
                position: sticky;
                top: 0;
            }

            .content {
                padding: 20px;
                background: var(--content-bgColor);
                width: 100%;
            }

            /* --- Sidebar & File Tree --- */

            #sidebar-contents {
                background: var(--sidebar-contents-bgColor);
                border-radius: 7px;
                overflow-y: hidden;
                padding-top: 5px;
            }

            .sidebar-reveal {
                display: none;
                position: sticky;
                top: 10px;
            }

            .search-area {
                border-bottom: 1px solid var(--search-area-borderColor);
            }

            .search-header {
                display: flex;
                justify-content: space-between;
            }

            .search-field {
                border: 1px solid var(--search-field-borderColor);
                border-radius: 5px;
                padding: 5px;
                margin: 5px;
                width: 80%;
            }

            .file-tree-controls {
                border-bottom: 1px solid var(--search-area-borderColor);
                padding: 5px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .file-tree-controls .folder-collapse button {
                display: inline-block;
                line-height: 0.7;
            }

            .file-tree-controls button,
            .sidebar-reveal button,
            .search-header button {
                cursor: pointer;
                border: 0px solid var(--search-field-borderColor);
                border-radius: 5px;
                background: none;
                padding: 5px;
                min-width: 3ch;
            }

            .file-tree-controls button:hover,
            .sidebar-reveal button:hover,
            .search-header button:hover {
                background-color: var(--sidebar-li-a-hover-bgColor);
            }

            .file-tree-more {
                display: none;
                padding: 5px;
                border-bottom: 1px solid var(--search-area-borderColor);
            }

            .file-tree-more-option {
                display: block;
            }

            .file-list {
                padding-left: 10px;
                width: 100%;
                overflow-x: clip;
            }

            .file-list ul li {
                width: 100%;
            }

            .file-list li ul {
                border-left: 1px solid var(--file-list-borderColor);
                margin-left: 3px;
            }

            li ul {
                padding-left: 1ch;
            }

            details.folder {
                text-wrap: nowrap;
            }

            .folder > summary {
                cursor: pointer;
                list-style: none;
            }

            .folder > summary:hover {
                background-color: var(--folder-summary-hover-bgColor);
            }

            .folder:not(:open) > summary:before {
                content: "\1F4C1";
                display: inline-block;
                margin-right: 3px;
            }

            .folder:open > summary:before {
                content: "\1F4C2";
                display: inline-block;
                margin-right: 3px;
            }

            details.folder ul:hover {
                border-left: 1px solid var(--folder-ul-hover-borderColor);
            }

            .sidebar li {
                line-height: 1.8;
                list-style: none;
                list-style-position: inside;
                user-select: none;
            }

            .sidebar li a {
                text-decoration: none;
                padding: 5px;
                color: var(--sidebar-link-color);
            }

            .sidebar li a:hover {
                text-decoration: none;
                border-radius: 5px;
                background-color: var(--sidebar-li-a-hover-bgColor);
                padding: 5px;
                color: var(--sidebar-link-hover-color);
            }

            .sidebar li a:visited {
                color: var(--sidebar-link-visited-color);
            }

            .side-link {
                text-wrap: nowrap;
            }

            /* File Status Indicators */
            .sidebar li.file-new,
            .sidebar li.file-old,
            .sidebar li.file-deleted,
            .sidebar li.file-renamed {
                list-style: none;
                padding-left: 0;
            }

            .sidebar li.file-new:before {
                content: "+";
                color: var(--sidebar-file-new-color);
                font-weight: bold;
            }

            .sidebar li.file-old:before {
                content: "\00B1";
                color: var(--sidebar-file-old-color);
            }

            .sidebar li.file-deleted:before {
                content: "-";
                color: var(--sidebar-file-deleted-color);
                font-weight: bold;
            }

            .sidebar li.file-renamed:before {
                content: "\2192";
                color: var(--sidebar-file-old-color);
                font-weight: bold;
            }

            /* --- Diff View Components --- */

            .diff-container {
                scroll-margin-top: 10px;
            }

            .diff-content {
                padding-bottom: 7px;
                border: 1px solid var(--diff-content-borderColor);
                border-radius: 7px;
                margin-bottom: 10px;
                background-color: var(--diff-content-bgColor);
            }

            .diff-container[data-is-linked="true"] .diff-content {
                border: 2px solid var(--diff-container-linked-borderColor);
            }

            details.diff-details summary.diff-summary {
                cursor: pointer;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid var(--diff-summary-borderColor);
                padding: 5px 0px;
                position: sticky;
                top: 0;
                background-color: var(--diff-summary-bgColor);
                border-radius: 7px 7px 0px 0px;
            }

            details.diff-details summary.diff-summary:hover {
                background-color: var(--diff-summary-hover-bgColor);
            }

            details:open .diff-summary .filename:before {
                content: "\25BC";
                display: inline-block;
            }

            details:not(:open) .diff-summary .filename:before {
                content: "\25B6";
                display: inline-block;
            }

            .diff-header {
                padding: 0px 5px 5px 5px;
            }

            .filename {
                font-size: 1.2em;
                padding-left: 10px;
            }

            .changes-count-container {
                font-size: 0.9em;
                padding-right: 10px;
            }

            .new-lines-count {
                color: var(--new-lines-count-color);
                font-weight: bold;
            }

            .old-lines-count {
                color: var(--old-lines-count-color);
                font-weight: bold;
            }

            /* --- Diff Line Styles --- */

            .content span {
                white-space: pre-wrap;
            }

            .context-chunk-header {
                list-style: none;
                background-color: var(--context-chunk-header-bgColor);
                color: var(--context-chunk-header-color);
                line-height: 2;
                cursor: pointer;
            }

            details:open .context-chunk-header .line-number:before {
                content: "\25BC";
                display: inline-block;
            }

            details:not(:open) .context-chunk-header .line-number:before {
                content: "\25B6";
                display: inline-block;
            }

            .diff-lines {
                line-break: anywhere;
            }

            .line-number {
                width: 4ch;
                min-width: 4ch;
                display: inline-block;
                text-align: center;
                user-select: none;
            }

            .context-line {
                color: var(--context-line-color);
            }

            .add {
                background-color: var(--added-line-bgColor);
                color: var(--added-line-color);
            }

            .del {
                background-color: var(--deleted-line-bgColor);
                color: var(--deleted-line-color);
            }

            .add,
            .del,
            .context-line {
                height: 100%;
            }

            .diff-line {
                display: flex;
                box-sizing: border-box;
                line-height: 1.5em;
            }

            .line-number.add {
                background-color: var(--line-number-added-bgColor);
            }

            .line-number.del {
                background-color: var(--line-number-deleted-bgColor);
            }

            .line-number.add,
            .line-number.del {
                height: auto;
            }

            .diff-symbol {
                display: inline-block;
                width: 1ch;
                user-select: none;
            }

            /* --- Utility & Page States --- */

            #empty_result {
                justify-content: center;
                align-items: center;
                color: black;
                font-weight: bold;
                font-size: 4em;
                text-align: center;
            }
        </style>
        <script>

            const data = {{ content | tojson | safe }};

            const oldPattern = "{{ src_prefix[:-1] }}{{ old_cache_path }}";
            const newPattern = "{{ dst_prefix[:-1] }}{{ new_cache_path }}";

            function extractLineNumbers(hunkHeader) {
                const regex = /@@ -(\d+),\d+ \+(\d+),\d+ @@/;
                const match = hunkHeader.match(regex);
                if (!match) {
                    return [0, 0];
                }
                return [parseInt(match[1]), parseInt(match[2])];
            }


            function makeDiffLines(lines) {
                const element = document.createElement("div");
                let seen_header = false;
                let new_line_index = 0;
                let old_line_index = 0;
                let new_line_count = 0;
                let old_line_count = 0;
                const headerDiv = document.createElement("div");
                let currentDetails = null;
                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i];
                    let spanLine = document.createElement("span");
                    const lineDiv = document.createElement("div");
                    lineDiv.className = "diff-line";
                    let shouldAddLine = true;
                    if (line.startsWith("+++")) {
                        seen_header = true;
                        spanLine.className = "add";
                        spanLine.textContent = line.replace(newPattern, "(new)");
                        headerDiv.appendChild(spanLine);
                        continue;
                    } else if (line.startsWith("---")) {
                        spanLine.className = "del";
                        spanLine.textContent = line.replace(oldPattern, "(old)");
                        headerDiv.appendChild(spanLine);
                        continue;
                    } else if (line.startsWith("@@")) {
                        currentDetails = document.createElement("details");
                        currentDetails.open = true;

                        const summary = document.createElement("summary");
                        summary.className = "context-chunk-header";
                        const summaryArrow = document.createElement("span");
                        summaryArrow.className = "line-number";
                        const summaryText = document.createElement("span");
                        summaryText.textContent = line;

                        summary.appendChild(summaryArrow);
                        summary.appendChild(summaryText);

                        currentDetails.appendChild(summary);
                        element.appendChild(currentDetails);
                        shouldAddLine = false;

                        const lineNumbers = extractLineNumbers(line);
                        old_line_index = lineNumbers[0];
                        new_line_index = lineNumbers[1];
                    } else if (line.startsWith("+")) {
                        const spanSymbol = document.createElement("span");
                        spanSymbol.textContent = "+";
                        spanSymbol.className = "diff-symbol";
                        spanLine.className = "add";
                        spanLine.textContent = line.substring(1);
                        spanLine.prepend(spanSymbol);

                        const lineNumberSpan = document.createElement("span");
                        lineNumberSpan.className = "line-number add";
                        lineNumberSpan.textContent = new_line_index;
                        lineDiv.appendChild(lineNumberSpan);

                        new_line_index += 1;
                        new_line_count += 1;
                    } else if (line.startsWith("-")) {
                        const spanSymbol = document.createElement("span");
                        spanSymbol.textContent = "-";
                        spanSymbol.className = "diff-symbol";
                        spanLine.className = "del";
                        spanLine.textContent = line.substring(1);
                        spanLine.prepend(spanSymbol);

                        const lineNumberSpan = document.createElement("span");
                        lineNumberSpan.className = "line-number del";
                        lineNumberSpan.textContent = old_line_index;
                        lineDiv.appendChild(lineNumberSpan);

                        old_line_index += 1;
                        old_line_count += 1;
                    } else {
                        spanLine.className = "context-line";
                        if (!seen_header) {
                            spanLine.textContent = line.replace(oldPattern, "(old)").replace(newPattern, "(new)");
                            headerDiv.appendChild(spanLine);
                            headerDiv.appendChild(document.createElement("br"));
                            continue;
                        } else {
                            const spanSymbol = document.createElement("span");
                            spanSymbol.className = "diff-symbol";
                            spanLine.textContent = line;
                            spanLine.prepend(spanSymbol);
                        }

                        const lineNumberSpan = document.createElement("span");
                        lineNumberSpan.className = "line-number context-line";
                        lineNumberSpan.textContent = new_line_index;
                        lineDiv.appendChild(lineNumberSpan);

                        new_line_index += 1;
                        old_line_index += 1;
                    }
                    if (shouldAddLine) {
                        lineDiv.appendChild(spanLine);

                        currentDetails.appendChild(lineDiv);
                        //currentDetails.appendChild(document.createElement("br"));
                    }
                }
                if (!seen_header) {
                    element.appendChild(headerDiv);
                }
                return [element, new_line_count, old_line_count];
            }

            function createChangesCountElement(new_count, old_count) {
                const changes = document.createElement("span");
                changes.className = "changes-count";
                changes.innerHTML = `<span class="new-lines-count">+${new_count}</span> <span class="old-lines-count">-${old_count}</span>`;
                return changes;
            }


            function intersectionCallback(entries) {
              entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    let elem = entry.target;
                    const path = elem.dataset.path;
                    const [lines, new_count, old_count] = makeDiffLines(data[path]);
                    const diffLines = elem.querySelector(".diff-lines")

                    // If we're scrolling up, new lines are added to the top, so we need to
                    // preserve the scroll position relative to the bottom of the new content
                    const prevRect = elem.getBoundingClientRect();


                    diffLines.appendChild(lines);

                    if (new_count !== 0 || old_count !== 0) {
                        elem.querySelector(".changes-count-container").appendChild(createChangesCountElement(new_count, old_count));
                    }

                    if (elem.getAttribute("data-is-linked") === "true") {
                        // We need to scroll to the element again now that its height has changed
                        elem.scrollIntoView({block: "start", inline: "nearest", behavior: "instant"});
                    } else {
                        if (prevRect.top < 0) {
                            const prevBottom = prevRect.bottom;
                            const newBottom = elem.getBoundingClientRect().bottom;
                            const container = document.querySelector('.container');
                            container.scroll(0, container.scrollTop + (newBottom - prevBottom));
                        }
                    }

                    observer.unobserve(elem);
                }
              });
            }

            const options = {
                root: document.querySelector('.content'),
                rootMargin: "0px",
                scrollMargin: "0px",
                threshold: 0.05,
            };

            const observer = new IntersectionObserver(intersectionCallback, options);

            document.addEventListener("DOMContentLoaded", (e) => {
                setDataIsLinked(null);
                document.querySelectorAll('.diff-container').forEach((section) => {
                    observer.observe(section);
                });
            });

            function debounce(func, delay) {
                let timeout;
                return function(...args) {
                    const context = this;
                    clearTimeout(timeout);
                    timeout = setTimeout(() => {
                        func.apply(context, args);
                    }, delay);
                };
            }
            let includeSearchQuery = "";
            let excludeSearchQuery = "";

            async function onSearchInput(event) {
                const sidebar = document.querySelectorAll(".sidebar li");
                const fileList = document.querySelector(".file-list");
                const content = document.querySelectorAll(".content .diff-container .diff-content");
                const searchingIcon = document.getElementById("searching_icon");

                searchingIcon.style.display = "inline-block";

                let emptySearch = true;
                let includedFiles = 0;

                const typeVisibility = {
                    "renamed": document.getElementById("show-moved-files").checked,
                    "deleted": document.getElementById("show-deleted-files").checked,
                    "new": document.getElementById("show-new-files").checked,
                    "old": document.getElementById("show-old-files").checked,
                };

                sidebar.forEach(async function(item) {
                    if (item.dataset.path === undefined) {
                        // A folder, those are handled later
                        return;
                    }
                    const text = item.dataset.path.toLowerCase();
                    const shouldInclude = includeSearchQuery === "" || text.includes(includeSearchQuery);
                    let shouldExclude = excludeSearchQuery !== "" && text.includes(excludeSearchQuery);
                    const associatedId = item.querySelector("a").getAttribute("href").substring(1)
                    const contentItem = document.getElementById(associatedId);

                    const fileType = item.dataset.type;
                    const isTypeVisible = typeVisibility[fileType] !== false;

                    shouldExclude = shouldExclude || !isTypeVisible;

                    if (shouldInclude) {
                        if (shouldExclude) {
                            item.style.display = "none";
                            contentItem.style.display = "none";
                        } else {
                            includedFiles += 1;
                            item.style.display = "list-item";
                            contentItem.style.display = "block";
                            emptySearch = false;
                        }
                    } else {
                        item.style.display = "none";
                        contentItem.style.display = "none";
                    }

                });

                searchingIcon.style.display = "none";
                const emptySearchTag = document.getElementById("empty_search");
                const emptyResultTag = document.getElementById("empty_result");
                if (emptySearch) {
                    emptySearchTag.style.display = "block";
                    emptyResultTag.style.display = "block";
                    fileList.style.display = "none";
                } else {
                    emptySearchTag.style.display = "none";
                    emptyResultTag.style.display = "none";
                    fileList.style.display = "block";
                }

                const fileCountTag = document.getElementById("file-count");
                fileCountTag.textContent = includedFiles;

                const allDetails = document.querySelectorAll(".sidebar details.folder");
                allDetails.forEach(function(details) {
                    details.style.display = "none";
                    details.querySelectorAll("li.file").forEach(function(li) {
                        if (li.style.display !== "none") {
                            details.style.display = "block";
                            return;
                        }
                    });
                });

            }

            const debouncedOnSearchInput = debounce(onSearchInput, 300);

            async function onExcludeSearchInput(event) {
                excludeSearchQuery = event.currentTarget.value.toLowerCase();
                debouncedOnSearchInput(event);
            }

            async function onIncludeSearchInput(event) {
                includeSearchQuery = event.currentTarget.value.toLowerCase();
                debouncedOnSearchInput(event);
            }

            function setDataIsLinked(event) {
                const hash = event ? event.currentTarget.getAttribute("href").substring(1) : window.location.hash.substring(1);
                document.querySelectorAll('.diff-container').forEach((section) => {
                    if (section.id === hash) {
                        section.setAttribute("data-is-linked", "true");
                        if (!event) {
                            // Scroll to the linked element on page load
                            section.scrollIntoView({block: "start", inline: "nearest", behavior: "instant"});
                        }
                    } else {
                        section.setAttribute("data-is-linked", "false");
                    }
                });
            }

            function toggleFolders(open) {
                if (open) {
                    const toOpen = document.querySelectorAll('details.folder:open > ul > li > details.folder:not(:open)');
                    if (toOpen.length === 0) {
                        // We might need to open the root folders
                        document.querySelectorAll('.file-list > li > details.folder:not(:open)').forEach(d => d.open = true);
                    } else {
                        toOpen.forEach(d => d.open = true);
                    }
                } else {
                    document.querySelectorAll('details.folder:open').forEach(d => d.open = false);
                }
            }

            function toggleSidebar(show) {
                const sidebar = document.querySelector('.sidebar');
                const sidebarReveal = document.querySelector('.sidebar-reveal');
                const content = document.querySelector('.content');
                if (show) {
                    sidebar.style.display = 'block';
                    sidebarReveal.style.display = 'none';
                    content.style.padding = '20px';
                } else {
                    sidebar.style.display = 'none';
                    sidebarReveal.style.display = 'block';
                    content.style.padding = '20px 20px 20px 5px';
                }
            }

            function toggleMoreFileTree() {
                const moreOptions = document.querySelector('.file-tree-more');
                console.log(moreOptions.style.display);
                const show = moreOptions.style.display !== 'block';
                if (show) {
                    moreOptions.style.display = 'block';
                } else {
                    moreOptions.style.display = 'none';
                }
            }
        </script>
    </head>
    <body>
        <div class='container'>
            <div class='sidebar'>
                <div id="sidebar-contents">
                    <div class="search-area">
                        <div class="search-header">
                            <div>
                                <input type="search" class="search-field" id="search-include" placeholder="Include search..." oninput="onIncludeSearchInput(event)" />
                                <input type="search" class="search-field" id="search-exclude" placeholder="Exclude search..." oninput="onExcludeSearchInput(event)" />
                                <span id="searching_icon" style="display:none">...</span>
                            </div>

                            <button onclick="toggleSidebar(false)" title="Hide">
                                &#x2190;
                            </button>
                        </div>
                        <p>Showing <b id="file-count">{{ content|length }}</b> out of <b>{{ content|length }}</b> files</p>
                    </div>
                    <div class="file-tree">
                        <div class="file-tree-controls">
                            <div class="folder-collapse">
                                <button onclick="toggleFolders(true)" title="Expand current level">
                                    &#x02C4;
                                    <br/>
                                    &#x02C5;
                                </button>
                                <button onclick="toggleFolders(false)" title="Collapse all">
                                    &#x02C5;
                                    <br/>
                                    &#x02C4
                                </button>
                            </div>
                            <button onclick="toggleMoreFileTree()" title="Show more options"
                                class="file-tree-reveal-more">
                                    &#x22EE;
                            </button>
                        </div>
                        <div class="file-tree-more">
                            <h4>Show...</h4>
                            <div class="file-tree-more-option">
                                <input type="checkbox" id="show-old-files" checked
                                    onclick="debouncedOnSearchInput(event)"/>
                                <label for="show-old-files">Old files</label>
                            </div>

                            <div class="file-tree-more-option">
                                <input type="checkbox" id="show-new-files" checked
                                    onclick="debouncedOnSearchInput(event)"/>
                                <label for="show-new-files">New files</label>
                            </div>

                            <div class="file-tree-more-option">
                                <input type="checkbox" id="show-deleted-files" checked
                                    onclick="debouncedOnSearchInput(event)"/>
                                <label for="show-deleted-files">Deleted files</label>
                            </div>

                            <div class="file-tree-more-option">
                                <input type="checkbox" id="show-moved-files" checked
                                    onclick="debouncedOnSearchInput(event)"/>
                                <label for="show-moved-files">Moved files</label>
                            </div>
                        </div>
                        <ul class="file-list">
                            {{ render_sidebar_folder("", per_folder) }}
                        </ul>
                    </div>
                </div>
                <span id="empty_search" style="display:none">No results found</span>
            </div>
            <div class='sidebar-reveal'>
                <button onclick="toggleSidebar(true)" title="Show">
                    &#x2192;
                </button>
            </div>
            <div class='content'>
                <div class="diff-header">
                    <h2>Diff Report Between <b class="del">{{ old_reference.repr_notime() }}</b> And <b class="add">{{ new_reference.repr_notime() }}</b></h2>
                </div>
                <span id="empty_result" style="display:none">No matches</span>
                {{ render_diff_folder(per_folder) }}
            </div>
        </div>
    </body>
</html>
"""
