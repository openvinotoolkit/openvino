list_packages_html_template = r"""
<!DOCTYPE html>
<html lang="en">

<head>
    <title>conan list results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet"
        integrity="sha384-rbsA2VBKQhggwzxH7pPCaAqO46MgnOM80zW1RWuH61DGLwZJEdK2Kadq2F9CUG65" crossorigin="anonymous">
    <style>
        body {
            font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 0.75rem;
        }

        .accordion-button {
            padding: 0.25rem 0.25rem;
        }

        .list-group {
            max-height: 100vh;
            overflow: auto;
        }

        .list-group-item {
            padding: 0.2rem 0.35rem;
            border: 0px;
        }
    </style>
    <script>
        var list_results = {{ results| safe }};

        function replaceChars(origin, ref) {
            return origin + "_" + ref.replaceAll(".", "_").replaceAll("/", "_").replaceAll("#", "_").replaceAll("@", "_").replaceAll(":", "_").replaceAll(" ", "_")
        }

        function getPackagesCount(revInfo) {
            if ("packages" in revInfo) {
                return Object.keys(revInfo["packages"]).length;
            }
            return 0;
        }

        function formatDate(timeStamp) {
            var options = {
                year: "numeric",
                month: "2-digit",
                day: "2-digit",
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
                hour12: false
            };
            return new Date(timeStamp * 1000).toLocaleDateString('en', options);
        }

        function getInfoFieldsBadges(info) {
            let style = '';
            let badges = '';
            for (property of ["settings", "options"]) {
                if (property in info) {
                    for (const [key, value] of Object.entries(info[property])) {
                        style = (key == 'os') ? 'text-bg-info' : 'text-bg-secondary';
                        badges += `<span class="badge ${style}">${key}: ${value}</span>&nbsp;`
                    }
                }
            }
            return badges;
        }

        function isSubset(setA, setB) {
            for (let elem of setA) {
                if (!setB.has(elem)) {
                    return false;
                }
            }
            return true;
        }

        function isFiltered(info, filters) {
            if (filters.length == 0) {
                return false;
            }
            packageProperties = [];
            for (property of ["settings", "options"]) {
                if (property in info) {
                    for (const [key, value] of Object.entries(info[property])) {
                        packageProperties.push(`${key}=${value}`);
                    }
                }
            }
            packageSet = new Set(packageProperties);
            filtersSet = new Set(filters);
            if (isSubset(filtersSet, packageSet)) {
                return false;
            }
            return true;
        }

        function getUniqueSettingsOptions(revInfo) {
            let options = new Set();
            let settings = new Set();
            for (const [package, packageInfo] of Object.entries(revInfo["packages"])) {
                let info = packageInfo["info"];
                if ("options" in info) {
                    for (const [key, value] of Object.entries(info["options"])) {
                        options.add(`${key}=${value}`)
                    }
                }
                if ("settings" in info) {
                    for (const [key, value] of Object.entries(info["settings"])) {
                        settings.add(`${key}=${value}`)
                    }
                }
            }
            return [options, settings]
        }

        function getFilters(revID, revInfo) {
            let options_settings = getUniqueSettingsOptions(revInfo);
            let options = Array.from(options_settings[0]);
            let settings = Array.from(options_settings[1]);
            let filter = `<h6>Filter packages:</h6>`
            for (setting of settings) {
                filter += `
                        <div class="form-check form-check-inline">
                            <input class="form-check-input" type="checkbox" id="${revID}#${setting}" value="${setting}">
                            <label class="form-check-label" for="${revID}#${setting}">${setting}</label>
                        </div>
                      `
            }
            for (option of options) {
                filter += `
                        <div class="form-check form-check-inline">
                            <input class="form-check-input" type="checkbox" id="${revID}#${option}" value="${option}">
                            <label class="form-check-label" for="${revID}#${option}">${option}</label>
                        </div>
                      `
            }
            return filter;
        }

        let activeRevision = {};
        let tabsInfo = {};

        function filterPackages() {
            const activeTab = document.querySelector('.tab-pane.active');
            const packageList = activeTab.querySelector('#packageList');
            activeRevision.revision = tabsInfo[activeTab.getAttribute('id')].revision;
            activeRevision.revInfo = tabsInfo[activeTab.getAttribute('id')].revInfo;
            packageList.innerHTML = getPackagesList(activeRevision.revision, activeRevision.revInfo);
        }

        function getPackagesList(revision, revInfo) {
            const checkboxes = document.querySelectorAll('input[type="checkbox"]');
            filters = [];
            checkboxes.forEach(function (checkbox) {
                if (checkbox.checked) {
                    filters.push(checkbox.value)
                }
            })

            activeRevInfo = revInfo;
            let packageList = ``;
            for (const [package, packageInfo] of Object.entries(revInfo["packages"])) {

                if (!isFiltered(packageInfo["info"], filters)) {
                    packageList += `<div class="bg-light">`;
                    packageList += `<h6 class="mb-1">${package}</h6>`;
                    packageList += `${getInfoFieldsBadges(packageInfo["info"])}`;

                    packageList += `<br><br>`;
                    if ("revisions" in packageInfo) {
                        packageList += `<br><br><b>Package revisions:</>`;
                        packageList += `<ul>`;
                        for (const [packageRev, packageRevInfo] of Object.entries(packageInfo["revisions"])) {
                            packageList += `<li>${packageRev}&nbsp(${formatDate(packageRevInfo["timestamp"])})</li>`;
                        }
                        packageList += `</ul>`;
                    }
                    packageList += `</div>`;
                    packageList += `<br>`;
                }
            }
            return packageList;
        }

        function getTabContent(tabID, revID, revInfo) {

            let tabContent = `<div class="tab-pane" id="${tabID}" role="tabpanel">`;

            if ("packages" in revInfo && Object.entries(revInfo["packages"]).length > 0) {

                tabContent += `<h3>Packages for revision ${revID}</h3>`;

                tabContent += getFilters(revID, revInfo);

                tabContent += `<div id="packageList">`;
                tabContent += getPackagesList(revID, revInfo);
                tabContent += `</div>`;

            }
            tabContent += `<h3>JSON</h3>`;
            tabContent += `<pre class="p-3 mb-2 bg-light text-dark">${JSON.stringify(revInfo, null, 2)}</pre>`;
            tabContent += `</div>`
            return tabContent;
        }

        let menu = `<div class="list-group list-group-flush" id="leftTabs" role="tablist">`;
        let tabs = `<div class="tab-content">`;

        for (const [origin, references] of Object.entries(list_results)) {
            if (Object.keys(references).length > 0) {
                menu += `<li class="list-group-item"><b>${origin}</b>`;
                if ("error" in references) {
                    menu += `<pre>${references["error"]}</pre>`;
                }
                else {
                    for (const [reference, revisions] of Object.entries(references)) {
                        let originStr = origin.replaceAll(" ", "_");
                        const refLink = replaceChars(originStr, reference);

                        menu += `<div class="accordion accordion-flush" id="accordion_${originStr}">`;
                        menu += `<div class="accordion-item">`;
                        menu += `<h2 class="accordion-header" id="heading_${refLink}">`;
                        menu += `<button class="accordion-button collapsed" type="button" id="left_${refLink}" data-bs-toggle="collapse" data-bs-target="#rev_list_${refLink}" aria-expanded="false" aria-controls="${refLink}">${reference}</button>`;
                        menu += `</h2>`;
                        menu += `<div id="rev_list_${refLink}" class="accordion-collapse collapse" aria-labelledby="heading_${refLink}" data-bs-parent="#accordion_${originStr}">`

                        if ("revisions" in revisions) {
                            for (const [revision, revInfo] of Object.entries(revisions["revisions"])) {
                                let packageCount = getPackagesCount(revInfo);
                                packageBadge = (packageCount == 0) ? '' : `&nbsp<span class="badge rounded-pill text-bg-success">${packageCount}</span>`;
                                let tabID = `${originStr}_${revision}`;
                                menu += `<a class="list-group-item list-group-item-action" id="left_${revision}" data-bs-toggle="list" href="#${tabID}" role="tab" aria-controls="list-home">${revision.substring(0, 6)}&nbsp(${formatDate(revInfo["timestamp"])})${packageBadge}</a>`;

                                tabsInfo[tabID] = { "revision": revision, "revInfo": revInfo }
                                tabs += getTabContent(tabID, revision, revInfo);
                            }
                        }
                        menu += `</div>`
                        menu += '</div>';
                        menu += '</div>';
                    }

                }

                menu += "</li>";
            }
        }
        menu += "</div>";
        tabs += "</div>";

        document.addEventListener("DOMContentLoaded", function () {
            let leftMenu = document.getElementById("leftmenu");
            let rightMenu = document.getElementById("rightmenu");
            leftMenu.innerHTML = menu;
            rightMenu.innerHTML = tabs;

            var triggerTabList = [].slice.call(document.querySelectorAll('#leftTabs a'))
            triggerTabList.forEach(function (triggerEl) {
                var tabTrigger = new bootstrap.Tab(triggerEl)
                triggerEl.addEventListener('click', function (event) {
                    // remove active from all, so only .list-group-item is selected
                    var listItems = document.querySelectorAll('.list-group-item');
                    for (var i = 0; i < listItems.length; i++) {
                        listItems[i].classList.remove('active');
                    }
                    event.preventDefault()
                    tabTrigger.show()
                })
            })

            const checkboxes = document.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach(function (checkbox) {
                checkbox.addEventListener('change', function () {
                    filterPackages();
                });
            });

        });
    </script>
</head>

<body>
    <nav class="navbar navbar-expand-lg bg-light">
        <div class="container-fluid">
            <a class="navbar-brand">conan list: {{ cli_args }}</a>
        </div>
    </nav>

    <div class="container-fluid">
        <div class="row">
            <div class="col-2" id="leftmenu"></div>
            <div class="col-10" id="rightmenu"></div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"
        integrity="sha384-kenU1KFdBIe4zVF0s0G1M5b4hcpxyD9F7jL+jjXkk+Q2h455rYXK/7HAuoJl+0I4"
        crossorigin="anonymous"></script>
</body>

<footer>
    <div class="text-center p-2" style="background-color: rgba(0, 0, 0, 0.05);">
        Conan <b>{{ version }}</b>
        <script>
            document.write(new Date().getFullYear());
        </script>
        JFrog LTD. <a href="https://conan.io">https://conan.io</a>
    </div>
</footer>

</html>
"""
