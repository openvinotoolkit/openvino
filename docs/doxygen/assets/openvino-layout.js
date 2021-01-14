/*
******************************************************************************
Copyright 2017-2021 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
******************************************************************************
*/

"use strict";

/**
 * Builds menus dynamically and displays proper side navigation based on current page URL.
 * Document overview is on the left of the page, in-page navigation is on the right
 * (Same nav shows up embedded at top of page on smaller breakpoints)
 * @author Amplified - http://amplifiedbydesign.com/
 * @returns {function}
 */
function openVinoMenu() {
    // Get active page path as "currentPath"
    var currentPath = getCurrentPath();
    var activeMenu = "";

    /**
     * Initialize/do this on page load
     * @public
     */
    function init() {
        var mainNav = document.getElementById("main-nav");

        if (mainNav) {
            // Remove first (unused "home" link) and last (search box) list items that are added
            // automatically by Doxygen script
            $("ul#main-menu li").first().remove();
            $("ul#main-menu li").last().remove();

            var $initialMenuContent = $("#main-nav #main-menu > li");
            // Have to make a copy of the menu because the first one is being so heavily manipulated
            var $secondaryMenuContent = $("#main-nav #main-menu").clone();

            // Remove .sub-arrow span from all menu content
            $initialMenuContent.find("span.sub-arrow").remove();
            $secondaryMenuContent.find("span.sub-arrow").remove();
            var $mainMenuItems = $(
                'a[href ="' + currentPath + '"]',
                $initialMenuContent
            )
                .parents("ul > li")
                .last();
            
            var $leftMenuContent = $($mainMenuItems).clone();

            var $activeMenuItem = $(
                'a[href ="' + currentPath + '"]',
                $secondaryMenuContent
            )
                .parents("ul > li")
                .last();
            var $activeMenu = $activeMenuItem.find("ul").first() || null;

            activeMenu = $mainMenuItems
                .find("a")
                .first()
                .text();

            // if current path is search.html no menu is renderedered
            // Else if "nav-path" exists OR activeMenu returns an empty string,
            //  then assume we're dealing with a generated API page and manually
            //  set the activeMenu/$activeMenu variables.
            // If the API REFERENCES category name changes, update this with correct name!
            if (currentPath.indexOf('g-search.html') != -1) {
                activeMenu = '';
                $activeMenu = null;
            }
            else if ($("#top #nav-path").length > 0 || activeMenu === "") {
                activeMenu = "API REFERENCES";
                $activeMenu = $($secondaryMenuContent)
                                .find('a[href^="' + activeMenu + '"]')
                                .last()
                                .addClass("active")
                                .parent("li")
                                .addClass("active");
            }
            
            renderMainMenu($initialMenuContent);
            renderLeftMenu($leftMenuContent);
            renderContentsMenus();

            $("#container").css({
                marginTop: document.getElementById("top").offsetHeight + 40
            });
        }
    }

    /**
     * Moves window scroll position to element, needed to handle clicks on both
     * in-page links and if you copy/paste a link from elsewhere
     * @public
     * @param {DOM Element} el
     */
    function scrollToElement(el) {
        var headerHeight = document.getElementById("top").offsetHeight;
        // Handle the in-page links we created elsewhere
        if (["H2", "H3", "H4"].includes(el.parentElement.nodeName)) {
            el = el.parentElement;
        } else {
            el = el.nextElementSibling;
        }

        // uses jQuery's offset method in order to get around issue with headings found in tables
        var offset = $(el).offset();
        var scrollValue = offset.top - headerHeight - 30;

        window.scrollTo(0, scrollValue);
    }

    /**
     * Makes click function accessible from outside openVinoMenu function
     * @public
     * @param {String} href
     * @returns function
     */
    function handleScrollToClick(href) {
        return function (event) {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();

            var anchor = document.getElementById(href);

            if (window.location.hash) {
                window.location.href = window.location.href.replace(
                    window.location.hash,
                    "#" + href
                );
            } else {
                window.location.href = window.location.href + "#" + href;
            }

            scrollToElement(anchor);
        };
    }

    /**
     * Builds out the "contents" menu, based on H2 and H3 DOM Elements found
     * within the current document. Returns markup.
     * @private
     */
    function makeContentsMenu() {
        function makeTree(data, className) {
            var result = "";
            if (data.length > 0) {
                if (className) {
                    result += '<ul class="' + className + '">';
                } else {
                    result += "<ul>";
                }
                // Create starting list item as either list-with-children or a straight list item
                for (var i = 0, dataLength = data.length; i < dataLength; i++) {
                    if (data[i].nodeName === "H2" && data[i].children.length > 0) {
                        result += '<li class="has-children">';
                    } else {
                        result += "<li>";
                    }
                    result +=
                        '<a href="' +
                        data[i].url +
                        '">' +
                        data[i].text +
                        "</a>" +
                        makeTree(data[i].children) +
                        "</li>";
                }
                result += "</ul>";
            }
            return result;
        }

        /**
         Create the list of links
         **/
        function generateList(arr) {
            if (arr && arr.length > 0) {
                var contentsArr = [];
                for (var i = 0, arrLength = arr.length; i < arrLength; i++) {
                    var obj = {
                        nodeName: arr[i].nodeName,
                        text: arr[i].innerText.replace(/[◆§]/gi, ""),
                        children: []
                    };
                    if (arr[i].querySelector("a.anchor")) {
                        obj.url = "#" + arr[i].querySelector("a.anchor").id;
                    } else if (arr[i].querySelector("span.permalink a")) {
                        obj.url = arr[i]
                            .querySelector("span.permalink a")
                            .getAttribute("href");
                    }
                    // handle in-page h2 and h3 links
                    if (
                        contentsArr.length > 0 &&
                        (arr[i].nodeName === "H3" ||
                            (arr[i].nodeName === "H2" &&
                                arr[i].classList.contains("memtitle")))
                    ) {
                        contentsArr[contentsArr.length - 1].children.push(obj);
                    }
                    if (
                        arr[i].classList &&
                        arr[i].nodeName === "H2" &&
                        !arr[i].classList.contains("memtitle")
                    ) {
                        contentsArr.push(obj);
                    }
                }

                return makeTree(contentsArr, "contents-list");
            }
        }

        // Create "anchors" on each heading we are navigating to (if they do not exist)
        $("h2, h3", ".contents").each(function () {
            if ($("a.anchor, span.permalink a", this).length > 0) return;
            var urlHash = this.innerText
                .replace(/[`~!@#$%^&*()|+=?;:'",.<>®™◆•§\{\}\[\]\\\/]/gi, "")
                .replace(/[\s/-]/g, "_")
                .toLowerCase();
            var anchor = $('<a class="anchor"></a>').attr("id", urlHash);
            $(this).prepend(anchor);
        });

        // Create/Format list of heading anchors
        return generateList($("h2, h3", ".contents"));
    }

    /**
     * Handle all scrolling effects relating to LEFT NAV
     * @param {Object} menu
     */
    function handleScrolling(menu) {
        var headerHeight = document.getElementById("top").offsetHeight;
        var footerOffset = document.querySelector(".footer").getBoundingClientRect()
            .top;
        var windowHeight = window.innerHeight;
        var maxHeight =
            windowHeight < footerOffset
                ? windowHeight - headerHeight - 60
                : footerOffset - headerHeight - 60;

        menu.css({
            top: headerHeight + 30,
            "max-height": maxHeight
        });

        if (menu[0].id === "contents-nav") {
            $("ul a", menu).each(function () {
                var $this = $(this);
                var href = $this.attr("href");
                if (href[0] !== "#") {
                    var regEx = new RegExp(currentPath + "\\#\\w+", "ig");
                    if (href.match(regEx)) {
                        href = href.replace(href, href.match(/\#\w+/gi)[0]).substring(1);
                    }
                } else {
                    href = href.substring(1);
                }
                var anchor = document.getElementById(href);

                // Handle scrolling for in-page links
                if (anchor) {
                    if (["H2", "H3"].includes(anchor.parentElement.nodeName)) {
                        anchor = anchor.parentElement;
                    } else {
                        anchor = anchor.nextElementSibling;
                    }

                    if (
                        anchor.getBoundingClientRect().top < headerHeight + 40 &&
                        $("ul a.active", menu) !== $this
                    ) {
                        $("ul a", menu).removeClass("active");
                        $this.addClass("active");
                        $this.parent("ul.contents-menu > li > ul").show();
                    }

                    $this.on("click", handleScrollToClick(href));
                }
            });
        }
    }

    /**
     ** Builds main menu based on the array provided to it when called
     * @private
     * @param {Object} menu
     */
    function renderMainMenu(menu) {
        $.each(menu, function () {
            var $link = $(this).children('a').first();
            var $submenu = $(this).children('ul').first();
            if ($submenu) {
                $submenu.addClass('dropdown-menu');
            }
            $('ul ul', $submenu).remove();

            $('> li', $submenu).addClass('col-lg-2 col-md-auto');

            $('ul', $submenu).each(function() {
                var sectionItems = $('> li', this);
                if (sectionItems.length > 5) {
                    sectionItems.slice(6).hide();
                    var sectionUrl = $('> a', $(this).parent('li')).last().attr('href');
                    var $seeAll = $('<li><a class="see-all" href="' + sectionUrl + '">See all ></a></li>');
                    $(this).append($seeAll);
                }
            });
            if ($link.text() === activeMenu) {
                $link.addClass("active");
            }
            return this;
        });

        $("#main-nav").insertAfter($('#projectalign'));
    }

    /**
     * Builds entire left menu
     * @private
     * @param {Object} leftMenuContent
     */
    function renderLeftMenu(leftMenuContent) {
        // Handle all accordion logic and functionality
        function handleAccordion() {
            var $target = $(this);
            var $parent = $target.parent();
            if ($parent.hasClass('accordion-heading')) {
                $target = $parent;
            }
            var $list = $target.siblings("ul");

            $target.parent("li").toggleClass("accordion-opened");
            $list.toggle();
        }

        // Set up LEFT NAV
        if (leftMenuContent) {
            var $leftNav = $('<nav id="left-nav"></nav>');
            var $menu = $('<ul class="main-menu"></ul>');
            $menu.append(leftMenuContent);
            var $listItems = $("li", $menu);
            var $listItemsWithChildren = $listItems.filter(function () {
                return $("ul", this).length > 0;
            });

            $listItemsWithChildren.each(function() {
                $(this).addClass("has-children accordion");
                var $heading = $('<div class="accordion-heading"></div>');
                $('> a', this).wrap($heading);
                $('> div.accordion-heading', this).prepend('<span class="accordion-trigger"></span>')
            });

            $menu
                .addClass("main-menu")
                .removeAttr(
                    "aria-hidden aria-labelledby aria-expanded aria-haspopup aria-controls"
                );

            $leftNav.append($menu);
        }

        // Set 'active' class on currently selected nav link
        $leftNav
            .find('a[href^="' + currentPath + '"]')
            .last()
            .closest("li")
            .addClass("active");

        // Add LEFT NAV to page
        if ($("a", $leftNav).length > 0) {
            $leftNav.insertBefore($(".contents"));
            var activeListItem = $("li.active", $leftNav)[0];
            // Show/Expand active list accordion
            var activeListItemStack = $(activeListItem).parents(
                "ul.main-menu li"
            );
            activeListItemStack.push(activeListItem);
            
            // get 2nd lvl node
            var secondLvlItems = $('> ul > li > ul > li.has-children.accordion', $leftNav)
            
            if (secondLvlItems) {
                activeListItemStack.push.apply(activeListItemStack, secondLvlItems);
            }

            if (activeListItemStack.length > 0 && activeListItem) {
                activeListItemStack.each(function () {
                    if ($(this).hasClass("accordion")) {
                        $(this).addClass("accordion-opened");
                        $("> ul", this).show();
                    }
                });

                // Scroll active list item to top of menu on page load
                $leftNav[0].scrollTop = $(activeListItem).offset().top - $($leftNav).offset().top - 60;
            }

            $("ul.main-menu li.has-children span.accordion-trigger", $leftNav).on(
                "click",
                handleAccordion
            );

            // Add event listeners to this navigation so we can detect scroll and resize when necessary
            window.addEventListener(
                "scroll",
                function () {
                    handleScrolling($leftNav);
                },
                false
            );
            window.addEventListener(
                "resize",
                function () {
                    handleScrolling($leftNav);
                },
                false
            );
            handleScrolling($leftNav);
        }
    }

    /**
     * Builds entire right/document in-page contents menu
     */
    function renderContentsMenus() {
        var $contentsNav = $('<nav id="contents-nav" class="contents-nav"></nav>');

        $contentsNav.append(
            $('<h2 class="contents-nav-title">In This Document</h2>')
        );
        $contentsNav.append(makeContentsMenu());

        var $innerContentsNav = $contentsNav.clone();
        $innerContentsNav[0].id = "inner-contents-nav";

        // add RIGHT NAV to page twice, once for the right column and a second time
        // to the top of the page so we can show it inline on smaller breakpoints
        if ($("a", $contentsNav).length > 0) {
            $contentsNav.insertAfter("#left-nav");
            $innerContentsNav.insertAfter($(".contents .headertitle"));

            // Add event handlers to in-page navigation to we can detect scroll and resize when necessary
            window.addEventListener(
                "scroll",
                function () {
                    handleScrolling($contentsNav);
                },
                false
            );
            window.addEventListener(
                "resize",
                function () {
                    handleScrolling($contentsNav);
                },
                false
            );
            handleScrolling($contentsNav);
        }
    }

    return {
        init: init,
        scrollToElement: scrollToElement,
        handleScrollToClick: handleScrollToClick
    };
}

/**
 * Restructures DOM content & layout.
 * @author Amplified - http://amplifiedbydesign.com/
 * @returns {object}
 */
function openVinoContent() {
    /**
     * Initialize/do this on page load
     * @public
     */
    function init() {
        // Init viewer.js instance
        new Viewer(document.querySelector('.contents'));

        var $container = $('<div id="container"></div>');
        // Move Search Box to titlearea
        $("#titlearea").append($("#MSearchBox"));
        // Add search slider
        var searchSlider = $('<div class="closed" id="search-slider"></div>');
        $('#MSearchBox').prepend(searchSlider);
        searchSlider.on('click', function() {
            $(this).toggleClass('closed open');
            $("#MSearchField").animate({width:'toggle'},200);
        });
       if (['http:', 'https:'].indexOf(window.location.protocol) !== -1) {
           $('#MSearchField').replaceWith('<input type="text" name="query" id="MSearchField" value="Search" accesskey="S" onfocus="searchBox.OnSearchFieldFocus(true)">');
           if (!$('#FSearchBox').length) {
                $('#MSearchField').wrap('<form id="FSearchBox" action="g-search.html" method="get"></form>');
           }
           var currentVersion = getCurrentVersion();
           if (currentVersion === 'cn') {
               currentVersion = 'latest';
           }
           if (currentVersion === 'latest') {
               var latestVersion = '';
               try {
                   latestVersion = JSON.parse(data)[1].version;
               }
               catch(e) {};
               currentVersion = latestVersion;
           }
           if(currentVersion) {
                $('#FSearchBox').append('<input type="hidden" name="version" value="' + currentVersion  + '">');
           }
       }
        
        // Add favicon.ico
        $('<link/>', {
            'href': '/assets/images/favicon.ico',
            'rel': 'shortcut icon'
        }).appendTo('head');

        // Restructure DOM contents
        if (!$("body").hasClass("homepage")) {
            $container.append($(".header, .contents")).insertAfter($("#top"));
            $(".contents").prepend($(".header"));
        }

        // Detect any anchor in the content that links to an image and add target="_blank" to it
        // This lets a user see very large images if they appear too narrow on page
        // Less processor-intensive than interpreting every single image dimension at pageload
        $(".textblock a").each(function () {
            if (this.href.indexOf('.png' || '.jpg' || 'jpeg' || 'gif') != -1) {
                $(this).attr('target', '_blank')
            }
        });

        // Look for any anchor that has an href of [none] and remove that href, add class so
        // it doesn't have a hover state
        $("a").each(function () {
            if (this.href.indexOf('[none]') != -1) {
                $(this).removeAttr("href").addClass("removehover");
            }
        });

        // Wrap tables in a class so we can style them
        $(".contents > table, .textblock > table").wrap(
            '<div class="table-wrapper"></div>'
        );

        // Handle any blockquotes and added the classes needed to style them
        // Doxygen doesn't handle hand-inserted HTML very well so we are left with blockquotes
        // Get the first bold item in a blockquote, apply class based on content
        $("blockquote p b").each(function () {
            // Set up variable to call as object literal
            var blockquoteDeclarations = {
                "NOTE": "blockquote_note",
                "NOTES": "blockquote_note",
                "TIP": "blockquote_tip",
                "TIPS": "blockquote_tip",
                "IMPORTANT": "blockquote_caution",
                "CAUTION": "blockquote_caution",
                "WARNING": "blockquote_warning"
            };
            // Strip punctuation from label so we do not force everybody to add every keyword and all variants
            var blockquoteLabel = $(this).html().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, "");
            $(this).parent().parent().addClass(blockquoteDeclarations[blockquoteLabel] || '');
        });
    }

    return {
        init: init
    };
}

function getURLParameter(name) {
    return decodeURIComponent((new RegExp('[?|&]'+name+
           '='+'([^&;]+?)(&|#|;|$)').exec(location.search)
           ||[,""])[1].replace(/\+/g, '%20'))||null;
}

/**
 * Handle building the versions links, updating URLs on "homepage", and managing
 * the versions "select".
 * @author Amplified - http://amplifiedbydesign.com/
 * @returns {object}
 */
function openVinoVersions() {
    /**
     * Initialize/do this on page load
     * @public
     */
    function init() {
        buildVersionsContent();

        if ($("body").hasClass("homepage")) {
            updateAllLinks();
        }
    }

    function getPageUrlWithVersion(url, version) {
        var fullURL = window.location.pathname.split('/');
        var lastElement = fullURL.slice(-1)[0];
        var newURL = url.replace(getCurrentVersion(), version);

        if ($("body").hasClass("homepage")) {
            if (window.location.protocol == "file:") {
                newURL = url.replace("index.html", version + "/" + landingPage);
            }
            if (lastElement == "404.html") {
                newURL = url.replace("404.html", version + "/" + landingPage);
            } else {
                newURL = documentationHost + "/" + version + "/" + landingPage;
            }
        }

        return newURL;
    }

    function buildVersionsContent() {
        if (versions && versions.length > 0) {
            var $versionsContent = $('<div id="versionsSelector"></div>');
            var $selectorButton = $(
                '<button type="button" class="current-version version-toggle"></button>'
            );
            var $versionsList = $('<ul id="versionsList"></ul>');
            var currentVersion = getCurrentVersion();
            var listItems = "";
            var url = window.location.href;
            var max_length = 0;

            for (var i = 0, verLength = versions.length; i < verLength; i++) {
                var ver = versions[i];
                if (ver.version === currentVersion) {
                    listItems += '<li class="active">';
                } else {
                    listItems += "<li>";
                }
                listItems +=
                    '<a href="' +
                    getPageUrlWithVersion(url, ver.version) +
                    '">Version ' +
                    ver.version +
                    "</a>";
                listItems += "</li>";
                if (ver.version.length > max_length){
                    max_length = ver.version.length;
                }
            }

            var scale = 7;
            var versionLength = 'version '.length;
            var align = 5;
            $versionsList.css("width", ((max_length + versionLength + align) * scale).toString() + "px");

            var handleVersionBlur = function () {
                if ($versionsList.hasClass("opened")) {
                    window.removeEventListener("click", handleVersionBlur, false);
                    $versionsList.removeClass("opened");
                } else {
                    $versionsList.toggleClass("opened");
                }
            };

            $selectorButton.on("click", function () {
                window.addEventListener("click", handleVersionBlur, false);
            });

            $versionsList.append(listItems);
            $versionsContent.append(
                $selectorButton.html("Version <strong>" + currentVersion + "</strong>")
            );
            $versionsContent.append($versionsList);

            if ($("#MSearchBox").length > 0) {
                $versionsContent.insertBefore($("#top #MSearchBox"));
            } else {
                $("#titlearea").append($versionsContent);
            }
        }
    }

    return {
        init: init
    };
}

function getDomainName() {
    var protocol = window.location.protocol + "//";
    var index = window.location.href.indexOf(protocol);
    var link = window.location.href.slice(index + protocol.length);
    return window.location.protocol + "//" + link.split('/')[0];
}

function getCurrentPath() {
    var windowLocationArr = window.location.pathname.split("/");
    var currentPath = windowLocationArr[windowLocationArr.length - 1];
    return currentPath;
}

// Get current version from URL
function getCurrentVersion() {
    var protocol = window.location.protocol + "//";
    var index = window.location.href.indexOf(protocol);
    var link = window.location.href.slice(index + protocol.length).split('/');
    var wordAfterDomain = link[1];
    if (wordAfterDomain === 'cn') {
        wordAfterDomain = link[2];
    }
    if (["index.html", "404.html", "", "latest"].indexOf(wordAfterDomain) >= 0) {
        /*
        * If this landing page, 404 or domain.com we should get first version
        * */
        return versions[0].version;
    }
    return encodeURI(wordAfterDomain);
}

function updateAllLinks() {
    var currentVersion = getCurrentVersion();
    var domainName = getDomainName();
    $(".openvino-content a, .api-content a, .footer a").each(function () {
        $(this).attr("href", function (index, old) {
            return old.replace('<version_placeholder>', currentVersion);
        });
    });
    $(".homelink-id, a.download").each(function () {
        $(this).attr("href", function (index, old) {
            return old.replace('<domain_placeholder>', domainName);
        });
    });
    $('[property="og:url"]').each(function () {
        $(this).attr("content", function (index, old) {
            return old.replace('<domain_placeholder>', domainName);
        });
    });
}

/**
 * Takes content generated by Doxygen and manipulates the DOM in order to "theme"
 * the output.
 * @author Amplified - http://amplifiedbydesign.com/
 */
(function openVinoLayout() {
    var LayoutBuilder = openVinoContent();
    var MenuBuilder = openVinoMenu();

    // Fire off init functions once the DOM content is loaded
    window.addEventListener(
        "DOMContentLoaded",
        function load() {
            window.removeEventListener("DOMContentLoaded", load, false);
            LayoutBuilder.init();
            MenuBuilder.init();
        },
        false
    );

    // Have to wait for FULL page load before firing the following functions...
    window.addEventListener(
        "load",
        function fullLoad() {
            window.removeEventListener("loaded", fullLoad, false);
            
            function setScrolling() {
                if (window.location.hash) {
                    var hash = window.location.hash;
                    var $selectedEl = $(hash)[0];
    
                    if ($selectedEl) {
                        MenuBuilder.scrollToElement($selectedEl);
                    }
                }
            }

            // check if "onhashchange" event is supported by Browser
            if ("onhashchange" in window) {
                $(window).on('hashchange', function() {
                    setScrolling();
                });
            }

            var contentsAnchors = document.querySelectorAll('.contents a[href^="#"]');
            if (contentsAnchors.length > 0) {
                for (
                    var i = 0, anchorsLength = contentsAnchors.length;
                    i < anchorsLength;
                    i++
                ) {
                    contentsAnchors[i].addEventListener(
                        "click",
                        MenuBuilder.handleScrollToClick(
                            contentsAnchors[i].hash.substring(1)
                        )
                    );
                }
            }

            setScrolling();
            updateAllLinks();
        },
        false
    );
})();
