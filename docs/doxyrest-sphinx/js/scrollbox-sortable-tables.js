$(document).ready(function () {
    sortableTables();
    floatFirstRowInTable();
});

function sortableTables() {
    
    var tables = $('.sortable-table').find('table');
    tables.each(function () {
        var table = $(this);
        var headings = table.find('th');
        headings.each(function () {
            var th = $(this);
            var index = th.index();
            var sortBtn = $('<span class="sort-btn"></span>');
            th.addClass('sort-header');
            th.click(function () {
                var counter = 0;
                sortBtn.addClass('sort-active');
                sortBy = sortBtn.data('sortby');
                var trs = table.find('tbody tr');
                sortBtn.toggleClass('ascending');
                trs.sort(function (item1, item2) {

                    if (sortBtn.hasClass('ascending')) {
                        var text1 = $(item1).find('td').eq(index).text();
                        var text2 = $(item2).find('td').eq(index).text();
                    }
                    else {
                        var text1 = $(item2).find('td').eq(index).text();
                        var text2 = $(item1).find('td').eq(index).text();
                    }
                    // try converting to num
                    var _text1 = parseFloat(text1);
                    var _text2 = parseFloat(text2);

                    if (!isNaN(_text1) && !isNaN(_text2)) {
                        text1 = _text1;
                        text2 = _text2;
                    }
                    if (text1 > text2) {
                        return 1;
                    }
                    else if (text1 < text2) {
                        return -1;
                    }
                    else {
                        return 0;
                    }
                }).map(function () {
                    var row = $(this);
                    if (counter % 2 === 0) {
                        row.removeClass('row-odd');
                        row.addClass('row-even');
                    }
                    else {
                        row.removeClass('row-even');
                        row.addClass('row-odd');
                    }
                    counter++;
                    table.find('tbody').append(row);
                });

                headings.each(function () {
                    if ($(this).index() !== index) {
                        $(this).find('.sort-btn').removeClass('ascending');
                        $(this).find('.sort-btn').removeClass('sort-active');
                    }
                });
            });
            th.find('p').append(sortBtn);
        });
    });
}

function floatFirstRowInTable() {
    var $table = $('.scrollbox').find('table.table');
    $table.floatThead({
        scrollContainer: function($table){
            return $table.closest('.scrollbox');
    }
    });
}

/** @preserve jQuery.floatThead 2.2.2 - https://mkoryak.github.io/floatThead/ - Copyright (c) 2012 - 2021 Misha Koryak **/

! function(bt) {
    bt.floatThead = bt.floatThead || {}, bt.floatThead.defaults = {
        headerCellSelector: "tr:visible:first>*:visible",
        zIndex: 1001,
        position: "auto",
        top: 0,
        bottom: 0,
        scrollContainer: function(t) {
            return bt([])
        },
        responsiveContainer: function(t) {
            return bt([])
        },
        getSizingRow: function(t, e, o) {
            return t.find("tbody tr:visible:first>*:visible")
        },
        ariaLabel: function(t, e, o) {
            return e.text()
        },
        floatTableClass: "floatThead-table",
        floatWrapperClass: "floatThead-wrapper",
        floatContainerClass: "floatThead-container",
        copyTableClass: !0,
        autoReflow: !1,
        debug: !1,
        support: {
            bootstrap: !0,
            datatables: !0,
            jqueryUI: !0,
            perfectScrollbar: !0
        },
        floatContainerCss: {
            "overflow-x": "hidden"
        }
    };
    var wt = function() {
            var n = {},
                o = Object.prototype.hasOwnProperty;
            n.has = function(t, e) {
                return o.call(t, e)
            }, n.keys = Object.keys || function(t) {
                if (t !== Object(t)) throw new TypeError("Invalid object");
                var e = [];
                for (var o in t) n.has(t, o) && e.push(o);
                return e
            };
            var r = 0;
            return n.uniqueId = function(t) {
                var e = ++r + "";
                return t ? t + e : e
            }, bt.each(["Arguments", "Function", "String", "Number", "Date", "RegExp"], function() {
                var e = this;
                n["is" + e] = function(t) {
                    return Object.prototype.toString.call(t) === "[object " + e + "]"
                }
            }), n.debounce = function(o, n, r) {
                var a, i, l, s, d;
                return function() {
                    l = this, i = arguments, s = new Date;
                    var e = function() {
                            var t = new Date - s;
                            t < n ? a = setTimeout(e, n - t) : (a = null, r || (d = o.apply(l, i)))
                        },
                        t = r && !a;
                    return a || (a = setTimeout(e, n)), t && (d = o.apply(l, i)), d
                }
            }, n
        }(),
        gt = "undefined" != typeof MutationObserver,
        mt = function() {
            for (var t = 3, e = document.createElement("b"), o = e.all || []; t = 1 + t, e.innerHTML = "\x3c!--[if gt IE " + t + "]><i><![endif]--\x3e", o[0];);
            return 4 < t ? t : document.documentMode
        }(),
        t = /Gecko\//.test(navigator.userAgent),
        yt = /WebKit\//.test(navigator.userAgent),
        Tt = /rtl/i.test(document.documentElement.dir || "");
    mt || t || yt || (mt = 11);
    var l = function() {
            if (yt) {
                var t = bt("<div>").css("width", "0").append(bt("<table>").css("max-width", "100%").append(bt("<tr>").append(bt("<th>").append(bt("<div>").css("min-width", "100px").text("X")))));
                bt("body").append(t);
                var e = 0 === t.find("table").width();
                return t.remove(), e
            }
            return !1
        },
        Ct = !t && !mt,
        xt = bt(window),
        Lt = t && window.matchMedia;
    if (!window.matchMedia || Lt) {
        var e = window.onbeforeprint,
            o = window.onafterprint;
        window.onbeforeprint = function() {
            e && e(), xt.triggerHandler("fth-beforeprint")
        }, window.onafterprint = function() {
            o && o(), xt.triggerHandler("fth-afterprint")
        }
    }

    function St(t) {
        var e = t[0].parentElement;
        do {
            if ("visible" !== window.getComputedStyle(e).getPropertyValue("overflow")) break
        } while (e = e.parentElement);
        return e === document.body ? bt([]) : bt(e)
    }

    function jt(t) {
        window && window.console && window.console.error && window.console.error("jQuery.floatThead: " + t)
    }

    function zt(t) {
        var e = t.getBoundingClientRect();
        return e.width || e.right - e.left
    }

    function It() {
        var t = document.createElement("scrolltester");
        t.style.cssText = "width:100px;height:100px;overflow:scroll!important;position:absolute;top:-9999px;display:block", document.body.appendChild(t);
        var e = t.offsetWidth - t.clientWidth;
        return document.body.removeChild(t), e
    }

    function Ht(t, e, o) {
        var n = o ? "outerWidth" : "width";
        if (l && t.css("max-width")) {
            var r = 0;
            o && (r += parseInt(t.css("borderLeft"), 10), r += parseInt(t.css("borderRight"), 10));
            for (var a = 0; a < e.length; a++) r += zt(e.get(a));
            return r
        }
        return t[n]()
    }
    bt.fn.floatThead = function(t) {
        if (t = t || {}, mt < 8) return this;
        if (wt.isFunction(l) && (l = l()), wt.isString(t)) {
            var r = t,
                a = Array.prototype.slice.call(arguments, 1),
                i = this;
            return this.filter("table").each(function() {
                var t = bt(this),
                    e = t.data("floatThead-lazy");
                e && t.floatThead(e);
                var o = t.data("floatThead-attached");
                if (o && wt.isFunction(o[r])) {
                    var n = o[r].apply(this, a);
                    void 0 !== n && (i = n)
                }
            }), i
        }
        var vt = bt.extend({}, bt.floatThead.defaults || {}, t);
        if (bt.each(t, function(t, e) {
                t in bt.floatThead.defaults || !vt.debug || jt("Used [" + t + "] key to init plugin, but that param is not an option for the plugin. Valid options are: " + wt.keys(bt.floatThead.defaults).join(", "))
            }), vt.debug) {
            var e = bt.fn.jquery.split(".");
            1 === parseInt(e[0], 10) && parseInt(e[1], 10) <= 7 && jt("jQuery version " + bt.fn.jquery + " detected! This plugin supports 1.8 or better, or 1.7.x with jQuery UI 1.8.24 -> http://jqueryui.com/resources/download/jquery-ui-1.8.24.zip")
        }
        return this.filter(":not(." + vt.floatTableClass + ")").each(function() {
            var e = wt.uniqueId(),
                m = bt(this);
            if (m.data("floatThead-attached")) return !0;
            if (!m.is("table")) throw new Error('jQuery.floatThead must be run on a table element. ex: $("table").floatThead();');
            var o = vt.autoReflow && gt,
                n = null,
                d = m.children("thead:first"),
                r = m.children("tbody:first");
            if (0 === d.length || 0 === r.length) return vt.debug && (0 === d.length ? jt("The thead element is missing.") : jt("The tbody element is missing.")), m.data("floatThead-lazy", vt), void m.unbind("reflow").one("reflow", function() {
                m.floatThead(vt)
            });
            m.data("floatThead-lazy") && m.unbind("reflow"), m.data("floatThead-lazy", !1);
            var y, T, a = !0,
                C = {
                    vertical: 0,
                    horizontal: 0
                };
            wt.isFunction(It) && (It = It());
            var f = 0;
            !0 === vt.scrollContainer && (vt.scrollContainer = St);
            var x = vt.scrollContainer(m) || bt([]),
                L = 0 < x.length,
                S = L ? bt([]) : vt.responsiveContainer(m) || bt([]),
                j = _(),
                z = null;
            "auto" === vt.position ? z = null : "fixed" === vt.position ? z = !1 : "absolute" === vt.position ? z = !0 : vt.debug && jt('Invalid value given to "position" option, valid is "fixed", "absolute" and "auto". You passed: ', vt.position), null == z && (z = L);
            var I = m.find("caption"),
                H = 1 === I.length;
            if (H) var W = "top" === (I.css("caption-side") || I.attr("align") || "top");
            var i = bt("<fthfoot>").css({
                    display: "table-footer-group",
                    "border-spacing": "0",
                    height: "0",
                    "border-collapse": "collapse",
                    visibility: "hidden"
                }),
                q = !1,
                l = bt([]),
                E = mt <= 9 && !L && z,
                c = bt("<table/>"),
                u = bt("<colgroup/>"),
                p = m.children("colgroup:first"),
                h = !0;
            0 === p.length && (p = bt("<colgroup/>"), h = !1);
            var v = h ? "col:visible" : "col",
                b = bt("<fthtr>").css({
                    display: "table-row",
                    "border-spacing": "0",
                    height: "0",
                    "border-collapse": "collapse"
                }),
                R = bt("<div>").css(vt.floatContainerCss).attr("aria-hidden", "true"),
                M = !1,
                s = bt("<thead/>"),
                w = bt('<tr class="size-row"/>'),
                g = bt([]),
                k = bt([]),
                D = bt([]),
                F = bt([]);
            s.append(w), m.prepend(p), Ct && (i.append(b), m.append(i)), c.append(u), R.append(c), vt.copyTableClass && c.attr("class", m.attr("class")), c.attr({
                cellpadding: m.attr("cellpadding"),
                cellspacing: m.attr("cellspacing"),
                border: m.attr("border")
            });
            var t = m.css("display");
            if (c.css({
                    borderCollapse: m.css("borderCollapse"),
                    border: m.css("border"),
                    display: t
                }), L || c.css("width", "auto"), "none" === t && (M = !0), c.addClass(vt.floatTableClass).css({
                    margin: "0",
                    "border-bottom-width": "0"
                }), z) {
                var O = function(t, e) {
                    var o = t.css("position"),
                        n = t;
                    if (!("relative" === o || "absolute" === o) || e) {
                        var r = {
                            paddingLeft: t.css("paddingLeft"),
                            paddingRight: t.css("paddingRight")
                        };
                        R.css(r), n = t.data("floatThead-containerWrap") || t.wrap(bt("<div>").addClass(vt.floatWrapperClass).css({
                            position: "relative",
                            clear: "both"
                        })).parent(), t.data("floatThead-containerWrap", n), q = !0
                    }
                    return n
                };
                L ? (l = O(x, !0)).prepend(R) : (l = O(m), m.before(R))
            } else m.before(R);
            R.css({
                position: z ? "absolute" : "fixed",
                marginTop: "0",
                top: z ? "0" : "auto",
                zIndex: vt.zIndex,
                willChange: "transform"
            }), R.addClass(vt.floatContainerClass), V();
            var N = {
                    "table-layout": "fixed"
                },
                A = {
                    "table-layout": m.css("tableLayout") || "auto"
                },
                Q = m[0].style.width || "",
                U = m.css("minWidth") || "";

            function G(t) {
                return t + ".fth-" + e + ".floatTHead"
            }

            function P() {
                var t = 0;
                if (d.children("tr:visible").each(function() {
                        t += bt(this).outerHeight(!0)
                    }), "collapse" === m.css("border-collapse")) {
                    var e = parseInt(m.css("border-top-width"), 10);
                    parseInt(m.find("thead tr:first").find(">*:first").css("border-top-width"), 10) < e && (t -= e / 2)
                }
                w.outerHeight(t), g.outerHeight(t)
            }

            function V() {
                y = (wt.isFunction(vt.top) ? vt.top(m) : vt.top) || 0, T = (wt.isFunction(vt.bottom) ? vt.bottom(m) : vt.bottom) || 0
            }

            function X() {
                if (!a) {
                    if (a = !0, z) {
                        var t = Ht(m, F, !0);
                        l.width() < t && m.css("minWidth", t)
                    }
                    m.css(N), c.css(N), c.append(d), r.before(s), P()
                }
            }

            function Y() {
                a && (a = !1, z && m.width(Q), s.detach(), m.prepend(d), m.css(A), c.css(A), m.css("minWidth", U), m.css("minWidth", Ht(m, F)))
            }
            var B = !1;

            function K(t) {
                B !== t && (B = t, m.triggerHandler("floatThead", [t, R]))
            }

            function $(t) {
                z !== t && (z = t, R.css({
                    position: z ? "absolute" : "fixed"
                }))
            }

            function J() {
                var l, s = function() {
                    var t, e = d.find(vt.headerCellSelector);
                    if (h ? t = p.find(v).length : (t = 0, e.each(function() {
                            t += parseInt(bt(this).attr("colspan") || 1, 10)
                        })), t !== f) {
                        f = t;
                        var o = [],
                            n = [];
                        w.empty();
                        for (var r = 0; r < t; r++) {
                            var a = document.createElement("th"),
                                i = document.createElement("span");
                            i.setAttribute("aria-label", vt.ariaLabel(m, e.eq(r), r)), a.appendChild(i), a.className = "floatThead-col", w[0].appendChild(a), o.push("<col/>"), n.push(bt("<fthtd>").css({
                                display: "table-cell",
                                height: "0",
                                width: "auto"
                            }))
                        }
                        o = h ? p.html() : o.join(""), Ct && (b.empty(), b.append(n), F = b.find("fthtd")), g = w.find("th"), h || p.html(o), k = p.find(v), u.html(o), D = u.find(v)
                    }
                    return t
                }();
                return function() {
                    var t = R.scrollLeft();
                    k = p.find(v);
                    var e, o, n, r, a = (e = m, o = k, n = F, r = mt, Ct ? n : r ? vt.getSizingRow(e, o, n) : o);
                    if (a.length === s && 0 < s) {
                        if (!h)
                            for (l = 0; l < s; l++) k.eq(l).css("width", "");
                        Y();
                        var i = [];
                        for (l = 0; l < s; l++) i[l] = zt(a.get(l));
                        for (l = 0; l < s; l++) D.eq(l).width(i[l]), k.eq(l).width(i[l]);
                        X()
                    } else c.append(d), m.css(A), c.css(A), P();
                    R.scrollLeft(t), m.triggerHandler("reflowed", [R])
                }
            }

            function Z(t) {
                var e = x.css("border-" + t + "-width"),
                    o = 0;
                return e && ~e.indexOf("px") && (o = parseInt(e, 10)), o
            }

            function _() {
                return "auto" === S.css("overflow-x")
            }

            function tt() {
                var i, l = x.scrollTop(),
                    s = 0,
                    d = H ? I.outerHeight(!0) : 0,
                    f = W ? d : -d,
                    c = R.height(),
                    u = m.offset(),
                    p = 0,
                    h = 0;
                if (L) {
                    var t = x.offset();
                    s = u.top - t.top + l, H && W && (s += d), p = Z("left"), h = Z("top"), s -= h
                } else i = u.top - y - c + T + C.horizontal;
                var v = xt.scrollTop(),
                    b = xt.scrollLeft(),
                    w = function() {
                        return (_() ? S : x).scrollLeft() || 0
                    },
                    g = w();
                return function(t) {
                    j = _();
                    var e = m[0].offsetWidth <= 0 && m[0].offsetHeight <= 0;
                    if (!e && M) return M = !1, setTimeout(function() {
                        m.triggerHandler("reflow")
                    }, 1), null;
                    if (e && (M = !0, !z)) return null;
                    if ("windowScroll" === t) v = xt.scrollTop(), b = xt.scrollLeft();
                    else if ("containerScroll" === t)
                        if (S.length) {
                            if (!j) return;
                            g = S.scrollLeft()
                        } else l = x.scrollTop(), g = x.scrollLeft();
                    else "init" !== t && (v = xt.scrollTop(), b = xt.scrollLeft(), l = x.scrollTop(), g = w());
                    if (!yt || !(v < 0 || Tt && 0 < b || !Tt && b < 0)) {
                        if (E) $("windowScrollDone" === t);
                        else if ("windowScrollDone" === t) return null;
                        var o, n;
                        u = m.offset(), H && W && (u.top += d);
                        var r = m.outerHeight();
                        if (L && z) {
                            if (l <= s) {
                                var a = s - l + h;
                                o = 0 < a ? a : 0, K(!1)
                            } else r - c < l - s ? o = r - c - l - s : (o = q ? h : l, K(!0));
                            n = p
                        } else !L && z ? (i + r + f < v ? o = r - c + f + T : u.top >= v + y ? (o = 0, Y(), K(!1)) : (o = y + v - u.top + s + (W ? d : 0), X(), K(!0)), n = g) : L && !z ? (l < s || r < l - s ? (o = u.top - v, Y(), K(!1)) : (o = u.top + l - v - s, X(), K(!0)), n = u.left + g - b) : L || z || (i + r + f < v ? o = r + y - v + i + f : u.top > v + y ? (o = u.top - v, X(), K(!1)) : (o = y, K(!0)), n = u.left + g - b);
                        return {
                            top: Math.round(o),
                            left: Math.round(n)
                        }
                    }
                }
            }

            function et() {
                var i = null,
                    l = null,
                    s = null;
                return function(t, e, o) {
                    if (null != t && (i !== t.top || l !== t.left)) {
                        if (8 === mt) R.css({
                            top: t.top,
                            left: t.left
                        });
                        else {
                            var n = "translateX(" + t.left + "px) translateY(" + t.top + "px)",
                                r = {
                                    "-webkit-transform": n,
                                    "-moz-transform": n,
                                    "-ms-transform": n,
                                    "-o-transform": n,
                                    transform: n,
                                    top: "0",
                                    left: "0"
                                };
                            R.css(r)
                        }
                        i = t.top, l = t.left
                    }
                    e && function() {
                        var t = Ht(m, F, !0),
                            e = j ? S : x,
                            o = e.length ? zt(e[0]) : t,
                            n = "hidden" !== e.css("overflow-y") ? o - C.vertical : o;
                        if (R.width(n), L) {
                            var r = 100 * t / n;
                            c.css("width", r + "%")
                        } else c.css("width", t + "px")
                    }(), o && P();
                    var a = (j ? S : x).scrollLeft();
                    z && s === a || (R.scrollLeft(a), s = a)
                }
            }

            function ot() {
                if (x.length)
                    if (vt.support && vt.support.perfectScrollbar && x.data().perfectScrollbar) C = {
                        horizontal: 0,
                        vertical: 0
                    };
                    else {
                        if ("scroll" === x.css("overflow-x")) C.horizontal = It;
                        else {
                            var t = x.width(),
                                e = Ht(m, F),
                                o = n < r ? It : 0;
                            C.horizontal = t - o < e ? It : 0
                        }
                        if ("scroll" === x.css("overflow-y")) C.vertical = It;
                        else {
                            var n = x.height(),
                                r = m.height(),
                                a = t < e ? It : 0;
                            C.vertical = n - a < r ? It : 0
                        }
                    }
            }
            ot();
            var nt = function() {
                J()()
            };
            nt();
            var rt = tt(),
                at = et();
            at(rt("init"), !0);
            var it = wt.debounce(function() {
                    at(rt("windowScrollDone"), !1)
                }, 1),
                lt = function() {
                    at(rt("windowScroll"), !1), E && it()
                },
                st = function() {
                    at(rt("containerScroll"), !1)
                },
                dt = wt.debounce(function() {
                    m.is(":hidden") || (ot(), V(), nt(), rt = tt(), at(rt("reflow"), !0, !0))
                }, 1),
                ft = function() {
                    Y()
                },
                ct = function() {
                    X()
                },
                ut = function(t) {
                    t.matches ? ft() : ct()
                },
                pt = null;
            if (window.matchMedia && window.matchMedia("print").addListener && !Lt ? (pt = window.matchMedia("print")).addListener(ut) : (xt.on("fth-beforeprint", ft), xt.on("fth-afterprint", ct)), L ? z ? x.on(G("scroll"), st) : (x.on(G("scroll"), st), xt.on(G("scroll"), lt)) : (S.on(G("scroll"), st), xt.on(G("scroll"), lt)), xt.on(G("load"), dt), function(t, e) {
                    if (8 === mt) {
                        var o = xt.width(),
                            n = wt.debounce(function() {
                                var t = xt.width();
                                o !== t && (o = t, e())
                            }, 1);
                        xt.on(t, n)
                    } else xt.on(t, wt.debounce(e, 1))
                }(G("resize"), function() {
                    m.is(":hidden") || (V(), ot(), nt(), rt = tt(), (at = et())(rt("resize"), !0, !0))
                }), m.on("reflow", dt), vt.support && vt.support.datatables && function(t) {
                    if (t.dataTableSettings)
                        for (var e = 0; e < t.dataTableSettings.length; e++) {
                            var o = t.dataTableSettings[e].nTable;
                            if (t[0] === o) return !0
                        }
                    return !1
                }(m) && m.on("filter", dt).on("sort", dt).on("page", dt), vt.support && vt.support.bootstrap && xt.on(G("shown.bs.tab"), dt), vt.support && vt.support.jqueryUI && xt.on(G("tabsactivate"), dt), o) {
                var ht = null;
                wt.isFunction(vt.autoReflow) && (ht = vt.autoReflow(m, x)), ht || (ht = x.length ? x[0] : m[0]), (n = new MutationObserver(function(t) {
                    for (var e = function(t) {
                            return t && t[0] && ("THEAD" === t[0].nodeName || "TD" === t[0].nodeName || "TH" === t[0].nodeName)
                        }, o = 0; o < t.length; o++)
                        if (!e(t[o].addedNodes) && !e(t[o].removedNodes)) {
                            dt();
                            break
                        }
                })).observe(ht, {
                    childList: !0,
                    subtree: !0
                })
            }
            m.data("floatThead-attached", {
                destroy: function() {
                    var t = ".fth-" + e;
                    return Y(), m.css(A), p.remove(), Ct && i.remove(), s.parent().length && s.replaceWith(d), K(!1), o && (n.disconnect(), n = null), m.off("reflow reflowed"), x.off(t), S.off(t), q && (x.length ? x.unwrap() : m.unwrap()), L ? x.data("floatThead-containerWrap", !1) : m.data("floatThead-containerWrap", !1), m.css("minWidth", U), R.remove(), m.data("floatThead-attached", !1), xt.off(t), xt.off("fth-beforeprint fth-afterprint"), pt && pt.removeListener(ut), ft = ct = function() {},
                        function() {
                            return m.floatThead(vt)
                        }
                },
                reflow: function() {
                    dt()
                },
                setHeaderHeight: function() {
                    P()
                },
                getFloatContainer: function() {
                    return R
                },
                getRowGroups: function() {
                    return a ? R.find(">table>thead").add(m.children("tbody,tfoot")) : m.children("thead,tbody,tfoot")
                }
            })
        }), this
    }
}(function() {
    var t = window.jQuery;
    return "undefined" != typeof module && module.exports && !t && (t = require("jquery")), t
}());