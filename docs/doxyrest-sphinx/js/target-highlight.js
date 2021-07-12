//..............................................................................
//
//  This file is part of the Doxyrest toolkit.
//
//  Doxyrest is distributed under the MIT license.
//  For details see accompanying license.txt file,
//  the public copy of which is also available at:
//  http://tibbo.com/downloads/archive/doxyrest/license.txt
//
//..............................................................................

(function () {

var g_oldTarget = $([]);
var g_suitableTargetClass = "pre, table, a, h2, h3, h4, h5, h6"; // don't highlight <h1>

function adjustTarget(target) {
	if (target.is("div"))
		target = target.children(":first");

	if (target.is(g_suitableTargetClass))
		return target;

	if (!target.is("span"))
		return $([]);

	if (target.hasClass("doxyrest-code-target")) {
		return target;
	}

	// ascend to the top-level <span>

	var parent = target.parent();
	while (parent.is("span")) {
		target = parent;
		parent = parent.parent();
	}

	// check the direct parent of the top-level <span>

	if (parent.is("pre"))
		return parent;

	// nope, skip all sibling spans

	target = target.next();
	while (target.is("span"))
		target = target.next();

	if (target.is(g_suitableTargetClass))
		return target;

	return $([]); // no target to highlight
}

function updateTarget() {
	var target = adjustTarget($(":target"));
	g_oldTarget.removeClass("doxyrest-target-highlight");
	target.addClass("doxyrest-target-highlight");
	g_oldTarget = target;
}

$(document).ready(updateTarget);
$(window).on('hashchange', updateTarget);

} ());

//..............................................................................
