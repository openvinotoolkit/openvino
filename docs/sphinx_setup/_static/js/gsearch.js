/* API config */
var cx = '012886673101693426607:wjpczzz2qi3';
var path = 'https://content.googleapis.com/discovery/v1/apis/customsearch/v1/rest';
var apiKey = 'AIzaSyBpwSxfoP2T_EK62wwNhZ9zyxUzeRjmOwA';


var potKeywords = ['int8', '8-bit', '8 bit', 'quantization', 'quantized', 
                    'quantizer', 'fakequantize', 'fq', 'model optimization',
                    'low precision', 'low-precision', 'lower precision', 'pot',
                    'compression', 'calibration', 'post-training', 'post training'];

// get the docs versions stored in versions_raw.js
var versions;
try {
  versions = JSON.parse(data);
}
catch(e) {
  versions = [];
};

function hasPotTerms(query) {
  q = query.toLowerCase();
  return potKeywords.some(function(k) {
    k = k.toLowerCase();
   return q.indexOf(k) != -1;
  });
}

function boostPage(name, pages, pos) {
  var tmp;
  pages = pages || [];
  for (var i = 1; i < pages.length; i++) {
    if (name === getPageName(pages[i].link)) {
      tmp = pages[pos];
      pages[pos] = pages[i];
      pages[i] = tmp;
      pos++;
    }
  }
}

function getPageName(url) {
  urlArr = url.split('/');
  return urlArr[urlArr.length - 1];
}

function getURLParameter(name) {
  return decodeURIComponent((new RegExp('[?|&]'+name+
         '='+'([^&;]+?)(&|#|;|$)').exec(location.search)
         ||[,""])[1].replace(/\+/g, '%20'))||null;
}

function search(query, selectedVersion, page, key) {
    // make a query to Google Search API and render the results
    var start = getStartIndex(page); // get start index from page number
    var documentationHost = 'https://docs.openvino.ai';
    if (selectedVersion === 'ALL') {
      var siteSearch = documentationHost;
    }
    else {
     var siteSearch = documentationHost + '/' + selectedVersion; 
    }
    var path = 'https://www.googleapis.com/customsearch/v1?cx=' + cx + '&q=' + query + '&siteSearch=' + siteSearch + '&start=' + start;
    var hasPot = hasPotTerms(query);
    if (hasPot) {
      path += '&qt="pot"';
    }
    
    return async function() {
        await gapi.client.init({
            'apiKey': apiKey,
        }).then(function() {
            return gapi.client.request({
                'path': path
            })
        }).then(function(response) {
            handleResponse(response, selectedVersion, hasPot);
        }, function(err) {
            console.log(err);
            handleNoResults();
        });
    }
}

function getPageNum(index) {
    // get page number from page index
    return Math.floor((index-1) / 10) + 1;
}

function getStartIndex(page) {
  // get start index from page number
  return (page - 1) * 10 + 1;
}

function handleNoResults() {
  var $noResults = $('<div>', {'class': 'gs-no-results', 'text': 'No Results'});
  $('#searchinfo').append($noResults);
}

function handleResponse(response, selectedVersion, hasPot) {
    var result = response.result;
    var searchInformation = result.searchInformation;
    var totalResults = parseInt(searchInformation.totalResults);
    var limit = null;
    var previousPage, nextPage;
    previousPage = nextPage = null;
    var queries = result.queries;
    var request = queries.request[0];
    var searchTerms = request.searchTerms;
    var startIndex = request.startIndex;
    var currentPage = getPageNum(startIndex);

    $('#searchinfo').text('Search Results for "' + searchTerms + '"');
    $('#searchfield').val(searchTerms);
    if (queries.hasOwnProperty('nextPage')) {
        nextPage = getPageNum(queries.nextPage[0].startIndex);
        if (nextPage > 10) {
          nextPage = null;
          limit = currentPage;
        }
    }
    else {
        limit = currentPage;
    }
    if (queries.hasOwnProperty('previousPage')) {
        previousPage = getPageNum(queries.previousPage[0].startIndex);
    }
    
    var count = parseInt(request.count);
    totalResults = (count + totalResults) > 100 ? 100 : totalResults;
    var query = request.searchTerms;
    var totalPages = Math.ceil(totalResults / count);
    var items = response.result.items;
    if (hasPot) {
      boostPage('pot_README.html', items, 0);
    }

    // add version tabs
    addVersionTabs(selectedVersion, query);

    if (!(totalPages || items)) {
      handleNoResults();
      totalPages = limit;
    }
    else {
      // render the results
      renderResults(items);
    }
    addPagination(query, selectedVersion, currentPage, previousPage, nextPage, totalPages, limit);
}

function addVersionTabs(selectedVersion, query) {
  // remove latest
  var tab_versions = [{'version': 'ALL'}];
  var latestVersion;
  if (versions.length) {
    tab_versions = [{'version': 'ALL'}].concat(versions.slice(0, -1));
    latestVersion = tab_versions[1].version;
  }
  for (var i = 0; i < tab_versions.length; i++) {
    var href;
    var $elem = $('<a>', { 'class': 'gs-tab', 'href': 'search.html?version=' + tab_versions[i].version + '&query=' + query, 'text': tab_versions[i].version });
    if (tab_versions[i].version === 'ALL') {
      href = 'search.html?query=' + query;
    }
    else {
      href = 'search.html?query=' + query + '&version=' + tab_versions[i].version
    }
    var $elem = $('<a>', { 'class': 'gs-tab', 'href': href, 'text': tab_versions[i].version });
    if (selectedVersion === tab_versions[i].version) {
      $elem.addClass('gs-tab-active');
    }
    if (latestVersion === tab_versions[i].version) {
      $elem.text('latest (' + $elem.text() + ')');
    }
    $('#gs-tabs-area').append($elem);
  }
}

function renderResults(items) {
    var $elements = $('<div>', {'class': 'gs-results'});
    $.each(items, function(i, item) {
         // result item
         var $resultItem = $('<div>', {'class': 'gs-result' });
         var $title = $('<div>', {'class': 'gs-title'});
         $title.append($('<a>', {'target': '_blank', 'href': item['link'], 'html': item['htmlTitle']}));
         var $url = $('<div>', {'class': 'gs-url', 'html': item['link']});
         var $context = $('<div>', {'class': 'gs-context'});
         var $snippet = $('<div>', {'class': 'gs-snippet', 'html': item['htmlSnippet'].replace(/\<br\>/g, '')});

        // media
        if (item.pagemap.hasOwnProperty('cse_thumbnail')) {
          var $media = $('<div>', {'class': 'gs-media'});
          var cse_thumbnail = item.pagemap.cse_thumbnail[0];
          var $link = $('<a>', {'href': item['link'], 'target': '_blank'});
          var $image = $('<img>', { 'src': cse_thumbnail.src, 'alt': 'Thumbnail image' });
          $link.append($image);
          $media.append($link);
          $context.append($media);
        }

        // file format
        if (item.hasOwnProperty('fileFormat')) {
          var $ffBox = $('<div>', {'class': 'gs-file-format-box'});
          var $fileFormat = $('<span>', {'class': 'gs-file-format', 'text': 'File format: '});
          var $type = $('<span>', {'class': 'gs-file-type', 'text': item['fileFormat']});
          $ffBox.append($fileFormat);
          $ffBox.append($type);
          $context.append($ffBox);
        }

        $context.append($snippet);

        $resultItem.append($title);
        $resultItem.append($url);
        $resultItem.append($context);
        $elements.append($resultItem);

     });
    $('#searchresults').append($elements);
}

function addPagination(query, selectedVersion, currentPage, previousPage, nextPage, totalPages, limit) {
    function getPageList(currentPage, totalPages, limit) {
        var x, left, right, pageList;
        left = right = currentPage;
        pageList = [];
        x = 1;
        while (x < 10 && x < totalPages) {
            if (left > 1) {
                left--;
                x++;
            }
            if (right < totalPages) {
                right++;
                x++;
            }
        }
        
        if (limit === currentPage) {
          right = currentPage;
        }

        var diff = (right - left) + 1;
        var rest = 10 - diff;

        if (rest !==  0) {
          left = left - rest < 1 ? 1 : left - rest;
        }

        for(i = left; i <= right; i++) {
            pageList.push(i);
        }
        return pageList;
    }

    var pageList = getPageList(currentPage, totalPages, limit);

    var $cursor = $('<div>', {'class': 'gs-cursor'});

    if (previousPage) {
        var $previous = $('<a>', {
          'text': 'Previous',
          'class': 'gs-cursor-page-previous',
          'href': 'search.html?page=' + previousPage + '&query=' + query + '&version=' + selectedVersion
        });
        $cursor.append($previous);
    }

    var $page;
    for(var i = 0; i < pageList.length; i++) {
        $page = $('<a>', {
          'class': 'gs-cursor-page',
          'href': 'search.html?page=' + pageList[i] + '&query=' + query + '&version=' + selectedVersion,
          'text': pageList[i]
        });
        if (currentPage === pageList[i]) {
            $page.addClass('active');
        }
        $cursor.append($page);
    }

    if (nextPage) {
        var $next = $('<a>', {
          'text': 'Next',
          'class': 'gs-cursor-page-next',
          'href': 'search.html?page=' + nextPage + '&query=' + query + '&version=' + selectedVersion
        });
        $cursor.append($next);
    }
    $('#searchresults').append($cursor);
}

function trim(s) {
  return s?s.replace(/^\s\s*/, '').replace(/\s\s*$/, ''):'';
}

var entityMap = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': '&quot;',
  "'": '&#39;',
  "/": '&#x2F;'
};

function escapeHtml(s) {
  return String(s).replace(/[&<>"'\/]/g, function (s) {
    return entityMap[s];
  });
}


function versionExists(v) {
  for (var i = 0; i < versions.length; i++) {
    if (versions[i].version === v) {
      return true;
    }
  }
  return false;
}

function getCurrentPath() {
  var windowLocationArr = window.location.href.split("/");
  var currentPath = encodeURI(
      windowLocationArr[windowLocationArr.length - 1].replace(
          window.location.hash,
          ""
  ));
  return currentPath;
}

function getURLParameter(name) {
  return decodeURIComponent((new RegExp('[?|&]'+name+
         '='+'([^&;]+?)(&|#|;|$)').exec(location.search)
         ||[,""])[1].replace(/\+/g, '%20'))||null;
}

$(document).ready(function() {
  // remove duplicate search
  $('#top > #MSearchField').remove();
  
  var query = trim(getURLParameter('query'));
  if (!query) {
    handleNoResults();
    return;
  }
  var page = trim(getURLParameter('page')) || 1;
  var selectedVersion = trim(getURLParameter('version'));
  if (versionExists(selectedVersion)) {
    if (window.location.pathname.startsWith('/cn')) {
      selectedVersion = 'cn/' + selectedVersion;
    }
  }
  else {
    selectedVersion = 'ALL';
  }

  // the key is used to store and retrieve data from sessionStorage
  var key = getCurrentPath();
  
  var data;
  if (key in sessionStorage) {
    try {
      data = JSON.parse(sessionStorage[key]);
    }
    catch(err) {};
  }
  gapi.load('client', search(query, selectedVersion,page, key));
});
