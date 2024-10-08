var cookieVersion = 0;

var hiddenBanner = document.getElementById("hidebanner");
if (hiddenBanner) {
  var dataUrl = hiddenBanner.getAttribute("data-url");
  window.onload = (e) => {
    fetch(dataUrl).then((data) => {
      data.json().then((item) => {
        if (item != null && item.p != "" && item.version != "" && item.style != "") {
          cookieVersion = item.version;
          var transitionBanner = document.getElementById("info-banner");
          if (document.cookie.split(';').filter(function (find_cookie_name) {
            return find_cookie_name.trim().indexOf('TransitionBannerIsHiddenX' + cookieVersion + '=') === 0;
          }
          ).length) {
            transitionBanner.classList.add("hidden-banner");
          }
          transitionBanner.classList.add(item.style);

          const p = document.createElement("p");
          p.innerHTML = item.p;
          transitionBanner.append(p);

          const button = document.createElement("button");
          button.className = "close-banner";
          button.type = "button";

          const span = document.createElement("span");
          span.setAttribute('aria-hidden', 'true');
          span.innerHTML = '&times;';
          button.appendChild(span);
          button.addEventListener("click", function () {
            var cookieContent = 'TransitionBannerIsHiddenX' + cookieVersion + '=true;';
            var expiry = 'expires=';
            var date = new Date();
            var expirationDate = date.getTime() + (365 * 24 * 60 * 60 * 1000);
            date.setTime(expirationDate);
            expiry += date.toUTCString();
            document.cookie = cookieContent + expiry;
            var transitionBanner = document.getElementById("info-banner");
            transitionBanner.classList.add("hidden-banner");
          });
          transitionBanner.append(button)
        }
      });
    });
  };
}

