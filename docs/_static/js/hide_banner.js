function closeTransitionBanner() {
   var cookieContent = 'TransitionBannerIsHiddenX=true;';
   var expiry ='expires=';
   var date = new Date();
   var expirationDate = date.getTime() + (365 * 24 * 60 * 60 * 1000);
   date.setTime(expirationDate);
   expiry += date.toUTCString();
   document.cookie = cookieContent + expiry;
   var transitionBanner = document.getElementById("info-banner");
   transitionBanner.classList.add("hidden-banner");
   }
 if (document.cookie.split(';').filter(function (find_cookie_name) {return find_cookie_name.trim().indexOf('TransitionBannerIsHiddenX=') === 0; }).length) {
     var transitionBanner = document.getElementById("info-banner");
     transitionBanner.classList.add("hidden-banner");
}