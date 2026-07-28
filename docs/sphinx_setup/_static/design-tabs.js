var sd_labels_by_text = {};
var pageHash = getPageHash(window.location.pathname)
var languages = ['py', 'cpp', 'c']

function ready() {
  const savedState = getTabsState(pageHash)
  const languageState = getLanguageState()
  const preselect = []
  let changeLanguage = false
  const li = document.getElementsByClassName("sd-tab-label");
  for (const label of li) {
    syncId = label.getAttribute("data-sync-id");
    if (syncId) {
      label.onclick = onLabelClick;
      if (!sd_labels_by_text[syncId]) {
        sd_labels_by_text[syncId] = [];
      }
      sd_labels_by_text[syncId].push(label);
      if (syncId == languageState) changeLanguage = true;
      if (savedState.includes(syncId) && !preselect.includes(syncId))
        preselect.push(syncId)
    }
  }
  for (item of preselect) selectItem(item);
  if (changeLanguage) selectItem(languageState);
}

function onLabelClick() {
  const syncId = this.getAttribute("data-sync-id");
  selectItem(syncId)

  const tabsState = getTabsState()
  saveCurrentTabsState(tabsState)
  if (languages.includes(syncId))
    saveLanguageState(syncId)
}

function selectItem(item) {
  for (label of sd_labels_by_text[item]) {
    label.previousElementSibling.checked = true;
  }
}

function getTabsState(currentPageHash) {
  allPagesState = JSON.parse(window.localStorage.getItem('sphinx-design-tabs-state')) || {}
  if (currentPageHash)
    return getCurrentPageTabsState(currentPageHash, allPagesState)
  return allPagesState
}

function getCurrentPageTabsState(currentPageHash, allPagesState) {
  return allPagesState[currentPageHash] || []
}

function getLanguageState() {
  return window.localStorage.getItem('sphinx-design-language-state')
}

function saveCurrentTabsState(tabsState) {
  const checkedElements = []
  for (label in sd_labels_by_text) {
    if(sd_labels_by_text[label][0].previousElementSibling.checked)
    checkedElements.push(label);
  }
  tabsState[pageHash] = checkedElements

  window.localStorage.setItem('sphinx-design-tabs-state', JSON.stringify(tabsState))
}

function saveLanguageState(language) {
  window.localStorage.setItem('sphinx-design-language-state', language)
}

function getPageHash(url) {
  return url.split('').map(v=>v.charCodeAt(0)).reduce((a,v)=>a+((a<<7)+(a<<3))^v).toString(16);
}

// document.addEventListener("DOMContentLoaded", ready, false);
