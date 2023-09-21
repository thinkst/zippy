/*
Noise-cancelling headphones for the internet

(C) 2023 Thinkst Applied Research
Author: Jacob Torrey

This extension classifies all the text in every <p> on a page using a fast LZMA-based LLM-detector model.
It sets the opacity of the <p> to the confidence that the <p> is human-written, so LLM-generated content
fades away.
*/
// Import the other background service worker code
importScripts('lzma_worker-min.js');
importScripts('nlzmadetect.js');

const CONTEXT_MENU_ID = 'NCH_CONTEXT_MENU';
function ext_add_to_lp(info, _tab) {
    if (info.menuItemId != CONTEXT_MENU_ID)
        return;
    //console.log("Adding to LP: " + info.selectionText);
    add_to_lp(info.selectionText);
    chrome.storage.local.set({lp: local_prelude});
}
chrome.contextMenus.removeAll();
chrome.contextMenus.create({
    title: "Add text as AI-generated",
    contexts: ["selection"],
    id: CONTEXT_MENU_ID
});
chrome.contextMenus.onClicked.addListener(ext_add_to_lp)

chrome.storage.local.get(["lp"], function(val) {
    if ("lp" in val)
        local_prelude = val.lp;
});

// How much to boost or decrease the scores
const skew_factor = 0.9

// Function to analyze text for its origin
async function query(text) {
    var score = detect_string(text);
    if (score == 1)
        return 1;
    return score / skew_factor;
}

// Tracking tabs that are dead to avoid sending responses to.
var dead_tabs = 0;
async function return_results(tab, id, score) {
    const msg = {id: id, opacity: score};
    chrome.tabs.sendMessage(tab, msg, {}, function(_res) { if (chrome.runtime.lastError) dead_tabs++; });
}

function handle_req(msg, sender, _sendResp) {
    query(msg.content).then((score) => { if (score != 1) return_results(sender.tab.id, msg.id, score) });
}

chrome.runtime.onMessage.addListener(handle_req);

