function summarize() {
	chrome.tabs.executeScript(null, { file: "jquery-2.2.4.min.js" }, function() {
	    chrome.tabs.executeScript(null, { file: "content.js" });
	});
}
document.getElementById('clickme').addEventListener('click', summarize);