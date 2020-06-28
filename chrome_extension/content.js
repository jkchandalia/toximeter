//TO DO: need to find way to grab only relevant text. Limit to say reddit. 

alert("Generating summary highlights. This may take up to 30 seconds depending on length of article.");

function unicodeToChar(text) {
	return text.replace(/\\u[\dA-F]{4}/gi, 
	      function (match) {
	           return String.fromCharCode(parseInt(match.replace(/\\u/g, ''), 16));
	      });
}

// capture all text
var textToSend = document.body.innerText;
console.log(typeof(textToSend));
var realText = document.querySelectorAll("[data-test-id='comment']")
//console.log(realText[0]);
//console.log(realText.length);
var counter = 0;
for (let i = 0; i<realText.length; i++) {
	//console.log(realText[i]);
	var comments = realText[i].getElementsByTagName('p');
	//console.log(comments);
	for (let j = 0; j<comments.length; j++){
		counter++;
		console.log(comments[j].innerHTML);
	}
}

var commentArray = new Array(counter);
counter=0;
for (let i = 0; i<realText.length; i++) {
	//console.log(realText[i]);
	var comments = realText[i].getElementsByTagName('p');
	//console.log(comments);
	for (let j = 0; j<comments.length; j++){
		commentArray[counter] = comments[j].innerHTML;
		counter++;
		console.log(comments[j].innerHTML);
	}
}

alert(JSON.stringify(commentArray));

console.log(realText);
var textArray = textToSend.split("\n");

for (let i = 0; i < 30; i++) {
	var value = textArray[i];
	//console.log("value");
	//console.log(value);
	//console.log("html split");
	//console.log(document.body.innerHTML.split(value));
	//if (value){
	//document.body.innerHTML = document.body.innerHTML.split(value).join('<span style="background-color: #fff799;">' + value + '</span>');}
  }
  
/* for (const value in textToSend.split("\n")) {
	document.body.innerHTML = document.body.innerHTML.split(value).join('<span style="background-color: #fff799;">' + value + '</span>');
  } */
/* $.each(textToSend, function( index, value ) {
	value = unicodeToChar(value).replace(/\\n/g, '');
	document.body.innerHTML = document.body.innerHTML.split(value).join('<span style="background-color: #fff799;">' + value + '</span>');
}) */


// summarize and send back
//const api_url = 'YOUR_GOOGLE_CLOUD_FUNCTION_URL';

/* fetch(api_url, {
  method: 'POST',
  body: JSON.stringify(textToSend),
  headers:{
    'Content-Type': 'application/json'
  } })
.then(data => { return data.json() })
.then(res => { 
	$.each(res, function( index, value ) {
		value = unicodeToChar(value).replace(/\\n/g, '');
		document.body.innerHTML = document.body.innerHTML.split(value).join('<span style="background-color: #fff799;">' + value + '</span>');
	});
 })
.catch(error => console.error('Error:', error)); */

