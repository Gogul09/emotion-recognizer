//------------------
// GLOBAL VARIABLES
//------------------
let localStream;
var isMobileOrTablet = false;
var isWebcamOn = 0;
var modelName = "mobilenet";
var NUM_CLASSES = 5;
var predictCount = 0;
var predictMax = 1000;

let extractor;
let classifier;
let xs;
let ys;

var SAMPLE_BOX = {
	0: 0,
	1: 0,
	2: 0,
	3: 0,
	4: 0
}

var CLASS_MAP = {
	0: "emoticon-laugh",
	1: "emoticon-excited",
	2: "emoticon-sad",
	3: "emoticon-angry",
	4: "emoticon-sleep"
}

//-----------------------------
// disable support for mobile 
// and tablet
//-----------------------------
function mobileAndTabletcheck() {
  var check = false;
  (function(a){if(/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino|android|ipad|playbook|silk/i.test(a)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0,4))) check = true;})(navigator.userAgent||navigator.vendor||window.opera);
  isMobileOrTablet = check;
};

window.onload = function() {
	mobileAndTabletcheck();
	if (isMobileOrTablet) {
		document.getElementById("emotion-container").style.display = "none";
		document.getElementById("mobile-tablet-warning").style.display = "block";
	} else {
		loadExtractor();
	}
}

//-----------------------
// start webcam capture
//-----------------------
function startWebcam() {
	predictCount = 0;
	var video = document.getElementById('main-stream-video');
	vendorUrl = window.URL || window.webkitURL;

	navigator.getMedia = navigator.getUserMedia ||
						 navigator.webkitGetUserMedia ||
						 navigator.mozGetUserMedia ||
						 navigator.msGetUserMedia;

	// capture video from webcam
	navigator.getMedia({
		video: true,
		audio: false
	}, function(stream) {
		localStream = stream;
		video.srcObject = stream;
		video.play();
		isWebcamOn = 1;
	}, function(error) {
		alert("Something wrong with webcam!");
		isWebcamOn = 0;
	});

}

//---------------------
// stop webcam capture
//---------------------
function stopWebcam() {
	localStream.getVideoTracks()[0].stop();
	isWebcamOn = 0;
	predictCount = predictMax + 1;
}

//------------------------------
// capture webcam stream and 
// assign it to a canvas object
//------------------------------
function captureWebcam() {
	var video = document.getElementById("main-stream-video");

	var canvas    = document.createElement("canvas");
	var context   = canvas.getContext('2d');
	canvas.width  = video.width;
	canvas.height = video.height;

	context.drawImage(video, 0, 0, video.width, video.height);
	tensor_image = preprocessImage(canvas);

	var canvasObj = {
    	canvasElement: canvas,
    	canvasTensor : tensor_image
  	};

	return canvasObj;
}

//---------------------------------
// take snapshot for each category
//---------------------------------
function captureSample(id, label) {
	if (isWebcamOn == 1) {
		
		canvasObj = captureWebcam();
		canvas = canvasObj["canvasElement"];
		tensor_image = canvasObj["canvasTensor"];

		var img_id = id.replace("sample", "image");
		var img    = document.getElementById(img_id);
		img.src    = canvas.toDataURL();

		// add the sample to the training tensor
		addSampleToTensor(extractor.predict(tensor_image), label);

		SAMPLE_BOX[label] += 1;
		document.getElementById(id.replace("sample", "count")).innerHTML = SAMPLE_BOX[label] + " samples";

	} else {
		alert("Please turn on the webcam first!")
	}
}

//------------------------------------
// preprocess the image from webcam
// to be mobilenet friendly
//------------------------------------
function preprocessImage(img) {
	const tensor        = tf.browser.fromPixels(img)
						  .resizeNearestNeighbor([224, 224]);
	const croppedTensor = cropImage(tensor);
	const batchedTensor = croppedTensor.expandDims(0);
	
	return batchedTensor.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
}

//------------------------------------
// crop the image from the webcam
// region of interest: center portion
//------------------------------------
function cropImage(img) {
	const size = Math.min(img.shape[0], img.shape[1]);
	const centerHeight = img.shape[0] / 2;
	const beginHeight = centerHeight - (size / 2);
	const centerWidth = img.shape[1] / 2;
	const beginWidth = centerWidth - (size / 2);
	return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}

//------------------------------------
// hold each sample as a tensor that
// has 4 dimensions
//------------------------------------
function addSampleToTensor(sample, label) {
	const y = tf.tidy(
		() => tf.oneHot(tf.tensor1d([label]).toInt(), NUM_CLASSES));
	if(xs == null) {
		xs = tf.keep(sample);
		ys = tf.keep(y);
	} else {
		const oldX = xs;
		xs = tf.keep(oldX.concat(sample, 0));
		const oldY = ys;
		ys = tf.keep(oldY.concat(y, 0));
		oldX.dispose();
		oldY.dispose();
		y.dispose();
	}
}

//------------------------------------
// train the classifier with the 
// obtained tensors from the user
//------------------------------------
async function train() {
	var selectLearningRate = document.getElementById("emotion-learning-rate");
	const learningRate     = selectLearningRate.options[selectLearningRate.selectedIndex].value;

	var selectBatchSize    = document.getElementById("emotion-batch-size");
	const batchSizeFrac    = selectBatchSize.options[selectBatchSize.selectedIndex].value;

	var selectEpochs       = document.getElementById("emotion-epochs");
	const epochs           = selectEpochs.options[selectEpochs.selectedIndex].value;

	var selectHiddenUnits  = document.getElementById("emotion-hidden-units");
	const hiddenUnits      = selectHiddenUnits.options[selectHiddenUnits.selectedIndex].value;

	if(xs == null) {
		alert("Please add some samples before training!");
	} else {
		classifier = tf.sequential({
			layers: [
				tf.layers.flatten({inputShape: [7, 7, 256]}),
				tf.layers.dense({
					units: parseInt(hiddenUnits),
					activation: "relu",
					kernelInitializer: "varianceScaling",
					useBias: true
				}),
				tf.layers.dense({
					units: parseInt(NUM_CLASSES),
					kernelInitializer: "varianceScaling",
					useBias: false,
					activation: "softmax"
				})
			]
		});
		const optimizer = tf.train.adam(learningRate);
		classifier.compile({optimizer: optimizer, loss: "categoricalCrossentropy"});

		const batchSize = Math.floor(xs.shape[0] * parseFloat(batchSizeFrac));
		if(!(batchSize > 0)) {
			alert("Please choose a non-zero fraction for batchSize!");
		}
		
		// create loss visualization
		var lossTextEle = document.getElementById("emotion-loss");
		if (typeof(lossTextEle) != 'undefined' && lossTextEle != null) {
			lossTextEle.innerHTML = "";
		} else {
			var lossText = document.createElement("P");
			lossText.setAttribute("id", "emotion-loss");
			lossText.classList.add('emotion-loss');
			document.getElementById("emotion-controller").insertBefore(lossText, document.getElementById("emotion-controller").children[1]);
			var lossTextEle = document.getElementById("emotion-loss");
		}

		classifier.fit(xs, ys, {
			batchSize,
			epochs: parseInt(epochs),
			callbacks: {
				onBatchEnd: async (batch, logs) => {
					lossTextEle.innerHTML = "Loss: " + logs.loss.toFixed(5);
					await tf.nextFrame();
				}
			}
		});
	}
}


//-------------------------------------
// load mobilenet model from Google
// and return a model that has the
// internal activations from a 
// specific feature layer in mobilenet
//-------------------------------------
async function loadExtractor() {
	// load mobilenet from Google
	const mobilenet = await tf.loadLayersModel("https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json");

	// return the mobilenet model with 
	// internal activations from "conv_pw_13_relu" layer
	const feature_layer = mobilenet.getLayer("conv_pw_13_relu");

	// return mobilenet model with feature activations from specific layer
	extractor = tf.model({inputs: mobilenet.inputs, outputs: feature_layer.output});
}

//------------------------------
// Predict what the user plays
//------------------------------
var isPredicting = false;
async function predictPlay() {
	isPredicting = true;
	while (isPredicting) {
		const predictedClass = tf.tidy(() => {
			canvasObj = captureWebcam();
			canvas = canvasObj["canvasElement"];
			const img = canvasObj["canvasTensor"];
			const features = extractor.predict(img);
			const predictions = classifier.predict(features);
			return predictions.as1D().argMax();
		});

		const classId = (await predictedClass.data())[0];
		predictedClass.dispose();
		highlightTile(classId);

		await tf.nextFrame();
	}
}

//------------------------------------------
// highlight the emoticon corresponding to
// user's emotion
//------------------------------------------
function highlightTile(classId) {
	var tile_play    = document.getElementById(CLASS_MAP[classId].replace("emoticon", "emotion"));	

	var tile_plays = document.getElementsByClassName("emotion-kit-comps");
	for (var i = 0; i < tile_plays.length; i++) {
		tile_plays[i].style.borderColor     = "#e9e9e9";
		tile_plays[i].style.backgroundColor = "#ffffff";
		tile_plays[i].style.transform       = "scale(1.0)";
	}

	tile_play.style.borderColor     = "#e88139";
	tile_play.style.backgroundColor = "#ff9c56";
	tile_play.style.transform       = "scale(1.1)";
}