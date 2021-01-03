// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
PoseNet example using p5.js
=== */
let currentScore = 0;
let totalScore = 0;
let data = imgModels;
let startCorrect = new Date();
let correctDurationThreshold = 2000; // in miliseconds
let checkPerFrames = 10;
let currentFrame = 0;
let prevMaxScore = 0;

// config
let counter = 20;
let goodAudio = new Audio('assets/audio/good.mp3');
let excellentAudio = new Audio('assets/audio/excellent.mp3');
let badAudio = new Audio('assets/audio/bad.mp3');
let recordData = {};
let {scale, rotate, translate, compose, applyToPoints} = window.TransformationMatrix;
let idx = Math.floor(Math.random() * data.length);
let example;
let exPoseData;
let video;
let poseNet;
let poses = [];
let userPoseData;
let isSimilar = false;
let options = {
    architecture: 'ResNet50',
    outputStride: 32,
    inputResolution: { width: 257, height: 200 },
    quantBytes: 2
}
let showKeypoints = true;

function playAudio(audio) {
    audio.play();
}

function getSimilarity(modelFeaturesObj, userFeaturesObj) {
    // remove unqualified features
    let [modelFeatures, userFeatures, modelConfidences, qualifiedFeatures] = removeUnqualifiedKeypoints(modelFeaturesObj, userFeaturesObj);

    // standardize features
    let modelFeaturesScaled = standardization(modelFeatures);
    let userFeaturesScaled = standardization(userFeatures)

    // split features in 3 parts
    let [modelFace, modelTorso, modelLegs] = splitInFaceLegsTorso(modelFeaturesScaled, qualifiedFeatures);
    let [userFace, userTorso, userLegs] = splitInFaceLegsTorso(userFeaturesScaled, qualifiedFeatures);

    // affine transformation
    let [transformedFace, AFace] = affineTransformation(modelFace, userFace);
    let [transformedTorso, ATorso] = affineTransformation(modelTorso, userTorso);
    let [transformedLegs, ALegs] = affineTransformation(modelLegs, userLegs);

    // average joint cosine similarity
    let avgCosineSimilarity = getBodyPartCosineSimilarity(modelFace, 
        modelTorso, modelLegs, transformedFace, transformedTorso, transformedLegs, modelConfidences);

    // max distance and rotations
    let [maxDistFace, avgDistFace, rotationFace, totalConfFace] = maxDistanceAndRotation(modelFace, 
        transformedFace, AFace, modelConfidences.slice(0, faceFeatureList.length));
    let [maxDistTorso, avgDistTorso, rotationTorso, totalConfTorso] = maxDistanceAndRotation(modelTorso, 
        transformedTorso, ATorso, modelConfidences.slice(faceFeatureList.length, faceFeatureList.length + torsoFeatureList.length));
    let [maxDistLegs, avgDistLegs, rotationLegs, totalConfLegs] = maxDistanceAndRotation(modelLegs, 
        transformedLegs, ALegs, modelConfidences.slice(faceFeatureList.length + torsoFeatureList.length, featureList.length));

    // let allAvgDist = maxDistFace + maxDistTorso + maxDistLegs;
    let allAvgDist = avgDistFace + avgDistTorso + avgDistLegs;
    currentScore = Math.floor(100 - 50*(avgCosineSimilarity + allAvgDist));
    if(currentScore > 70) {
        document.getElementById('current').style['color'] = 'green';
    } else if(currentScore > 55) {
        document.getElementById('current').style['color'] = 'blue';
    } else {
        document.getElementById('current').style['color'] = 'red';
    }
    document.getElementById("current").innerHTML = currentScore;
}

function setExampleImage() {
  // get example
  example = data[idx];
  exFilename = example.filename;
  recordData[exFilename] = {};
  // exResult = example.result;
  imgObj = document.getElementById("img-example");
  imgObj.src = exFilename;
  exPoseData = {
    score: example.score,
    keypoints: example.keypoints
  };
}

function setup() {
  generateNewModel();
  var canvas = createCanvas(360, 360);
  canvas.parent('main');
  video = createCapture(VIDEO);
  video.size(width, height);

  // Create a new poseNet method with a single detection
  poseNet = ml5.poseNet(video, options, modelReady);
  
  // This sets up an event that fills the global variable "poses"
  // with an array every time new poses are detected
  poseNet.on('pose', function(results) {
    currentFrame += 1;
    poses = results;
    if(poses && poses.length > 0 && currentFrame %checkPerFrames == 0) {
      currentFrame = 0;
      userPoseData = {
        score: poses[0].pose.score,
        keypoints: poses[0].pose.keypoints
      };
      getSimilarity(exPoseData, userPoseData);
    }
  });
  // Hide the video element, and just show the canvas
  video.hide();
}

function modelReady() {
  window.setInterval(function(){
    counter--;
    document.getElementById("pose-status-duration").innerHTML = `Time Remaining: ${counter}s`;
    if(counter <= 3){
        document.getElementById('pose-status-duration').style['color'] = 'orange';
    }
    if(counter === 0){
        if(currentScore > 70){
            playAudio(excellentAudio);
        } else if(currentScore > 55){
            playAudio(goodAudio);
        } else {
            playAudio(badAudio);
        }
        generateNewModel();
        totalScore += currentScore;
        document.getElementById("total").innerHTML = Math.floor(totalScore);
    }
  }, 1000);
  document.getElementById('loading').style['display'] = 'none';
  document.getElementById('main').style['display'] = 'block';
}

function generateNewModel() {
    counter = 20;
    document.getElementById('pose-status-duration').style['color'] = 'black';
    document.getElementById("pose-status-duration").innerHTML = `Time Remaining: ${counter}s`;
    idx = Math.floor(Math.random() * data.length);
    setExampleImage();
}

function draw() {
  image(video, 0, 0, width, height);

  // We can call both functions to draw all keypoints and the skeletons
  if(showKeypoints){
    drawKeypoints();
    drawSkeleton();
  }
}

// toggle model image
function toggleModel(){
    var modelImg = document.getElementById('img-example');
    var toggleImg = document.getElementById('toggleModel');
    if(modelImg.style.display === 'none') {
        modelImg.style.display = 'inline-block';
        toggleImg.innerHTML = 'Hide Image';
    } else {
        modelImg.style.display = 'none';
        toggleImg.innerHTML = 'Show Image';
    }
}

// toggle model image
function toggleKeypoints(){
    showKeypoints = !showKeypoints;
}