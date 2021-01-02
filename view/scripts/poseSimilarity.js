// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
PoseNet example using p5.js
=== */
let totalScore = 0;
let data = imgModels;
let startCorrect = new Date();
let correctDurationThreshold = 2000; // in miliseconds
let checkPerFrames = 10;
let currentFrame = 0;
let prevMaxScore = 0;

// config
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

function restartCorrectTime(){
    startCorrect = new Date();
}

function isCompleteCorrectDuration(timeLeft) {
    if(timeLeft <= 0) {
        return true;
    }
    return false;
}

function setCorrectPoseStatus() {
    let endCorrect = new Date();
    let timeLeft = correctDurationThreshold - (endCorrect.getTime() - startCorrect.getTime());

    document.getElementById('pose-status').style['color'] = 'green';
    document.getElementById("pose-status-duration").innerHTML = `Correct! Keep the pose for ${Math.round(timeLeft/1000 * 10) / 10} s left`;

    return timeLeft;
}

function setIncorrectPoseStatus() {
    document.getElementById('pose-status').style['color'] = 'orange';
    document.getElementById("pose-status-duration").innerHTML = 'Incorrect Pose!!';
}

function setBadPoseStatus() {
    document.getElementById('pose-status').style['color'] = 'red';
    document.getElementById("pose-status-duration").innerHTML = 'Pose not detected!! Ensure the whole body was seen!';
}

function getSimilarity(modelFeaturesObj, userFeaturesObj) {
    // remove unqualified features
    let [modelFeatures, userFeatures, modelConfidences, qualifiedFeatures] = removeUnqualifiedKeypoints(modelFeaturesObj, userFeaturesObj);

    // check qualified features threshold
    if(qualifiedFeatures.length < minFeaturesThreshold){
        currentFrame = 0;
        prevMaxScore = 0;
        restartCorrectTime();
        setBadPoseStatus();
        return false;
    }

    // standardize features
    let modelFeaturesScaled = standardization(modelFeatures);
    let userFeaturesScaled = standardization(userFeatures)

    // split features in 3 parts
    let [modelFace, modelTorso, modelLegs] = splitInFaceLegsTorso(modelFeaturesScaled, qualifiedFeatures);
    let [userFace, userTorso, userLegs] = splitInFaceLegsTorso(userFeaturesScaled, qualifiedFeatures);

    if(userFace.length < minFaceFeaturesThreshold || userTorso.length < minTorsoFeaturesThreshold || userLegs.length < minLegsFeaturesThreshold){
        currentFrame = 0;
        prevMaxScore = 0;
        restartCorrectTime();
        setBadPoseStatus();
        return false;
    }

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

    let maxDistances = {
        face: maxDistFace,
        torso: maxDistTorso,
        legs: maxDistLegs
    }

    let avgDistances = {
        face: avgDistFace,
        torso: avgDistTorso,
        legs: avgDistLegs
    }

    let rotations = {
        face: rotationFace,
        torso: rotationTorso,
        legs: rotationLegs
    }

    let allAvgDist = maxDistFace + maxDistTorso + maxDistLegs;
    // let allAvgDist = avgDistFace + avgDistTorso + avgDistLegs;

    document.getElementById("face").innerHTML = avgCosineSimilarity;
    document.getElementById("torso").innerHTML = allAvgDist;

    if(avgCosineSimilarity <= 0.3 & allAvgDist <= 0.3){
        let correctTimeLeft = setCorrectPoseStatus();

        if(isCompleteCorrectDuration(correctTimeLeft)){
            totalScore += 100;
            document.getElementById("total").innerHTML = Math.floor(totalScore);
            return true;
        }
        return false;
    }

    restartCorrectTime();
    setIncorrectPoseStatus();
    return false;

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
  setExampleImage();

  var canvas = createCanvas(360, 360);
  canvas.parent('main');
  video = createCapture(VIDEO);
  video.size(width, height);

  // Create a new poseNet method with a single detection
  poseNet = ml5.poseNet(video, options, modelReady);
  
  // This sets up an event that fills the global variable "poses"
  // with an array every time new poses are detected
  poseNet.on('pose', function(results) {
    poses = results;
    if(poses && poses.length > 0) {
      userPoseData = {
        score: poses[0].pose.score,
        keypoints: poses[0].pose.keypoints
      };
      isSimilar = getSimilarity(exPoseData, userPoseData);
    }
  });
  // Hide the video element, and just show the canvas
  video.hide();
}

function modelReady() {
  document.getElementById('loading').style['display'] = 'none';
  document.getElementById('main').style['display'] = 'block';
}

function generateNewModel() {
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

  if(isSimilar) {
    currentFrame = 0;
    prevMaxScore = 0;
    restartCorrectTime();
    isSimilar = !isSimilar;
    generateNewModel();
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