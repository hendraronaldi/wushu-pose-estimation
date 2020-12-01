// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
PoseNet example using p5.js
=== */
let score = 0;
let data = imgModels;

// config
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
let confidenceLevelThreshold = 0.5;
let minFaceFeaturesThreshold = 3;
let minTorsoFeaturesThreshold = 5;
let minLegsFeaturesThreshold = 5;
let minFeaturesThreshold = minFaceFeaturesThreshold + minTorsoFeaturesThreshold + minLegsFeaturesThreshold;
let featureList = [
    "Nose",
    "leftEye",
    "rightEye",
    "leftEar",
    "rightEar",
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
    "leftHip",
    "rightHip",
    "leftKnee",
    "rightKnee",
    "leftAnkle",
    "rightAnkle"
]

let faceFeatureList = [
    "Nose",
    "leftEye",
    "rightEye",
    "leftEar",
    "rightEar"
]

let torsoFeatureList = [
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist"
]

let legsFeatureList = [
    "leftHip",
    "rightHip",
    "leftKnee",
    "rightKnee",
    "leftAnkle",
    "rightAnkle"
]

function standardization(features){
    // min max scale
    const {MinMaxScaler} = ml.preprocessing;
    const scalerX = new MinMaxScaler({featureRange: [0, 1]});
    const scalerY = new MinMaxScaler({featureRange: [0, 1]});

    const X = features.map((arr) => {return arr[0]});
    const Y = features.map((arr) => {return arr[1]});

    const scaledX = scalerX.fit_transform(X);
    const scaledY = scalerY.fit_transform(Y);
    const scaledFeatures = scaledX.map((_, i) => {return [scaledX[i], scaledY[i]]});

    return scaledFeatures;
}

function removeUnqualifiedKeypoints(modelFeaturesObj, userFeaturesObj){
    let qualifiedFeatures = [];
    let modelArr = [];
    let userArr = [];

    for(var i=0; i<featureList.length; i++){
        if(!(modelFeaturesObj.keypoints[i].score < confidenceLevelThreshold ||
            userFeaturesObj.keypoints[i].score < confidenceLevelThreshold)){
            
                qualifiedFeatures.push(featureList[i]);

                modelArr.push([modelFeaturesObj.keypoints[i].position.x, modelFeaturesObj.keypoints[i].position.y]);
                userArr.push([userFeaturesObj.keypoints[i].position.x, userFeaturesObj.keypoints[i].position.y]);
        }
    }
    return [modelArr, userArr, qualifiedFeatures];
}

function splitInFaceLegsTorso(featuresArr, qualifiedArr){
    let faceFeatures = [];
    let torsoFeatures = [];
    let legsFeatures = [];

    for(var i=0; i<qualifiedArr.length; i++){
        if(faceFeatureList.includes(qualifiedArr[i])){
            faceFeatures.push(featuresArr[i]);
        }else if(torsoFeatureList.includes(qualifiedArr[i])){
            torsoFeatures.push(featuresArr[i]);
        }else if(legsFeatureList.includes(qualifiedArr[i])){
            legsFeatures.push(featuresArr[i]);
        }
    }

    return [faceFeatures, torsoFeatures, legsFeatures];
}

function affineTransformation(modelFeatures, userFeatures){
    // Handle empty features
    if(userFeatures.length < 1){
        return [[], Math.PI];
    }

    let sx = 1;
    let sy = 1;
    let tx = 0;
    let ty = 0;
    let theta = 0;

    let lr = 0.001;
    let n = userFeatures.length;
    let sumErr = 100;
    
    for(var j=0; j<1000; j++) {
    	// least square error
    	let sumErrX = 0;
    	let sumErrY = 0;
        
        
        let sxGrad = 0;
        let syGrad = 0;
        let txGrad = 0;
        let tyGrad = 0;
        let thetaGrad = 0;

        for(var i=0; i<n; i++){
            let errX = sx * Math.cos(theta) * userFeatures[i][0] - sy * Math.sin(theta) * userFeatures[i][1] + tx - modelFeatures[i][0];
            let errY = sx * Math.sin(theta) * userFeatures[i][0] + sy * Math.cos(theta) * userFeatures[i][1] + ty - modelFeatures[i][1];

            sumErrX += Math.pow(errX, 2);
            sumErrY += Math.pow(errY, 2);

            sxGrad += (Math.cos(theta) + Math.sin(theta)) * userFeatures[i][0] * (errX + errY);
            syGrad += (Math.cos(theta) - Math.sin(theta)) * userFeatures[i][1] * (errX + errY);
            txGrad += errX;
            tyGrad += errY;
            thetaGrad += (sx * userFeatures[i][0] * Math.cos(theta) - sy * userFeatures[i][1] * Math.sin(theta)) * errY -
            (sx * userFeatures[i][0] * Math.sin(theta) + sy * userFeatures[i][1] * Math.cos(theta)) * errX;
        }
        
        if(Math.abs(sumErr - (sumErrX + sumErrY)) < 0.0001) {
        	break;
       	}

        sumErr = sumErrX + sumErrY;
        
        if(sumErr < 0.001){
        	break;
        }

        sx -= lr * sxGrad;
        sy -= lr * syGrad;
        tx -= lr * txGrad;
        ty -= lr * tyGrad;
        theta -= lr * thetaGrad;
    }
    
    let affMatrix = compose(
        translate(tx, ty),
        rotate(theta),
        scale(sx, sy)
    );

    let transformedFeatures = applyToPoints(affMatrix, userFeatures);
    return [transformedFeatures, [sx, sy, theta, tx, ty]];
}

function maxDistanceAndRotation(modelFeatures, transformedFeatures, A){
    let sx = A[0];
    let sy = A[1];
    let theta = A[2];

    if(transformedFeatures.length < 2 || sx < 0 || sy < 0){
        return [1, 1, 1]; // set distance and rotations to max
    }

    // Max Euclidean Distance
    let maxDist = 0;
    let totalDist = 0;

    for(var i=0; i < modelFeatures.length; i++){
        let dist = Math.sqrt(Math.pow((modelFeatures[i][0] - transformedFeatures[i][0]), 2) + 
        Math.pow((modelFeatures[i][1] - transformedFeatures[i][1]), 2));

        totalDist += dist;

        if(dist > maxDist){
            maxDist = dist;
        }
    }

    let avgDist = totalDist / modelFeatures.length;

    // Rotations
    let rotations = Math.abs((theta * 180 / Math.PI) % 360) ;
    if (rotations > 180) {
        rotations -= 180;
    }
    rotations /= 180; // scale rotations 0 to 1

    return [maxDist, avgDist, rotations];
}

function getSimilarityScore(maxDistances, avgDistances, rotations){
    maxScore = 100;

    let faceScore = maxScore - (maxDistances.face + rotations.face) * 100;
    let torsoScore = maxScore - (maxDistances.torso + rotations.torso) * 100;
    let legsScore = maxScore - (maxDistances.legs + rotations.legs) * 100;

    return [faceScore, torsoScore, legsScore];
}

function getSimilarity(modelFeaturesObj, userFeaturesObj) {
    // remove unqualified features
    let [modelFeatures, userFeatures, qualifiedFeatures] = removeUnqualifiedKeypoints(modelFeaturesObj, userFeaturesObj);

    // check qualified features threshold
    if(qualifiedFeatures.length < minFeaturesThreshold){
        return false;
    }

    // standardize features
    let modelFeaturesScaled = standardization(modelFeatures);
    let userFeaturesScaled = standardization(userFeatures)

    // split features in 3 parts
    let [modelFace, modelTorso, modelLegs] = splitInFaceLegsTorso(modelFeaturesScaled, qualifiedFeatures);
    let [userFace, userTorso, userLegs] = splitInFaceLegsTorso(userFeaturesScaled, qualifiedFeatures);

    if(userFace.length < minFaceFeaturesThreshold || userTorso.length < minTorsoFeaturesThreshold || userLegs.length < minLegsFeaturesThreshold){
        return false;
    }

    // affine transformation
    let [transformedFace, AFace] = affineTransformation(modelFace, userFace);
    let [transformedTorso, ATorso] = affineTransformation(modelTorso, userTorso);
    let [transformedLegs, ALegs] = affineTransformation(modelLegs, userLegs);

    // max distance and rotations
    let [maxDistFace, avgDistFace, rotationFace] = maxDistanceAndRotation(modelFace, transformedFace, AFace);
    let [maxDistTorso, avgDistTorso, rotationTorso] = maxDistanceAndRotation(modelTorso, transformedTorso, ATorso);
    let [maxDistLegs, avgDistLegs, rotationLegs] = maxDistanceAndRotation(modelLegs, transformedLegs, ALegs);

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

    // similarity score
    let [faceScore, torsoScore, legsScore] = getSimilarityScore(maxDistances, avgDistances, rotations);
    let totalScore = 0.2*faceScore + 0.4*torsoScore + 0.4*legsScore;

    document.getElementById("face").innerHTML = Math.floor(faceScore);
    document.getElementById("torso").innerHTML = Math.floor(torsoScore);
    document.getElementById("legs").innerHTML = Math.floor(legsScore);
    document.getElementById("total").innerHTML = Math.floor(totalScore);

    if(totalScore >= 80){
        return true;
    }
    return false;
}

function setExampleImage() {
  // get example
  example = data[idx];
  exFilename = example.filename;
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
  drawKeypoints();
  drawSkeleton();

  if(isSimilar) {
    alert("Congratulations!!");
    isSimilar = !isSimilar;
    generateNewModel();
  }
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints()  {
  // Loop through all the poses detected
  for (let i = 0; i < poses.length; i++) {
    // For each pose detected, loop through all the keypoints
    let pose = poses[i].pose;
    for (let j = 0; j < pose.keypoints.length; j++) {
      // A keypoint is an object describing a body part (like rightArm or leftShoulder)
      let keypoint = pose.keypoints[j];
      // Only draw an ellipse is the pose probability is bigger than 0.2
      if (keypoint.score > 0.2) {
        fill(255, 0, 0);
        noStroke();
        ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
      }
    }
  }
}

// A function to draw the skeletons
function drawSkeleton() {
  // Loop through all the skeletons detected
  for (let i = 0; i < poses.length; i++) {
    let skeleton = poses[i].skeleton;
    // For every skeleton, loop through all body connections
    for (let j = 0; j < skeleton.length; j++) {
      let partA = skeleton[j][0];
      let partB = skeleton[j][1];
      stroke(255, 0, 0);
      line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
    }
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