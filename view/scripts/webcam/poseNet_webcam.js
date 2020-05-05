// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
poseNetWebcam example using p5.js
=== */

let video;
let poseNetWebcam;
let posesWebcam = [];

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.size(width, height);

  // Create a new poseNetWebcam method with a single detection
  poseNetWebcam = ml5.poseNet(video, modelReadyWCam);
  // This sets up an event that fills the global variable "posesWebcam"
  // with an array every time new posesWebcam are detected
  poseNetWebcam.on('pose', function(results) {
    posesWebcam = results;
    // console.log(results)
  });
  // Hide the video element, and just show the canvas
  video.hide();
}

function modelReadyWCam() {
  select('#status-wcam').html('Model Loaded');
}

function draw() {
  image(video, 0, 0, width, height);

  // We can call both functions to draw all keypoints and the skeletons
  drawKeypointsWcam();
  drawSkeletonWcam();
}

// A function to draw ellipses over the detected keypoints
function drawKeypointsWcam()  {
  // Loop through all the posesWebcam detected
  for (let i = 0; i < posesWebcam.length; i++) {
    // For each pose detected, loop through all the keypoints
    let pose = posesWebcam[i].pose;
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
function drawSkeletonWcam() {
  // Loop through all the skeletons detected
  for (let i = 0; i < posesWebcam.length; i++) {
    let skeleton = posesWebcam[i].skeleton;
    // For every skeleton, loop through all body connections
    for (let j = 0; j < skeleton.length; j++) {
      let partA = skeleton[j][0];
      let partB = skeleton[j][1];
      stroke(255, 0, 0);
      line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
    }
  }
}