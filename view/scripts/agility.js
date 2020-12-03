// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
PoseNet example using p5.js
=== */
let score = 0;
let diameter = 0.1 * window.innerHeight;
let example;
let video;
let poseNet;
let poses = [];
let bodyPartKeypoint = {};

let bodyParts = [
  "leftWrist",
  "rightWrist",
  "leftAnkle",
  "rightAnkle"
];

let bodyPartsColor = {
  leftWrist: '#00D506',
  rightWrist: '#0037D5',
  leftAnkle: '#FF9002',
  rightAnkle: '#E40000'
};

let idx = Math.floor(Math.random() * bodyParts.length);
let circleX = Math.random() * (window.innerWidth - 2*diameter) + diameter;
let circleY = Math.random() * (0.2*window.innerHeight - 2*diameter) + diameter + (idx > 1 ? 0.8*window.innerHeight:0);
let distanceThreshold = 0.25;

function setup() {
  createCanvas(displayWidth, displayHeight);
  video = createCapture(VIDEO);
  video.size(width, height);

  // Create a new poseNet method with a single detection
  poseNet = ml5.poseNet(video, modelReady);
  // This sets up an event that fills the global variable "poses"
  // with an array every time new poses are detected
  poseNet.on('pose', function(results) {
    poses = results;
    if(poses && poses.length > 0){
        bodyPartKeypoint = {
          score: poses[0].pose[bodyParts[idx]].confidence,
          x: poses[0].pose[bodyParts[idx]].x,
          y: poses[0].pose[bodyParts[idx]].y
        };

        let pointToEvaluate = [
          [circleX, circleY],
          [poses[0].pose[bodyParts[idx]].x, poses[0].pose[bodyParts[idx]].y],
          [0, 0],
          [displayWidth, displayHeight]
        ];
        evaluatePosition(pointToEvaluate);
    }
  });
  // Hide the video element, and just show the canvas
  video.hide();
}

function modelReady() {
  select('#status').html('');
}

function draw() {
    //move image by the width of image to the left
    translate(video.width, 0);
    //then scale it by -1 in the x-axis
    //to flip the image
    scale(-1, 1);

    image(video, 0, 0, width, height);

    // We can call both functions to draw all keypoints and the skeletons
    drawBodyPartKeypoint();
    drawTarget();
    // drawKeypoints();
    // drawSkeleton();

    //move image by the width of image to the left
    translate(video.width, 0);
    //then scale it by -1 in the x-axis
    //to flip the image
    scale(-1, 1);
    drawScore();
}

// Resize canvas
function windowResized() {
    if(displayHeight >= displayWidth){
        resizeCanvas(0, 0);
        select('#status').html('Tilt your phone to start');
    } else {
        resizeCanvas(displayWidth, displayHeight);
    }
}

// A function to draw the targets
function drawTarget() {
    fill(bodyPartsColor[bodyParts[idx]]);
    noStroke();
    circle(circleX, circleY, diameter);
}

// Function to evaluate position
function evaluatePosition(pointToEvaluate) {
    const scaledPoints = standardization(pointToEvaluate);
    const circleScaledPoints = scaledPoints[0];
    const bodyPartScaledPoints = scaledPoints[1];

    const distance = Math.sqrt(Math.pow(circleScaledPoints[0] - bodyPartScaledPoints[0], 2) +
      Math.pow(circleScaledPoints[1] - bodyPartScaledPoints[1], 2)
    );
    if(distance <= distanceThreshold){
      score++;
      generateNewPosition();
    }
}

// Draw Bodypart keypoint
function drawBodyPartKeypoint() {
    if (bodyPartKeypoint.score > 0.2) {
      fill(255, 0, 0);
      noStroke();
      ellipse(bodyPartKeypoint.x, bodyPartKeypoint.y, 25, 25);
    }
}

// Draw Score
function drawScore(){
    textSize(32);
    text(`Score: ${score}; Use your ${bodyParts[idx]}`, 10, 30);
}

function generateNewPosition() {
  idx = Math.floor(Math.random() * bodyParts.length);
  circleX = Math.random() * (window.innerWidth - 2*diameter) + diameter;
  circleY = Math.random() * (0.2*window.innerHeight - 2*diameter) + (idx > 1 ? 0.8*window.innerHeight:0);
}