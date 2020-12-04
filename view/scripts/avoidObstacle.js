// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
PoseNet example using p5.js
=== */
let score = 0;
let diameter = 0.2 * window.innerHeight;
let example;
let video;
let poseNet;
let poses = [];

let speed = window.innerWidth / 100;
let circleX = 0;
let circleY = Math.random() * (window.innerHeight - 2*diameter) + diameter;
let distanceThreshold = 0.1;

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
        let pointToEvaluate = [
            [0, 0],
            [displayWidth, displayHeight],
            [circleX, circleY],
        ];

        poses[0].pose.keypoints.map((obj) => {
            if(obj.score > 0.2) {
                pointToEvaluate.push([obj.position.x, obj.position.y]);
            }
        });
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
    // drawBodyPartKeypoint();
    drawTarget();
    drawKeypoints();
    drawSkeleton();

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
    fill(255, 0, 0);
    noStroke();
    circle(circleX, circleY, diameter);
}

// Function to evaluate position
function evaluatePosition(pointToEvaluate) {
    const scaledPoints = standardization(pointToEvaluate);
    const circleScaledPoints = scaledPoints[2];
    const bodyPartScaledPoints = scaledPoints.slice(3, scaledPoints.length);

    let minDistance = Math.sqrt(Math.pow(circleScaledPoints[0] - bodyPartScaledPoints[0][0], 2) +
        Math.pow(circleScaledPoints[1] - bodyPartScaledPoints[0][1], 2)
    );

    for(var i=1; i < bodyPartScaledPoints.length; i++) {
        let dist = Math.sqrt(Math.pow(circleScaledPoints[0] - bodyPartScaledPoints[i][0], 2) +
            Math.pow(circleScaledPoints[1] - bodyPartScaledPoints[i][1], 2)
        );

        if(dist < minDistance) {
            minDistance = dist;
        }
    }

    if(minDistance <= distanceThreshold){
        if(score > 0){
            score--;
        }
        generateNewPosition();
    } else {
        if(circleX >= displayWidth) {
            score++;
            speed += score * 0.1;
            generateNewPosition();
        } else {
            circleX += speed;
        }
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
    text(`Score: ${score}`, 10, 30);
}

function generateNewPosition() {
    circleX = 0;
    circleY = Math.random() * (window.innerHeight - 2*diameter) + diameter;
}