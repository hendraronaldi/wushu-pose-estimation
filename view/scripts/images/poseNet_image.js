let img;
let poseNetImg;
let posesImg = [];

function setup() {
    createCanvas(696, 563);

    // create an image using the p5 dom library
    // call modelReady() when it is loaded
    img = createImg('assets/images/Screen Shot 2020-05-02 at 06.05.01.png', imageReady);
    img.attribute("crossorigin", "anonymous");
    // set the image size to the size of the canvas
    img.size(width, height);

    img.hide(); // hide the image in the browser
    frameRate(1); // set the frameRate to 1 since we don't need it to be running quickly in this case
}

// when the image is ready, then load up poseNet
function imageReady(){
    // set some options
    let options = {
        imageScaleFactor: 1,
        minConfidence: 0.1
    }
    
    // assign poseNet
    poseNetImg = ml5.poseNet(modelReadyImg, options);
    // This sets up an event that listens to 'pose' events
    poseNetImg.on('pose', function (results) {
        posesImg = results;
    });
}

// when poseNet is ready, do the detection
function modelReadyImg() {
    select('#status-img').html('Model Loaded');
     
    // When the model is ready, run the singlePose() function...
    // If/When a pose is detected, poseNet.on('pose', ...) will be listening for the detection results 
    // in the draw() loop, if there are any posesImg, then carry out the draw commands
    poseNetImg.singlePose(img)
}

// draw() will not show anything until posesImg are found
function draw() {
    if (posesImg.length > 0) {
        image(img, 0, 0, width, height);
        drawSkeletonImg(posesImg);
        drawKeypointsImg(posesImg);
        noLoop(); // stop looping when the posesImg are estimated
    }

}

// The following comes from https://ml5js.org/docs/posenet-webcam
// A function to draw ellipses over the detected keypoints
function drawKeypointsImg() {
    // Loop through all the posesImg detected
    for (let i = 0; i < posesImg.length; i++) {
        // For each pose detected, loop through all the keypoints
        let pose = posesImg[i].pose;
        for (let j = 0; j < pose.keypoints.length; j++) {
            // A keypoint is an object describing a body part (like rightArm or leftShoulder)
            let keypoint = pose.keypoints[j];
            // Only draw an ellipse is the pose probability is bigger than 0.2
            if (keypoint.score > 0.2) {
                fill(255);
                stroke(20);
                strokeWeight(4);
                ellipse(round(keypoint.position.x), round(keypoint.position.y), 8, 8);
            }
        }
    }
}

// A function to draw the skeletons
function drawSkeletonImg() {
    // Loop through all the skeletons detected
    for (let i = 0; i < posesImg.length; i++) {
        let skeleton = posesImg[i].skeleton;
        // For every skeleton, loop through all body connections
        for (let j = 0; j < skeleton.length; j++) {
            let partA = skeleton[j][0];
            let partB = skeleton[j][1];
            stroke(255);
            strokeWeight(1);
            line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
        }
    }
}
