let img;
let poseNet;
let poses = [];
let idx = 0;
let totalModel = 121;

let userFeaturesObj;
let modelValid = false;
let validModel = [];
let invalidModel = [];

function setupImage() {
    img = createImg('assets/images/model/model'+idx+'.png', imageReady);
    // set the image size to the size of the canvas
    img.size(width, height);

    img.hide(); // hide the image in the browser
    // frameRate(1); // set the frameRate to 1 since we don't need it to be running quickly in this case
}

function isModelValid(userFeaturesObj) {
    // remove unqualified features
    let [_, userFeatures, _not_used, qualifiedFeatures] = removeUnqualifiedKeypoints(userFeaturesObj, userFeaturesObj);

    // check qualified features threshold
    if(qualifiedFeatures.length < minFeaturesThreshold){
        return false;
    }

    // standardize features
    let userFeaturesScaled = standardization(userFeatures)

    // split features in 3 parts
    let [userFace, userTorso, userLegs] = splitInFaceLegsTorso(userFeaturesScaled, qualifiedFeatures);

    if(userFace.length < minFaceFeaturesThreshold || userTorso.length < minTorsoFeaturesThreshold || userLegs.length < minLegsFeaturesThreshold){
        return false;
    }

    return true;
}

function setup() {
    createCanvas(360, 360);

    // create an image using the p5 dom library
    // call modelReady() when it is loaded
    setupImage()
}

// when the image is ready, then load up poseNet
function imageReady(){
    // set some options
    let options = {
        architecture: 'ResNet50',
        outputStride: 32,
        inputResolution: { width: 257, height: 200 },
        quantBytes: 2
    };
    
    // assign poseNet
    poseNet = ml5.poseNet(modelReady, options);
    // This sets up an event that listens to 'pose' events
    poseNet.on('pose', function (results) {
        poses = results;
        
        if(poses && poses.length > 0) {
            userFeaturesObj = {
                score: poses[0].pose.score,
                keypoints: poses[0].pose.keypoints
            };
            modelValid = isModelValid(userFeaturesObj);
        }

        if(modelValid){
            userFeaturesObj.filename = 'assets/images/model/model'+idx+'.png';
            validModel.push(userFeaturesObj);
            modelValid = false;
        }else{
            invalidModel.push(userFeaturesObj);
        }

        idx++;
        if(idx<totalModel){
            // setupImage()
            img = createImg('assets/images/model/model'+idx+'.png', modelReady);
            // set the image size to the size of the canvas
            img.size(width, height);
    
            img.hide(); 
        } else {
            // print valid model
            console.log(JSON.stringify(validModel));
        }
    });
}

// when poseNet is ready, do the detection
function modelReady() {
    select('#status').html('Model Loaded');
     
    // When the model is ready, run the singlePose() function...
    // If/When a pose is detected, poseNet.on('pose', ...) will be listening for the detection results 
    // in the draw() loop, if there are any poses, then carry out the draw commands
    poseNet.singlePose(img)
}

// draw() will not show anything until poses are found
function draw() {
    if (poses.length > 0) {
        image(img, 0, 0, width, height);
        drawSkeleton(poses);
        drawKeypoints(poses);
        noLoop(); // stop looping when the poses are estimated
    }

}

// The following comes from https://ml5js.org/docs/posenet-webcam
// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
    // Loop through all the poses detected
    for (let i = 0; i < poses.length; i++) {
        // For each pose detected, loop through all the keypoints
        let pose = poses[i].pose;
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
function drawSkeleton() {
    // Loop through all the skeletons detected
    for (let i = 0; i < poses.length; i++) {
        let skeleton = poses[i].skeleton;
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