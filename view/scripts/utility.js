let confidenceLevelThreshold = 0;
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
];

let faceFeatureList = [
    "Nose",
    "leftEye",
    "rightEye",
    "leftEar",
    "rightEar"
];

let torsoFeatureList = [
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist"
];

let legsFeatureList = [
    "leftHip",
    "rightHip",
    "leftKnee",
    "rightKnee",
    "leftAnkle",
    "rightAnkle"
];

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

function normalization(features){
    // normalize
    const {normalize} = ml.preprocessing;
    return normalize(features);
}

function removeUnqualifiedKeypoints(modelFeaturesObj, userFeaturesObj){
    let qualifiedFeatures = [];
    let modelArr = [];
    let modelConfArr = [];
    let userArr = [];

    for(var i=0; i<featureList.length; i++){
        if(!(modelFeaturesObj.keypoints[i].score < confidenceLevelThreshold ||
            userFeaturesObj.keypoints[i].score < confidenceLevelThreshold)){
            
                qualifiedFeatures.push(featureList[i]);

                modelArr.push([modelFeaturesObj.keypoints[i].position.x, modelFeaturesObj.keypoints[i].position.y]);
                userArr.push([userFeaturesObj.keypoints[i].position.x, userFeaturesObj.keypoints[i].position.y]);

                modelConfArr.push(modelFeaturesObj.keypoints[i].score);
        }
    }
    return [modelArr, userArr, modelConfArr, qualifiedFeatures];
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

function maxDistanceAndRotation(modelFeatures, transformedFeatures, A, confidenceScores){
    let sx = A[0];
    let sy = A[1];
    let theta = A[2];
    let totalConfidenceScore = 0;

    if(transformedFeatures.length < 2 || sx < 0 || sy < 0){
        return [1, 1, 1]; // set distance and rotations to max
    }

    // Max Euclidean Distance
    let maxDist = 0;
    let totalDist = 0;

    for(var i=0; i < modelFeatures.length; i++){
        let dist = confidenceScores[i] * Math.sqrt(Math.pow((modelFeatures[i][0] - transformedFeatures[i][0]), 2) + 
        Math.pow((modelFeatures[i][1] - transformedFeatures[i][1]), 2));
        totalConfidenceScore += confidenceScores[i];

        totalDist += dist;

        if(dist > maxDist){
            maxDist = dist;
        }
    }

    let avgDist = totalDist / totalConfidenceScore;

    // Rotations
    let rotations = Math.abs((theta * 180 / Math.PI) % 360) ;
    if (rotations > 180) {
        rotations -= 180;
    }
    rotations /= 180; // scale rotations 0 to 1

    return [maxDist, avgDist, rotations, totalConfidenceScore];
}

function getSimilarityScore(maxDistances, avgDistances, rotations){
    maxScore = 100;

    let faceScore = maxScore - (maxDistances.face + rotations.face) * 50;
    let torsoScore = maxScore - (maxDistances.torso + rotations.torso) * 50;
    let legsScore = maxScore - (maxDistances.legs + rotations.legs) * 50;

    return [faceScore, torsoScore, legsScore];
}

function getCosineSimilarity(A, B){
    A = normalization(A);
    B = normalization(B);
    var dotproduct = 0;
    var mA = 0;
    var mB = 0;
    for(var i = 0; i < A.length; i++){
        dotproduct += (A[i][0] * B[i][0]);
        mA += (A[i][0]*A[i][0]);
        mB += (B[i][0]*B[i][0]);

        dotproduct += (A[i][1] * B[i][1]);
        mA += (A[i][1]*A[i][1]);
        mB += (B[i][1]*B[i][1]);
    }
    mA = Math.sqrt(mA);
    mB = Math.sqrt(mB);
    var similarity = (dotproduct)/(mA*mB);
    return similarity;
}

function getBodyPartCosineSimilarity(modelFace, modelTorso, modelLegs, userFace, userTorso, userLegs, modelConfArr){
    let model = [...modelFace, ...modelTorso, ...modelLegs];
    let user = [...userFace, ...userTorso, ...userLegs];

    let cosineSimilarity = 0;
    let confidenceScores = 0;

    let degToCheck = [
        ['leftElbow', 'leftWrist', 'leftShoulder'],
        ['rightElbow', 'rightWrist', 'rightShoulder'],
        ['leftShoulder', 'leftElbow', 'leftHip'],
        ['rightShoulder', 'rightElbow', 'rightHip'],
        ['leftKnee', 'leftAnkle', 'leftHip'],
        ['rightKnee', 'rightAnkle', 'rightHip'],
        ['leftHip', 'leftKnee', 'leftShoulder'],
        ['rightHip', 'rightKnee', 'rightShoulder'],
    ];

    degToCheck.map((arr) => {
        let modelPart = [
            [
                [model[featureList.indexOf(arr[1])][0] - model[featureList.indexOf(arr[0])][0],
                model[featureList.indexOf(arr[1])][1] - model[featureList.indexOf(arr[0])][1]]
            ],
            [
                [model[featureList.indexOf(arr[2])][0] - model[featureList.indexOf(arr[0])][0],
                model[featureList.indexOf(arr[2])][1] - model[featureList.indexOf(arr[0])][1]]
            ]
        ];

        let userPart = [
            [
                [user[featureList.indexOf(arr[1])][0] - user[featureList.indexOf(arr[0])][0],
                user[featureList.indexOf(arr[1])][1] - user[featureList.indexOf(arr[0])][1]]
            ],
            [
                [user[featureList.indexOf(arr[2])][0] - user[featureList.indexOf(arr[0])][0],
                user[featureList.indexOf(arr[2])][1] - user[featureList.indexOf(arr[0])][1]]
            ]
        ];

        let modelCosine = getCosineSimilarity(modelPart[0], modelPart[1]);
        let userCosine = getCosineSimilarity(userPart[0], userPart[1]);
        let confidenceScore = (modelConfArr.indexOf(arr[0]) + modelConfArr.indexOf(arr[1]) + modelConfArr.indexOf(arr[2]))/3;

        cosineSimilarity += confidenceScore * Math.abs(modelCosine - userCosine);
        confidenceScores += confidenceScore;
    })

    return cosineSimilarity / confidenceScores;
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