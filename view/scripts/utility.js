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

    let faceScore = maxScore - (maxDistances.face + rotations.face) * 50;
    let torsoScore = maxScore - (maxDistances.torso + rotations.torso) * 50;
    let legsScore = maxScore - (maxDistances.legs + rotations.legs) * 50;

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

    if(totalScore >= 95){
        return true;
    }
    return false;
}