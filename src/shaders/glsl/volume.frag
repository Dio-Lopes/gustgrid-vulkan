#version 450

layout(location = 0) in vec3 FragPos;
layout(location = 1) in vec3 texCoord;
layout(location = 0) out vec4 FragColor;

layout(binding = 0, std140) uniform VolumeUniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec3 camPos;
    vec3 gridSize;
    vec3 worldMin;
    vec3 worldMax;
    int displayPressure;
    float stepSize;
} ubo;

layout(binding = 1) uniform sampler3D volumeTex;
layout(binding = 2) uniform sampler3D temperatureTex;

vec3 thermalColorMap(float normalizedTemp){
    vec3 color = vec3(0.0);
    if(normalizedTemp < 0.167){
        float t = normalizedTemp / 0.167;
        color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t);
    } else if(normalizedTemp < 0.333){
        float t = (normalizedTemp - 0.167) / (0.333 - 0.167);
        color = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), t);
    } else if(normalizedTemp < 0.5){
        float t = (normalizedTemp - 0.333) / (0.5 - 0.333);
        color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), t);
    } else if(normalizedTemp < 0.667){
        float t = (normalizedTemp - 0.5) / (0.667 - 0.5);
        color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), t);
    } else if(normalizedTemp < 0.833){
        float t = (normalizedTemp - 0.667) / (0.833 - 0.667);
        color = mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0), t);
    } else{
        float t = (normalizedTemp - 0.833) / (1.0 - 0.833);
        color = mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 0.5, 0.5), t);
    }
    return color;
}
float thermalAlpha(float normalizedTemp, float volumeVal){
    float tempSensitivity = 0.8;
    float baseAlpha = pow(normalizedTemp, 0.7) * tempSensitivity;
    if(normalizedTemp > 0.6) baseAlpha *= 1.5;
    if(normalizedTemp < 0.2) baseAlpha = max(baseAlpha, 0.05);
    return clamp(baseAlpha * 0.005, 0.0, 1.0);
}
void main(){
    vec3 rayDir = normalize(FragPos - ubo.camPos);
    float accumAlpha = 0.0;
    vec3 accumColor = vec3(0.0);

    vec3 worldSize = ubo.worldMax - ubo.worldMin;
    vec3 voxelSize = worldSize / ubo.gridSize;

    float maxVoxelDim = max(max(voxelSize.x, voxelSize.y), voxelSize.z);
    float baseWorldStep = maxVoxelDim * ubo.stepSize;
    float worldStep = baseWorldStep;
    int emptySteps = 0;

    vec3 currentWorldPos = FragPos;
    vec3 currentTexCoord = texCoord;

    int maxSteps = int(length(worldSize) / baseWorldStep) + 1;

    float opacity = ubo.displayPressure == 1 ? 0.05 : 0.001;

    float ambientTemp = 22.0;
    float maxExpectedTemp = 100.0;
    float tempRange = maxExpectedTemp - ambientTemp;

    for(int i = 0; i < maxSteps; i++){
        if(any(lessThan(currentTexCoord, vec3(0.0))) || any(greaterThan(currentTexCoord, vec3(1.0)))) break;
        
        float volumeVal = texture(volumeTex, currentTexCoord).r;
        float tempValue = texture(temperatureTex, currentTexCoord).r;
        float tempThreshold = ambientTemp + 0.5;
        if(volumeVal < 1e-4 && tempValue < tempThreshold){
            emptySteps++;
            if(emptySteps > 2.0) worldStep = min(baseWorldStep * 2.0, baseWorldStep * (1.0 + float(emptySteps - 2) * 0.5));
            currentWorldPos += rayDir * worldStep;
            currentTexCoord = (currentWorldPos - ubo.worldMin) / (ubo.worldMax - ubo.worldMin);
            if(any(lessThan(currentTexCoord, vec3(-0.1))) || any(greaterThan(currentTexCoord, vec3(1.1)))) break;
            continue;
        } else{
            emptySteps = 0;
            worldStep = baseWorldStep;
        }

        float normalizedTemp = clamp((tempValue - ambientTemp) / tempRange, 0.0, 1.0);
        vec3 thermalColor = thermalColorMap(normalizedTemp);
        float thermalAlphaVal = thermalAlpha(normalizedTemp, volumeVal);
        float volumeAlpha = clamp(volumeVal * ubo.stepSize * opacity, 0.0, 1.0);
        float totalAlpha = max(volumeAlpha, thermalAlphaVal);
        
        float alphaToAdd = totalAlpha * (1.0 - accumAlpha);
        accumAlpha += alphaToAdd;
        accumColor += thermalColor * alphaToAdd;
        
        if(accumAlpha > 0.95) break;
        
        currentWorldPos += rayDir * worldStep;
        currentTexCoord = (currentWorldPos - ubo.worldMin) / (ubo.worldMax - ubo.worldMin);
    }
    FragColor = vec4(vec3(accumColor), accumAlpha);
}