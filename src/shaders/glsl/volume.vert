#version 450

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

layout(location = 0) out vec3 FragPos;
layout(location = 1) out vec3 texCoord;

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

void main(){
    vec4 worldPosition = ubo.model * vec4(aPos, 1.0);
    FragPos = worldPosition.xyz;
    texCoord = (FragPos - ubo.worldMin) / (ubo.worldMax - ubo.worldMin);
    gl_Position = ubo.projection * ubo.view * worldPosition;
}