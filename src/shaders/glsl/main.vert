#version 450

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 cameraPos;
} ubo;

layout(location = 0) out vec3 FragPos;
layout(location = 1) out vec3 normalVec;
layout(location = 2) out vec2 texCoord;
layout(location = 3) out vec3 camPos;

void main(){
    normalVec = normalize(mat3(transpose(inverse(ubo.model))) * aNormal);
    texCoord = aTexCoord;
    FragPos = vec3(ubo.model * vec4(aPos, 1.0));
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(aPos, 1.0);
    camPos = ubo.cameraPos;
}