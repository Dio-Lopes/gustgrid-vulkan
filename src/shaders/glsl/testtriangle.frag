#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec3 fragViewPos;

layout(binding = 1) uniform sampler2D baseColorTexture;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 lightDirection = normalize(vec3(-1.0, -1.0, -1.0));
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    float lightIntensity = 1.0;
    
    vec3 baseColor = texture(baseColorTexture, fragTexCoord).rgb;
    vec3 normal = normalize(fragNormal);
    vec3 viewDir = normalize(fragViewPos - fragWorldPos);
    vec3 lightDir = -lightDirection;
    
    vec3 ambient = 0.1 * baseColor;
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor * baseColor * lightIntensity;
    
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 64.0);
    vec3 specular = spec * lightColor * lightIntensity * 0.5;
    
    vec3 result = ambient + diffuse + specular;
    
    outColor = vec4(result, 1.0);
}