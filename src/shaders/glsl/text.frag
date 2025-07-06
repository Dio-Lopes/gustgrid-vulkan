#version 450
layout (location = 0) in vec2 fragTexCoord;
layout(binding = 0) uniform sampler2D fontTexture;
layout(push_constant) uniform PushConstants {
    vec3 textColor;
} pc;
layout (location = 0) out vec4 outColor;

void main(){
    float alpha = texture(fontTexture, fragTexCoord).r;
    outColor = vec4(pc.textColor, alpha);
    if(alpha < 0.01) discard;
}