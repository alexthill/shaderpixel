#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec2 resolution;
    float texture_weight;
    float time;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 model;
} pcs;

layout(location = 0) in vec3 vPosition;

layout(location = 0) out vec3 fragPos;
layout(location = 1) flat out vec3 cameraPos;
layout(location = 2) flat out float cameraDistToContainer;
layout(location = 3) flat out vec2 iResolution;
layout(location = 4) flat out float iTime;

void main() {
    fragPos = vPosition;
    cameraPos = -transpose(mat3(ubo.view)) * ubo.view[3].xyz;
    // apply the inverse of the model matrix to the camera, this way the
    // container can stay the unit cube which will make calulcations nicer
    cameraPos = vec3(inverse(pcs.model) * vec4(cameraPos, 1.0));
    // assuming container is the unit cube
    cameraDistToContainer = length(max(vec3(0.0), abs(cameraPos) - 1.0));
    iResolution = ubo.resolution;
    iTime = ubo.time;

    gl_Position = ubo.proj * ubo.view * pcs.model * vec4(vPosition, 1.0);
}
