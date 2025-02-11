#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    float texture_weight;
    float time;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 model;
} pcs;

layout(location = 0) in vec3 vPosition;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 cameraPos;
layout(location = 2) out float cameraDistToContainer;

void main() {
    fragPos = vPosition;
    cameraPos = -transpose(mat3(ubo.view)) * ubo.view[3].xyz;
    // apply the inverse of the model matrix to the camera, this way the
    // conatiner can stay the unit cube which will make calulcations nicer
    cameraPos = vec3(inverse(pcs.model) * vec4(cameraPos, 1.0));
    // assuming container is the unit cube
    cameraDistToContainer = length(max(vec3(0.0), abs(cameraPos) - 1.0));

    gl_Position = ubo.proj * ubo.view * pcs.model * vec4(vPosition, 1.0);
}
