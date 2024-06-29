#version 450

#extension GL_GOOGLE_include_directive : require

#include "input_structures.glsl"

layout (location = 0) in vec3 inNormal; // world space normal.
layout (location = 1) in vec3 inColor;  // vertex color.
layout (location = 2) in vec2 inUV;     // texture coordinate.

layout (location = 0) out vec4 outFragColor;

void main() 
{
    // dot(normal, sunlight direction to light)
	float lightValue = max(dot(inNormal, sceneData.sunlightDirection.xyz), 0.1f);
    // vertex color * texture color.
	vec3 color = inColor * texture(colorTex, inUV).xyz;
    // ambient light: vertex color * texture color * ambient color.
    //  not related to directional light.
	vec3 ambient = color * sceneData.ambientColor.xyz;

    // (vertex and texture) color modulated by sun direction and intensity + ambient.
    //  
    // vertex/texture color * directional light cos * sunlightintensity
    //  +
    // fixed ambient distribution.
	outFragColor = vec4(color * lightValue * sceneData.sunlightColor.w + ambient ,1.0f);
}