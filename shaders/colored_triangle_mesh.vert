#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 outUV;

// buffer reference doesn't need descriptor set.
struct Vertex {
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 color;
};

// VertexBuffer is used from buffer address.
//  CRUCIAL: this isn't descriptor set. This specify how shader parses the pointer passed into the push constant.
//  
layout(buffer_reference, scalar) readonly buffer VertexBuffer { 
	Vertex vertices[];
};

//push constants block
layout( push_constant ) uniform constants
{	
	mat4 render_matrix;
    // buffer reference, an address.
	VertexBuffer vertexBuffer;
} PushConstants;

void main() 
{	
	//load vertex data from device adress
    //  gl_VertexIndex is the vertex index from the index buffer for the current vertex.
	Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];

	//output vertex data
	gl_Position = PushConstants.render_matrix * vec4(v.position, 1.0f);
	outColor = v.color.xyz;
	outUV.x = v.uv_x;
	outUV.y = v.uv_y;
}