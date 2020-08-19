#version 460
#extension GL_GOOGLE_include_directive : enable
// -------------------------------------------------------

// ###### VERTEX SHADER/PIPELINE INPUT DATA ##############
// Several vertex attributes (These are the buffers passed
// to command_buffer_t::draw_indexed in the same order):
layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec2 aTexCoords;

// ###### DATA PASSED ON ALONG THE PIPELINE ##############
// Data from vert -> tesc or frag:
layout (location = 0) out VertexData {
	vec2 texCoords;
} v_out;
// -------------------------------------------------------

// ###### VERTEX SHADER MAIN #############################
void main()
{
	v_out.texCoords = aTexCoords;
	gl_Position = vec4(aPosition, 1.0);
}
// -------------------------------------------------------

