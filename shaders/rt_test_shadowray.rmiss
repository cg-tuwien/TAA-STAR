#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT float shadowHitValue;

void main()
{
    shadowHitValue = 0;
}
