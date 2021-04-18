#pragma once

// https://stackoverflow.com/questions/400257/how-can-i-pass-a-class-member-function-as-a-callback

class RayTraceCallback {
public:
	virtual void ray_trace_callback(avk::command_buffer &cmd) = 0;
	virtual int  getNumRayTraceSamples() = 0;
	virtual void setNumRayTraceSamples(int aNumSamples) = 0;
	virtual bool getRayTraceAugmentTaaDebug() = 0;
	virtual void setRayTraceAugmentTaaDebug(bool aDebug) = 0;
};