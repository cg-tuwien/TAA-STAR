#pragma once

// https://stackoverflow.com/questions/400257/how-can-i-pass-a-class-member-function-as-a-callback

class RayTraceCallback {
public:
	virtual void ray_trace_callback(avk::command_buffer &cmd) {};
};