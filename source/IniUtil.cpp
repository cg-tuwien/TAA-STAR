#include "IniUtil.h"

// TODO: check if floats always use "." as comma when using to_string
void iniWriteString(mINI::INIStructure &ini, const std::string &section, const std::string &name, const std::string &val)	{ ini[section][name] = val;	}
void iniWriteBool  (mINI::INIStructure &ini, const std::string &section, const std::string &name, const bool &val)			{ ini[section][name] = val ? "1" : "0";	}
void iniWriteBool32(mINI::INIStructure &ini, const std::string &section, const std::string &name, const uint32_t &val)		{ ini[section][name] = std::to_string(val);	}
void iniWriteInt   (mINI::INIStructure &ini, const std::string &section, const std::string &name, const int &val)			{ ini[section][name] = std::to_string(val);	}
void iniWriteFloat (mINI::INIStructure &ini, const std::string &section, const std::string &name, const float &val)			{ ini[section][name] = std::to_string(val);	}
void iniWriteVec2  (mINI::INIStructure &ini, const std::string &section, const std::string &name, const glm::vec2 &val)		{
	iniWriteFloat(ini, section, name + ".x", val.x);
	iniWriteFloat(ini, section, name + ".y", val.y);
}
void iniWriteIVec2 (mINI::INIStructure &ini, const std::string &section, const std::string &name, const glm::ivec2 &val)		{
	iniWriteInt(ini, section, name + ".x", val.x);
	iniWriteInt(ini, section, name + ".y", val.y);
}
void iniWriteVec3  (mINI::INIStructure &ini, const std::string &section, const std::string &name, const glm::vec3 &val)		{
	iniWriteFloat(ini, section, name + ".x", val.x);
	iniWriteFloat(ini, section, name + ".y", val.y);
	iniWriteFloat(ini, section, name + ".z", val.z);
}
void iniWriteIVec3 (mINI::INIStructure &ini, const std::string &section, const std::string &name, const glm::ivec3 &val)		{
	iniWriteInt(ini, section, name + ".x", val.x);
	iniWriteInt(ini, section, name + ".y", val.y);
	iniWriteInt(ini, section, name + ".z", val.z);
}
void iniWriteVec4  (mINI::INIStructure &ini, const std::string &section, const std::string &name, const glm::vec4 &val)		{
	iniWriteFloat(ini, section, name + ".x", val.x);
	iniWriteFloat(ini, section, name + ".y", val.y);
	iniWriteFloat(ini, section, name + ".z", val.z);
	iniWriteFloat(ini, section, name + ".w", val.w);
}
void iniWriteIVec4 (mINI::INIStructure &ini, const std::string &section, const std::string &name, const glm::ivec4 &val)		{
	iniWriteInt(ini, section, name + ".x", val.x);
	iniWriteInt(ini, section, name + ".y", val.y);
	iniWriteInt(ini, section, name + ".z", val.z);
	iniWriteInt(ini, section, name + ".w", val.w);
}
void iniWriteQuat  (mINI::INIStructure &ini, const std::string &section, const std::string &name, const glm::quat &val)		{
	iniWriteFloat(ini, section, name + ".x", val.x);
	iniWriteFloat(ini, section, name + ".y", val.y);
	iniWriteFloat(ini, section, name + ".z", val.z);
	iniWriteFloat(ini, section, name + ".w", val.w);
}

void iniReadString(mINI::INIStructure &ini, const std::string &section, const std::string &name, std::string &val)			{ if (ini[section][name] != "") val = ini[section][name]; }
void iniReadBool  (mINI::INIStructure &ini, const std::string &section, const std::string &name, bool &val)					{ if (ini[section][name] != "") val = (ini[section][name] == "1") || (ini[section][name] == "true"); }
void iniReadBool32(mINI::INIStructure &ini, const std::string &section, const std::string &name, uint32_t &val)				{ if (ini[section][name] != "") val = std::stoul(ini[section][name]); }
void iniReadInt   (mINI::INIStructure &ini, const std::string &section, const std::string &name, int &val)					{ if (ini[section][name] != "") val = std::stol(ini[section][name]); }
void iniReadInt   (mINI::INIStructure &ini, const std::string &section, const std::string &name, uint32_t &val)				{ if (ini[section][name] != "") val = std::stoul(ini[section][name]); }
void iniReadFloat (mINI::INIStructure &ini, const std::string &section, const std::string &name, float &val)				{ if (ini[section][name] != "") val = std::stof(ini[section][name]); }
void iniReadVec2  (mINI::INIStructure &ini, const std::string &section, const std::string &name, glm::vec2 &val)			{
	iniReadFloat(ini, section, name + ".x", val.x);
	iniReadFloat(ini, section, name + ".y", val.y);
}
void iniReadIVec2 (mINI::INIStructure &ini, const std::string &section, const std::string &name, glm::ivec2 &val)			{
	iniReadInt(ini, section, name + ".x", val.x);
	iniReadInt(ini, section, name + ".y", val.y);
}
void iniReadVec3  (mINI::INIStructure &ini, const std::string &section, const std::string &name, glm::vec3 &val)			{
	iniReadFloat(ini, section, name + ".x", val.x);
	iniReadFloat(ini, section, name + ".y", val.y);
	iniReadFloat(ini, section, name + ".z", val.z);
}
void iniReadIVec3 (mINI::INIStructure &ini, const std::string &section, const std::string &name, glm::ivec3 &val)			{
	iniReadInt(ini, section, name + ".x", val.x);
	iniReadInt(ini, section, name + ".y", val.y);
	iniReadInt(ini, section, name + ".z", val.z);
}
void iniReadVec4  (mINI::INIStructure &ini, const std::string &section, const std::string &name, glm::vec4 &val)			{
	iniReadFloat(ini, section, name + ".x", val.x);
	iniReadFloat(ini, section, name + ".y", val.y);
	iniReadFloat(ini, section, name + ".z", val.z);
	iniReadFloat(ini, section, name + ".w", val.w);
}
void iniReadIVec4 (mINI::INIStructure &ini, const std::string &section, const std::string &name, glm::ivec4 &val)			{
	iniReadInt(ini, section, name + ".x", val.x);
	iniReadInt(ini, section, name + ".y", val.y);
	iniReadInt(ini, section, name + ".z", val.z);
	iniReadInt(ini, section, name + ".w", val.w);
}
void iniReadQuat  (mINI::INIStructure &ini, const std::string &section, const std::string &name, glm::quat &val)			{
	iniReadFloat(ini, section, name + ".x", val.x);
	iniReadFloat(ini, section, name + ".y", val.y);
	iniReadFloat(ini, section, name + ".z", val.z);
	iniReadFloat(ini, section, name + ".w", val.w);
}


//#define DEF_INI_DO(NAME,TYPE) \
//void iniDo ## NAME (bool write, mINI::INIStructure & ini, const std::string & section, const std::string &name, TYPE &val) { \
//	if (write)	\
//		iniWrite ## NAME(ini, section, name, val);	\
//	else	\
//		iniRead ## NAME(ini, section, name, val);	\
//}
//
//
//DEF_INI_DO(String, std::string)

