#pragma once

#include <mini/ini.h>
#include <glm/glm.hpp>


void iniWriteString(mINI::INIStructure & ini, const std::string & section, const std::string & name, const std::string & val);
void iniWriteBool  (mINI::INIStructure & ini, const std::string & section, const std::string & name, const bool & val);
void iniWriteInt   (mINI::INIStructure & ini, const std::string & section, const std::string & name, const int & val);
void iniWriteFloat (mINI::INIStructure & ini, const std::string & section, const std::string & name, const float & val);
void iniWriteVec2  (mINI::INIStructure & ini, const std::string & section, const std::string & name, const glm::vec2 & val);
void iniWriteIVec2 (mINI::INIStructure & ini, const std::string & section, const std::string & name, const glm::ivec2 & val);
void iniWriteVec3  (mINI::INIStructure & ini, const std::string & section, const std::string & name, const glm::vec3 & val);
void iniWriteIVec3 (mINI::INIStructure & ini, const std::string & section, const std::string & name, const glm::ivec3 & val);
void iniWriteVec4  (mINI::INIStructure & ini, const std::string & section, const std::string & name, const glm::vec4 & val);
void iniWriteQuat  (mINI::INIStructure & ini, const std::string & section, const std::string & name, const glm::quat & val);

void iniReadString (mINI::INIStructure & ini, const std::string & section, const std::string & name, std::string & val);
void iniReadBool   (mINI::INIStructure & ini, const std::string & section, const std::string & name, bool & val);
void iniReadInt    (mINI::INIStructure & ini, const std::string & section, const std::string & name, int & val);
void iniReadInt    (mINI::INIStructure & ini, const std::string & section, const std::string & name, uint32_t & val);
void iniReadFloat  (mINI::INIStructure & ini, const std::string & section, const std::string & name, float & val);
void iniReadVec2   (mINI::INIStructure & ini, const std::string & section, const std::string & name, glm::vec2 & val);
void iniReadIVec2  (mINI::INIStructure & ini, const std::string & section, const std::string & name, glm::ivec2 & val);
void iniReadVec3   (mINI::INIStructure & ini, const std::string & section, const std::string & name, glm::vec3 & val);
void iniReadIVec3  (mINI::INIStructure & ini, const std::string & section, const std::string & name, glm::ivec3 & val);
void iniReadVec4   (mINI::INIStructure & ini, const std::string & section, const std::string & name, glm::vec4 & val);
void iniReadQuat   (mINI::INIStructure & ini, const std::string & section, const std::string & name, glm::quat & val);

//void iniDoString   (bool write, mINI::INIStructure & ini, const std::string & section, const std::string & name, std::string & val);

