#pragma once
#ifndef FACE_H
#define FACE_H

#include <vector>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

struct profile 
{
	char name[100];
	vector<vector<double>> weights;
};
class FACE 
{
public:
	FACE(void);
	string LoadDataFromDisk(string Path);
	string SaveDataToDisk(void);
	string LoadPack(string Path);


private:
	vector<profile> profiles;
	

	
		
	
	
	
};

#endif