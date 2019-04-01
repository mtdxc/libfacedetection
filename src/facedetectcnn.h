/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


License Agreement For libfacedetection
(3-clause BSD License)

Copyright (c) 2018-2019, Shiqi Yu, all rights reserved.
shiqi.yu@gmail.com

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

* Neither the names of the copyright holders nor the names of the contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#pragma once

//#define _ENABLE_AVX2 //Please enable it if X64 CPU
//#define _ENABLE_NEON //Please enable it if ARM CPU


int * facedetect_cnn(unsigned char * result_buffer, //buffer memory for storing face detection results, !!its size must be 0x20000 Bytes!!
	unsigned char * rgb_image_data, int width, int height, int step); //input image, it must be BGR (three channels) insteed of RGB image!




//DO NOT EDIT the following code if you don't really understand it.

#if defined(_ENABLE_AVX2)
#include <immintrin.h>
#endif

#if defined(_ENABLE_NEON)
#include "arm_neon.h"
#define _ENABLE_INT8_CONV
#endif

#if defined(_ENABLE_AVX2) 
#define _MALLOC_ALIGN 256
#else
#define _MALLOC_ALIGN 128
#endif

#if defined(_ENABLE_AVX2)&& defined(_ENABLE_NEON)
#error Cannot enable the two of SSE2 AVX and NEON at the same time.
#endif


#if defined(_OPENMP)
#include <omp.h>
#endif


#include <string.h>
#include <vector>
#include <string>
using std::vector;

void* myAlloc(size_t size);
void myFree_(void* ptr);
#define myFree(ptr) (myFree_(*(ptr)), *(ptr)=0);

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

typedef struct FaceRect_
{
	float score;
	int x;
	int y;
	int w;
	int h;
}FaceRect;


class CDataBlob
{
public:
	float * data_float;
	signed char * data_int8;
	int width;
	int height;
	int channels;
	int floatChannelStepInByte;
	int int8ChannelStepInByte;
	float int8float_scale;
	bool int8_data_valid;
public:
	CDataBlob() {
		data_float = 0;
		data_int8 = 0;
		width = 0;
		height = 0;
		channels = 0;
		floatChannelStepInByte = 0;
		int8ChannelStepInByte = 0;
		int8float_scale = 1.0f;
		int8_data_valid = false;
	}
	CDataBlob(int w, int h, int c)
	{
		data_float = 0;
		data_int8 = 0;
		create(w, h, c);
	}
	~CDataBlob()
	{
		setNULL();
	}

	void setNULL();
	bool create(int w, int h, int c);

	bool setInt8DataFromCaffeFormat(signed char * pData, int dataWidth, int dataHeight, int dataChannels);
	bool setFloatDataFromCaffeFormat(float * pData, int dataWidth, int dataHeight, int dataChannels);

	bool setDataFromImage(const unsigned char * imgData, int imgWidth, int imgHeight, int imgChannels, int imgWidthStep,
		int * pChannelMean);
	bool setDataFrom3x3S2P1to1x1S1P0FromImage(const unsigned char * imgData, int imgWidth, int imgHeight, int imgChannels, int imgWidthStep,
		int * pChannelMean);
	float getElementFloat(int x, int y, int channel)
	{
		if (this->data_float)
		{
			if (x >= 0 && x < this->width &&
				y >= 0 && y < this->height &&
				channel >= 0 && channel < this->channels)
			{
				float * p = (float*)(this->data_float + (y*this->width + x)*this->floatChannelStepInByte / sizeof(float));
				return p[channel];
			}
		}

		return 0.f;
	}
	int getElementint8(int x, int y, int channel)
	{
		if (this->data_int8 && this->int8_data_valid)
		{
			if (x >= 0 && x < this->width &&
				y >= 0 && y < this->height &&
				channel >= 0 && channel < this->channels)
			{
				signed char * p = this->data_int8 + (y*this->width + x)*this->int8ChannelStepInByte / sizeof(char);
				return p[channel];
			}
		}

		return 0;
	}
	std::string str() const;
};

/*
#include <iostream>
inline std::ostream &operator<<(std::ostream &output, const CDataBlob &dataBlob)
{
	output << dataBlob.str();
	return output;
}*/

class Filters {
public:
	vector<CDataBlob *> filters;
	int pad;
	int stride;
	float scale; //element * scale = original value
};

bool convolution(CDataBlob *inputData, const Filters* filters, CDataBlob *outputData);
bool maxpooling2x2S2(const CDataBlob *inputData, CDataBlob *outputData);
bool concat4(const CDataBlob *inputData1, const CDataBlob *inputData2, const CDataBlob *inputData3, const CDataBlob *inputData4, CDataBlob *outputData);
bool scale(CDataBlob * dataBlob, float scale);
bool relu(const CDataBlob *inputOutputData);
bool priorbox(const CDataBlob * featureData, const CDataBlob * imageData, int num_sizes, float * pWinSizes, CDataBlob * outputData);
bool normalize(CDataBlob * inputOutputData, float * pScale);
bool blob2vector(const CDataBlob * inputData, CDataBlob * outputData, bool isFloat);
bool detection_output(const CDataBlob * priorbox, const CDataBlob * loc, const CDataBlob * conf, float overlap_threshold, float confidence_threshold, int top_k, int keep_top_k, CDataBlob * outputData);
/* the input data for softmax must be a vector, the data stored in a multi-channel blob with size 1x1 */
bool softmax1vector2class(const CDataBlob *inputOutputData);

vector<FaceRect> objectdetect_cnn(unsigned char * rgbImageData, int with, int height, int step);
