#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp> 
#include <opencv2/highgui/highgui.hpp>

#define w1 0.5f
#define w2 (1.0f-w1)

__device__ float change(uchar * d_frame_in1, uchar * d_frame_in2, int width, int width2, int we1, int we2, int y, int z, int x)
{
	float total;

	total = w1 * d_frame_in1[y * width + 3 * x + z] + w2 * d_frame_in2[y * width + 3 * x + z];

	if (total > 255) total = 255;

	return total;
}


__global__ void Plus_Kernel(uchar* d_frame_out, uchar* d_frame_in1, int height, int width, uchar* d_frame_in2, int height2, int width2, int weight1, int weight2)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	int y, z;
	float total;

	for (y = 0; y < height; y++)
	{
		for (z = 0; z < 3; z++)
		{
			total = change(d_frame_in1, d_frame_in2, width, width2, weight1, weight2, y, z, idx);

			d_frame_out[y * width + 3 * idx + z] = (uchar)total;
		}
	}

}


int main()
{

	int y, x, z;

	IplImage* Image1 = cvLoadImage("3.jpg", 1);
	IplImage* Image2 = cvLoadImage("4.jpg", 1);
	IplImage* Image3 = cvCreateImage(cvSize(Image1->widthStep, Image1->height), IPL_DEPTH_8U, 3);
	IplImage* Image4 = cvCreateImage(cvSize(Image2->widthStep, Image2->height), IPL_DEPTH_8U, 3);

	cvNamedWindow("Result", CV_WINDOW_AUTOSIZE);

	uchar* frame1 = (uchar*)calloc(Image1->imageSize, sizeof(uchar));
	uchar* frame2 = (uchar*)calloc(Image2->imageSize, sizeof(uchar));
	uchar* dis = (uchar*)calloc(Image1->imageSize, sizeof(uchar));



	//*****  transfer to input image *****//
	for (y = 0; y < (Image1->height) - 2; y++)
		for (x = 0; x < Image1->widthStep; x++)
			for (z = 0; z < 3; z++)
				frame1[y * (Image1->widthStep) + 3 * x + z] = Image1->imageData[y * Image1->widthStep + 3 * x + z];

	for (y = 0; y < (Image2->height) - 2; y++)
		for (x = 0; x < Image2->widthStep; x++)
			for (z = 0; z < 3; z++)
				frame2[y * (Image2->widthStep) + 3 * x + z] = Image2->imageData[y * Image2->widthStep + 3 * x + z];


	uchar* d_frame_in1;
	uchar* d_frame_in2;
	uchar* d_frame_out;


	cudaMalloc((void**)& d_frame_in1, sizeof(uchar) * (Image1->imageSize));
	cudaMalloc((void**)& d_frame_in2, sizeof(uchar) * (Image2->imageSize));
	cudaMalloc((void**)& d_frame_out, sizeof(uchar) * (Image1->imageSize));

	cudaMemcpy(d_frame_in1, frame1, sizeof(uchar) * (Image1->imageSize), cudaMemcpyHostToDevice);
	cudaMemcpy(d_frame_in2, frame2, sizeof(uchar) * (Image2->imageSize), cudaMemcpyHostToDevice);

	Plus_Kernel << <16, 64 >> > (d_frame_out, d_frame_in1, Image1->height, Image1->widthStep, d_frame_in2, Image2->height, Image2->widthStep, w1, w2);

	cudaMemcpy(dis, d_frame_out, sizeof(uchar) * (Image1->imageSize), cudaMemcpyDeviceToHost);

	//*****  transfer to output image *****//
	for (y = 0; y < (Image1->height) - 2; y++)
		for (x = 0; x < Image1->widthStep; x++)
			for (z = 0; z < 3; z++)
				Image1->imageData[y * Image1->widthStep + 3 * x + z] = dis[y * (Image1->widthStep) + 3 * x + z];



	cvShowImage("Result", Image1);
	cvWaitKey(0);


	free(frame1);
	free(frame2);
	free(dis);

	cudaFree(d_frame_in1);
	cudaFree(d_frame_in2);
	cudaFree(d_frame_out);
	cvDestroyWindow("Result");

}
