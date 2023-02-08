#define CL_TARGET_OPENCL_VERSION 120
//#include <CL/cl_platform.h>
#include <CL/cl.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>

char *readfile(char *file_name){
  FILE *fp = fopen(file_name, "r");
  fseek(fp, 0L, SEEK_END);
  unsigned int size = ftell(fp);
	//fprintf(stderr, "%s file size = %u", file_name, size);
  fseek(fp, 0L, SEEK_SET);
  char *result = malloc(size+1);
  memset(result, 0, size+1);
  fread(result, size, 1, fp);
	result[size] = '\0';
	return result;
}

int runCL(unsigned char *result, int g_work_size, int image_width, int image_hieght){
	cl_program program[1];
	cl_kernel kernel[2];

	cl_command_queue cmd_queue;
	cl_context context;
	cl_device_id cpu = NULL, device = NULL;
	cl_int err_cpu = 0;
	cl_int err_gpu = 0;
	cl_int err = 0;
	
	size_t returned_size = 0;
	size_t buffer_size;
	cl_int cl_image_width = image_width;
	cl_int cl_image_hieght = image_hieght;
	
	cl_mem buffer;

#pragma mark Device Information
	{
		fprintf(stderr, "start Device Information\n");
		err_cpu = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &cpu, NULL);
		err_gpu = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

		if(err_gpu != CL_SUCCESS && err_cpu != CL_SUCCESS){
			fprintf(stderr, "there is no capble opencl device\n");
			exit(-1);
		}

		if(err_gpu != CL_SUCCESS && err_cpu == CL_SUCCESS){
			device = cpu;
		}

		assert(device);

		cl_char vendor_name[1024] =  {0};
		cl_char device_name[1024] =  {0};

		err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, &returned_size);		
		err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, &returned_size);		
		
		assert(err == CL_SUCCESS);
		fprintf(stderr, "connecting to %s %s\n", vendor_name, device_name);

		fprintf(stderr, "end Device Information\n");
	}
#pragma mark Context and Command cmd_queue
	{
		fprintf(stderr, "start Context and Command cmd_queue\n");
		context = clCreateContext(0, 1, &device, NULL, NULL, &err);
		assert(err == CL_SUCCESS);

		cmd_queue = clCreateCommandQueue(context, device, 0, NULL);

		fprintf(stderr, "end Context and Command cmd_queue\n");
	}

#pragma mark Program and Kernel Creation
{
		fprintf(stderr, "start Program and Kernel Creation\n");
	//const char *file_name = "./example.cl";
	const char *program_source = readfile("main.cl");//load_program_source(file_name);
	//fprintf(stderr, "%s\n", program_source);
	program[0] = clCreateProgramWithSource(context, 1, &program_source, NULL, &err);

	assert(err == CL_SUCCESS);
	fprintf(stderr, "helloooooooo\n");	

	err = clBuildProgram(program[0], 1, &device, NULL, NULL, NULL);
	
	char log[4096] = {0};
	
	clGetProgramBuildInfo(program[0], device, CL_PROGRAM_BUILD_LOG, 4096, log, &returned_size);
	fprintf(stderr, "log = \n%s\n", log);
	
	assert(err == CL_SUCCESS);

	kernel[0] = clCreateKernel(program[0], "engine", &err);

		fprintf(stderr, "end Program and Kernel Creation\n");
}

#pragma mark Memory Allocation
{

		fprintf(stderr, "start Memory Allocation\n");
	buffer_size = sizeof(char) * image_hieght * image_width * 3;
/*
	a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, NULL);
	err = clEnqueueWriteBuffer(cmd_queue, a_mem, CL_TRUE, 0, buffer_size, (void*)a, 0, NULL, NULL);

	assert(err == CL_SUCCESS);

	b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, NULL);
	err = clEnqueueWriteBuffer(cmd_queue, b_mem, CL_TRUE, 0, buffer_size, (void*)b, 0, NULL, NULL);

	assert(err == CL_SUCCESS);
	n_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned int), NULL, NULL);
	err = clEnqueueWriteBuffer(cmd_queue, n_mem, CL_TRUE, 0, sizeof(unsigned int), (void*)&n, 0, NULL, NULL);

	assert(err == CL_SUCCESS);
*/
	
	buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, &err);

	assert(err == CL_SUCCESS);

		fprintf(stderr, "end Memory Allocation\n");
}
#pragma mark Kernel Arguments
{
		fprintf(stderr, "start Kernel Arguments\n");
	err = clSetKernelArg(kernel[0], 0, sizeof(cl_int), &cl_image_width);
	err = clSetKernelArg(kernel[0], 1, sizeof(cl_int), &cl_image_hieght);
	err = clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &buffer);
	//err = clSetKernelArg(kernel[0], 3, sizeof(cl_mem), &n_mem);
	assert(err == CL_SUCCESS);
		fprintf(stderr, "end Kernel Arguments\n");
	
}

#pragma mark Execution and Read
{	
	fprintf(stderr, "start Execution and Read\n");
	
	size_t global_work_size = g_work_size;
	//for(unsigned int i=0; i<300; i++){
	
	struct timeval stop, start;
	gettimeofday(&start, NULL);
	
	err = clEnqueueNDRangeKernel(cmd_queue, kernel[0], 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

	gettimeofday(&stop, NULL);
	fprintf(stderr, "took %lu ms\n", (stop.tv_sec - start.tv_sec) * 1000 + (stop.tv_usec - start.tv_usec) / 1000); 
	fprintf(stderr, "took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec)); 
	
	assert(err == CL_SUCCESS);
	//}
	clFinish(cmd_queue);

	err = clEnqueueReadBuffer(cmd_queue, buffer, CL_TRUE, 0, buffer_size, result, 0, NULL, NULL);
	
	assert(err == CL_SUCCESS);
	clFinish(cmd_queue);

	fprintf(stderr, "end Execution and Read\n");
}
#pragma mark Teardown
{
	fprintf(stderr, "start Teardown\n");
	//clReleaseMemObject(a_mem);
	//clReleaseMemObject(b_mem);
	clReleaseMemObject(buffer);
	
	clReleaseCommandQueue(cmd_queue);
	clReleaseContext(context);
	fprintf(stderr, "end Teardown\n");
}
return CL_SUCCESS;
}

int main(){
	int image_width = 2000;
	int image_hieght = 2000;
	unsigned char *result = malloc(sizeof(char) * image_hieght * image_width * 3);
	memset(result, 0, sizeof(char) * image_hieght * image_width * 3);

	runCL(result, image_width, image_width, image_hieght);

    printf("%s%d%c%d%s", "P3\n", image_width, ' ', image_hieght, "\n255\n");

	for(int i=image_hieght-1; i>=0; i--){
		//fprintf(stderr, "i = %u", i);
		for(unsigned int j=0; j<image_width; j++)
			printf("%u %u %u\n",result[i*image_width*3+j*3], result[i*image_width*3+j*3+1], result[i*image_width*3+j*3+2]);
	}

	free(result);
				

	return 0;
}
