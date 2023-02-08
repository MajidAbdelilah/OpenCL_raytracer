#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include <time.h>


const char *kernel_source =
"__kernel void life(                                                     \n"
"   constant bool*input,                                                 \n"
"   global bool* output,                                                 \n"
"   const unsigned int height,                                           \n"
"   const unsigned int width)                                            \n"
"{                                                                       \n"
"   int i = get_global_id(0);                                            \n"
"   int rowUp = i - width;                                               \n"
"   int rowDown = i + width;                                             \n"
"   bool outOfBounds = (i < width);                                      \n"
"   outOfBounds |= (i > (width * (height-1)));                           \n"
"   outOfBounds |= (i % width == 0);                                     \n"
"   outOfBounds |= (i % width == width-1);                               \n"
"   if (outOfBounds) { output[i] = false; return; }                      \n"
"   int neighbours = input[rowUp-1] + input[rowUp] + input[rowUp+1];     \n"
"   neighbours += input[i-1] + input[i+1];                               \n"
"   neighbours += input[rowDown-1] + input[rowDown] + input[rowDown+1];  \n"
"   if (neighbours == 3 || (input[i] && neighbours == 2))                \n"
"       output[i] = true;                                                \n"
"   else                                                                 \n"
"       output[i] = false;                                               \n"
"}                                                                       \n";

void fail(const char *message)
{
    fprintf(stderr, "%s\n", message);
    exit(1);
}

cl_kernel createKernelFromSource(cl_device_id device_id, cl_context context,
        const char *source, const char *name)
{
    int err;
    // Load the source
    cl_program program = clCreateProgramWithSource(context, 1, &source , NULL,
            &err);
    if (err != CL_SUCCESS) { fail("Unable to create program"); }

    // Compile it.
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                          sizeof(buffer), buffer, &len);
        fprintf(stderr, "%s\n", buffer);
        fail("Unable to build program");
    }
    // Load it.
    cl_kernel kernel = clCreateKernel(program, name, &err);
    if (!kernel || err != CL_SUCCESS) { fail("Unable to create kernel"); }
    clReleaseProgram(program);

    return kernel;
}

// The board
static const int width = 128;
static const int height = 64;
static const size_t board_size = width * height;
static _Bool board[128*64];
// Storage for the board.
static cl_mem input;
static cl_mem output;
// OpenCL state
static cl_command_queue queue;
static cl_kernel kernel;
static cl_device_id device_id;
static cl_context context;

void printBoard(void)
{
    unsigned i = 0;

    for (unsigned y=0 ; y<height ; y++)
    {
        for (unsigned x=0 ; x<width ; x++)
        {
            putc(board[i++] ? '*' : ' ', stdout);
        }
        puts("");
    }
    puts("\n");
}

void createQueue(void)
{
    int err;
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);
    if (err != CL_SUCCESS) { fail("Unable to enumerate device IDs"); }

    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context) { fail("Unable to create context"); }

    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (!queue) { fail("Unable to create command queue"); }
}

void prepareKernel(void)
{
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(board), NULL,
            NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(board), NULL,
            NULL);
    if (!input || !output) { fail("Unable to create buffers"); }

    int err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &height);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &width);
    if (err != CL_SUCCESS) { fail("Unable to set arguments"); }
}

void runGame(int iterations)
{
    if (iterations == 0)  { return; }
    int err;
    size_t workgroup_size;
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE,
                                              sizeof(size_t), &workgroup_size, NULL);
    if (err != CL_SUCCESS) { fail("Unable to get kernel work-group size"); }

    // Send the board to the OpenCL stack
    err = clEnqueueWriteBuffer(queue, input, CL_TRUE, 0,
                                   sizeof(board), board, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fail("Unable to enqueue buffer"); }

    for (unsigned int i=0 ; i<iterations ; i++)
    {
        // Run the kernel on every cell in the board
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &board_size,
                &workgroup_size, 0, NULL, NULL);
        if (err) { fail("Unable to enqueue kernel"); }
        if (i < iterations - 1)
        {
            // Copy the output to the input for the next iteration
            err = clEnqueueCopyBuffer(queue, output, input, 0, 0,
                    sizeof(board), 0, NULL, NULL);
            if (err) { fail("Unable to enqueue copy"); }
        }
    }

    err = clEnqueueReadBuffer(queue, output, CL_TRUE, 0,
                                sizeof(board), board, 0, NULL, NULL );
    if (err != CL_SUCCESS) { fail("Unable to read results"); }
}

int main(int argc, char** argv)
{
    srand(time(NULL));
    for(unsigned int i=0 ; i<board_size ; i++)
        board[i] = rand() > (INT_MAX / 2);

    createQueue();

    kernel = createKernelFromSource(device_id, context,
                    kernel_source, "life");

    prepareKernel();

    printBoard();
    int iterations = 0;
    do
    {
        printf("Run for how many iterations (0 to exit)? ");
        scanf("%d", &iterations);
        runGame(iterations);
        printBoard();
    } while(iterations > 0);

    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
