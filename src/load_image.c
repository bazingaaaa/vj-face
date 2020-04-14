// You probably don't want to edit this file
#include <stdio.h>
#include <stdlib.h>

#include "image.h"


image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = (float*)calloc(h*w*c, sizeof(float));
    return out;
}

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void save_image_stb(image im, const char *name, int png)
{
    char buff[256];
    unsigned char *data = (unsigned char*)calloc(im.w*im.h*im.c, sizeof(char));
    int i,k;
    for(k = 0; k < im.c; ++k){
        for(i = 0; i < im.w*im.h; ++i){
            data[i*im.c+k] = (unsigned char) roundf((255*im.data[i + k*im.w*im.h]));
        }
    }
    int success = 0;
    if(png){
        sprintf(buff, "%s.png", name);
        success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
    } else {
        sprintf(buff, "%s.jpg", name);
        success = stbi_write_jpg(buff, im.w, im.h, im.c, data, 100);
    }
    free(data);
    if(!success) fprintf(stderr, "Failed to write image %s\n", buff);
}

void save_png(image im, const char *name)
{
    save_image_stb(im, name, 1);
}

void save_image(image im, const char *name)
{
    save_image_stb(im, name, 0);
}

// 
// Load an image using stb
// channels = [0..4]
// channels > 0 forces the image to have that many channels
//
image load_image_stb(char *filename, int channels, int norm_flag)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n",
            filename, stbi_failure_reason());
        exit(0);
    }
    if (channels) c = channels;
    int i,j,k;
    image im = make_image(w, h, c);
    if(norm_flag == 1)
    {
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int dst_index = i + w*j + w*h*k;
                    int src_index = k + c*i + c*w*j;
                    im.data[dst_index] = (float)data[src_index]/255.;
                }
            }
        }
    }
    else
    {
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int dst_index = i + w*j + w*h*k;
                    int src_index = k + c*i + c*w*j;
                    im.data[dst_index] = data[src_index];
                }
            }
        } 
    }
    //We don't like alpha channels, #YOLO
    if(im.c == 4) im.c = 3;
    free(data);
    return im;
} 

image load_image(char *filename, int norm_flag)
{
    image out = load_image_stb(filename, 0, norm_flag);
    return out;
}

void free_image(image im)
{
    free(im.data);
}

#ifdef OPENCV


Mat image_to_mat(image im)
{
    image copy = copy_image(im);
    constrain_image(copy);
    if(im.c == 3) rgbgr_image(copy);
    IplImage *ipl = image_to_ipl(copy);
    Mat m = cvarrToMat(ipl, true);
    cvReleaseImage(&ipl);
    free_image(copy);
    return m;
}

void rgbgr_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
}


image mat_to_image(Mat &src)
{
    int h = src.rows;
    int w = src.cols;
    int c = src.channels();
    int i, j, k;
    image im = make_image(w, h, c);

    for(i = 0; i < h; ++i){
        const uchar* inData = src.ptr<uchar>(i);  
        for(j = 0; j < w; ++j){
            for(k = 0; k < c; ++k){
                im.data[k*w*h + i*w + j] = inData[j * c + k]/255.;
            }
        }
    }
    return im;
}


IplImage *image_to_ipl(image im)
{
    int x,y,c;
    IplImage *disp = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(c= 0; c < im.c; ++c){
                float val = im.data[c*im.h*im.w + y*im.w + x];
                disp->imageData[y*step + x*im.c + c] = (unsigned char)(val*255);
            }
        }
    }
    return disp;
}


image get_image_from_stream(VideoCapture *cap)
{
    //IplImage* src = cvQueryFrame(cap);
    Mat src;
    *cap >> src;
    if (src.empty()) return make_empty_image(0,0,0);
    image im = mat_to_image(src);
    rgbgr_image(im);
    return im;
}


int show_image(image im, const char* name, int ms)
{
    Mat m = image_to_mat(im);
    imshow(name, m);
    int c = waitKey(ms);
    if (c != -1) c = c%256;
    return c;
}

#endif
