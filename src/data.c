#include "const.h"
#include "type.h"
#include "feature.h"
#include "image.h"
#include "classifier.h"
#include "model.h"
#include "data.h"
#include <vector>
#include "list.h"
#include "proto.h"
#include <time.h>

using namespace std;

char *fgetl(FILE *fp);


/*
���ܣ�����һ�����������е�ÿ��Ԫ�ش����ļ��е�ÿһ��
*/
List *get_lines(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) {
        fprintf(stderr, "Couldn't open file %s\n", filename);
        exit(0);
    }
    List *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}


/*
���ܣ�װ��ͼ������
������images-ͼ������Ŀ¼�ļ�
*/
Data load_image_data(char *images)
{
    Data im_data;
    List *image_list = get_lines(images);
    image *im_array = (image*)malloc(sizeof(image) * image_list->size);
    node *nd = image_list->front;
    i32 count = 0;

    while(nd){
        im_array[count] = load_image_extend((char*)nd->val);

        nd = nd->next;
        count++;
    }

    im_data.im_array = im_array;
    im_data.im_num = image_list->size;

    free_list_contents(image_list);
    free_list(image_list);
    
    return im_data;
}


/*
���ܣ�׼�����������ݣ��͸�������ͬ����������ѵ��������һֱ����
*/
void prepare_pos_examples(Data data)
{
    i32 i;
    float mean;
    float var;
    
    for(i = 0; i < data.im_num; i++)
    {
        mean = calc_im_mean(data.im_array[i]);
        var = calc_im_var(data.im_array[i], mean);
        image tmp =  normalize_integral_image(data.im_array[i], mean, var);
        free_image(data.im_array[i]);
        data.im_array[i] = tmp;
    }
}


/*
���ܣ�������������Ԥ������ѵ������
��ע�������������й�һ���������������ͼ��
    �������͸���������ͬ����������Ҫ�ü����Ӵ������Ӵ��ڽ��й�һ���ͻ���ͼ����
    ������������ֱ�Ӷ�����ͼ����У�����ü���
*/
Train_example * make_pos_example(Data data)
{
    i32 i;
    
    Train_example *examples = (Train_example*)malloc(sizeof(Train_example) * data.im_num);
    for(i = 0; i < data.im_num; i++)
    {
        examples[i].integ = data.im_array[i];
        examples[i].label = 1;
    }
    return examples;
}



/*
���ܣ��Ը���������Ԥ������ѵ������
������data-ͼ����Ϣ
     init_flag-�Ƿ��ǵ�һ�λ�ȡ����
     example_num-��Ҫ����������
     wnd_size-����ͼ���С
     m-����ģ��,����ɸѡ����������
��ע����ʼʱ���ȡ��ָ�������ĸ��������ݣ���ѵ����������Ҫ����ģ����ѡ��ָ�������ļ���������
     ��fpr���������ʣ���Сʱ���������������ѻ�ȡ����ʱ���ñ�������ͼƬ�ķ�ʽѰ��
*/
Train_example *make_neg_example(Data data, i32 init_flag, i32 example_num, i32 wnd_size, Model *model, float fpr, float scale_size, i32 step_size)
{
    i32 count = 0;
    i32 im_size = data.im_num;
    float mean;
    float var;
    i32 i, j, k;
    image cropped, candidate;
    static i32 im_idx_recorder = 0;/*��¼��һ�γ�ȡ������ͼƬ����*/
    Train_example *examples;

    //srand(1);
    if(init_flag == 1 || fpr > (1.2 * 10e-4))/*�״λ�ȡ�������߼������ʽϸ�*/
    {
        /*���������ȡ�ķ�����ȡ����������*/
        examples = (Train_example*)malloc(sizeof(Train_example) * example_num);
        while(count < example_num)
        {
            i32 im_idx = rand() % im_size;
            i32 w = data.im_array[im_idx].w;
            i32 h = data.im_array[im_idx].h;
            if(w < wnd_size || h < wnd_size)/*ͼƬ̫С*/
            {   
                continue;
           }
            i32 dx = rand()%(w - wnd_size);
            i32 dy = rand()%(h - wnd_size);
            cropped = crop_image_extend(data.im_array[im_idx], dx, dy, wnd_size, wnd_size);
            mean = calc_im_mean(cropped);
            var = calc_im_var(cropped, mean);
            candidate = normalize_integral_image(cropped, mean, var);
            free_image(cropped);
            if(var < 1 || (init_flag == 0 && 1 != model_func(model, candidate)))/*����̫С�����ݶ�ģ��û�а���*/
            {
                //free_image(cropped);
                free_image(candidate);
                continue;
            }
            examples[count].integ = candidate;
            examples[count].label = -1;
            
            count++;
        }
    }
    else/*�������ʼ���*/
    {
        /*����ɨ��ÿ�Ÿ�����ͼƬ��ȡ����������*/
        examples = collect_false_positives(data, model, example_num, wnd_size, scale_size, step_size);
    }
   
    return examples;
}


/*
���ܣ������������ϲ�
*/
Train_example *merge_pos_neg(Train_example *pos, i32 pos_num, Train_example *neg, i32 neg_num)
{
    Train_example *array = (Train_example*)malloc(sizeof(Train_example) * (pos_num + neg_num));
    memcpy(array, pos, sizeof(Train_example) * pos_num);
    memcpy(array + pos_num, neg, sizeof(Train_example) * neg_num);
    return array;
}


/*
���ܣ������ɲ��д���������ڴ�ռ�
��ע�����ڴ�ռ������Ѱ����Ѿ���׮ʱ�Ĳ��д������
*/
Feat_info **make_parallel_examples(i32 example_num, i32 feat_num)
{
    i32 i;
    Feat_info **parallel_examples = (Feat_info**)malloc(sizeof(Feat_info*) * feat_num);/*���ڲ��м��㣬����ٶ�*/

    for(i = 0; i < feat_num; i++)
    {
        parallel_examples[i] = (Feat_info*)malloc(sizeof(Feat_info) * example_num);
    }
    return parallel_examples;
}


/*
���ܣ��ͷŲ��д���������ڴ�ռ�
*/
void free_parallel_examples(Feat_info **parallel_examples, i32 feat_num)
{
    i32 i;

    for(i = 0; i < feat_num; i++)
    {
        free(parallel_examples[i]);
    }
    free(parallel_examples);
}


/*
���ܣ��ͷ�ͼƬ����
*/
void free_image_data(Data data)
{
    i32 i;

    for(i = 0; i < data.im_num; i++)
    {
        free_image(data.im_array[i]);
    }
}


/*
���ܣ��������������
*/
Train_example *collect_false_positives(Data data, Model *model, i32 example_num, i32 wnd_size, float scale_size, i32 step_size)
{
    Train_example *examples = (Train_example*)malloc(sizeof(Train_example) * example_num);
    static i32 im_idx_recorder = 0;/*��¼֮ǰ�ռ�������������ͼ������*/
    i32 count = 0;
    vector<Sub_wnd> candidate;
    i32 i;

    while(count < example_num)
    {
        if(im_idx_recorder >= data.im_num)
        {
            im_idx_recorder = 0;
        }
        image im = data.im_array[im_idx_recorder];
        scan_image_for_training(candidate, model, im, wnd_size, scale_size, step_size);
        for(i = 0; i < candidate.size(); i++)
        {
            if(count < example_num)
            {
                examples[count].integ = candidate[i].integ;
                examples[count].label = -1;
                count++;
            }
            else/*�ͷ��ռ��ĳ���ָ�����������ļ�������������ѵ�����ڻ������ᷢ������Ϊһ��ͼ������ļ���������������*/
            {
                free_image(candidate[i].integ);
            }
        }
        times("");
        printf("replenish examples %d/%d\n", count, example_num);
        candidate.resize(0);
        im_idx_recorder++;
    }   

    return examples;
} 