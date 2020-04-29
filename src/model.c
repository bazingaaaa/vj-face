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
#include "connectedComponent.h"

using namespace std;

/*
���ܣ�ģ�ͼ���
��ע��ͨ���ļ�����ģ��
*/
Model *load_model(const char* path)
{
	if(NULL == path)
	{
		return NULL;
	}
	char magic = 0xcc;
	Model *m = (Model*)malloc(sizeof(Model));
	FILE *fp = fopen(path, "rb");
	char ch;
	i32 i, j;

	if(NULL == fp)/*��Ч�ļ�*/
	{
		printf("modelfile doesn't exist!\n");
		return NULL;
	}
	fread(&ch, sizeof(char), 1, fp);
	if(magic != ch)/*��ģ���ļ�*/
	{
		printf("%s is not a valid modelfile!\n", path);
		return NULL;
	}

	fread(&m->stage_num, sizeof(i32), 1, fp);
	fread(&m->fpr, sizeof(double), 1, fp);
	Stage *stage = (Stage*)malloc(sizeof(Stage) * m->stage_num);
	m->head_stage = stage;

	/*��˳�����ÿ��stage*/
	for(i = 0; i < m->stage_num; i++)
	{
		i32 stump_num;
		fread(&stage[i], sizeof(Stage), 1, fp);
		Stump *stump = (Stump*)malloc(sizeof(Stump) * stage[i].stump_num);
		stage[i].head_stump = stump;
		stump_num = stage[i].stump_num;
		for(j = 0; j < stump_num; j++)
		{
			fread(&stump[j], sizeof(Stump), 1, fp);

			if(j != (stage[i].stump_num - 1))
			{
				stump[j].next_stump = &stump[j + 1];
			}
		}
		/*���һ������׮*/
		stage[i].tail_stump = &stump[j - 1];
		stump[j - 1].next_stump = NULL;
		if(i != (m->stage_num - 1))
		{
			stage[i].next_stage = &stage[i + 1];
		}
	}
	stage[i - 1].next_stage = NULL;

	printf("load model successfully\n");
	printf("initial model: stage_num:%d fpr:%.10lf\n", m->stage_num, m->fpr);

	fclose(fp);
	
	return m;
}


/*
���ܣ�����ģ��
������m-�������ģ��
	 path-ģ�ͱ����·��
��ע����ģ�ͱ������ļ�ϵͳ
*/
i8 save_model(Model *m, const char* path)
{
	char magic = 0xcc;
	FILE *fp = fopen(path, "wb");
	if(NULL == fp)/*�����ļ�*/
	{
		printf("save model error!\n");
		return -1;
	}
	/*д��magic No*/
	fwrite(&magic, sizeof(char), 1, fp);

	/*��ģ��д���ļ�*/
	i32 stage_count = 0;
	Stage *stage = m->head_stage;
	
	fwrite(&m->stage_num, sizeof(i32), 1, fp);
	fwrite(&m->fpr, sizeof(double), 1, fp);

	while(stage_count < m->stage_num)
	{
		//printf("stage_count:%d stage_num��%d\n", stage_count, m->stage_num);
		i32 stump_count = 0;
		Stump *stmp = stage->head_stump;
		fwrite(stage, sizeof(Stage), 1, fp);
		while(stump_count < stage->stump_num)
		{
			//printf("stump_count:%d stump_num:%d\n", stump_count, stage->stump_num);
			fwrite(stmp, sizeof(Stump), 1, fp);	
			stmp = stmp->next_stump;
			stump_count++;
		}
		stage = stage->next_stage;
		stage_count++;
	}
	
	fclose(fp);

	return 0;
}


/*
���ܣ�ģ���ƶϺ���
������m-ģ��
	 integ-ģ�����루����ͼ��
����ֵ��1-������
      -1-������
*/
i8 model_func(Model *m, image integ)
{
	i32 l = 0, predict_label;
	Stage *stage = m->head_stage;

	for(l = 0; l < m->stage_num; l++)
	{
		assert(stage != NULL);
		predict_label = stage_func(stage, integ, 0);
		stage = stage->next_stage;
		if(predict_label == -1)
		{
			return -1;
		}
	}

	return 1;
}


/*
���ܣ�ģ�Ͳ���
����ֵ����ȷԤ���������ռ���������ı���
*/
float test_model(Model *m, Train_example *examples, i32 example_num)
{
	i32 i;
	i32 count = 0;

	for(i = 0; i < example_num; i++)
	{
		if(examples[i].label == model_func(m, examples[i].integ))
		{
			count++;
		}
	}
	return 1.0 * count / example_num;
}


/*
���ܣ��жϴ���1�������Ƿ�λ�ڴ���2��
������
����ֵ��1-�� 0-��
*/
i32 is_inside(Sub_wnd w1, Sub_wnd w2)
{
	i32 ret = 0;
	float center_i = w1.pos_i + w1.size / 2.0;
	float center_j = w1.pos_j + w1.size / 2.0;

	if(center_i > w2.pos_i && center_i < w2.pos_i + w2.size &&
		center_j > w2.pos_j && center_j < w2.pos_j + w2.size)
	{
		ret = 1;
	}
	return ret;
}


bool cmp(Sub_wnd wnd1, Sub_wnd wnd2)
{
	return wnd1.size > wnd2.size;
}

/*
���ܣ��������Լ�ⴰ���н�һ��ɸѡ���޳����龯���ظ����
������candidate-ͨ��ģ�ͼ�������ͼ���еĺ�ѡ����
     confidence_thresh-���Ŷ����ޣ��� ��ͨ���������� / ���ڴ�С
��ע����Ӧ����An Analysis of the Viola-Jones Face Detection Algorithm�е��㷨11
     �˴��õ���connected component�㷨��Ali Rahimi�ṩ��
*/
void post_processing(vector<Sub_wnd> &candidate, i32 w, i32 h, float confidence_thresh)
{
	i32 component_num = 0;
	i32 wnd_num = candidate.size();
	i32 *in_img = (i32*)calloc(w * h, sizeof(i32));
	i32 i, j;

	/**/
	sort(candidate.begin(), candidate.end(), cmp);

	/*�ü�ⴰ��С�������ͼ����г�ʼ��*/
	for(i = 0; i < wnd_num; i++)
	{
		in_img[candidate[i].pos_i * w + candidate[i].pos_j] = candidate[i].size;
	}

	/*ִ��connected component�㷨*/
	i32 *out_img = (i32*)calloc(w * h, sizeof(i32));
	ConnectedComponents cc(30);
	component_num = cc.connected(in_img, out_img, w, h, equal_to<int>(), true);
	
	/*������ͨ����id������ʹ��ڴ�С�Ķ�Ӧ��ϵ�����㷨���ɵ���ͨ����ID�Ǵ�0��ʼ����������*/
	vector<int> cc_labels(out_img, out_img + w * h);

	/*�Ա�ǹ��Ĵ��ڽ������򣬲�ȷ��ÿ����ͨ�����Ĵ�С�Ͷ�Ӧ��ID*/
	sort(cc_labels.begin(), cc_labels.end());

	vector<int> cc_ids;/*ÿ����ͨ������ID*/
	vector<int> cc_size;/*ÿ����ͨ�����Ĵ�С*/
	cc_ids.push_back(cc_labels[0]);
	cc_size.push_back(1);
	i32 cur_cc_id = cc_labels[0];
	i32 cc_id_idx = 0;
	/*ÿ����ͨ����������һ��������Ϊ����*/
	for(i = 1; i < w * h; i++) 
	{
        if(cur_cc_id != cc_labels[i])
        {
        	cc_ids.push_back(cc_labels[i]);
			cc_size.push_back(1);
			cc_id_idx++;
			cur_cc_id = cc_labels[i];
        }
        else
		{
            cc_size[cur_cc_id]++;
		}
    }

	/*ȥ�����ŶȽϵ͵���ͨ����*/
	vector<bool> flags;
	flags.resize(component_num);
	for(i = 0; i < component_num; i++)
		flags[i] = true;
	vector<Sub_wnd> representatives;
	vector<float> confidence_tab;
	for(int k = 0; k < wnd_num; k++){
		i32 cc_id = out_img[candidate[k].pos_i * w + candidate[k].pos_j];
		if(flags[cc_id])
		{
			for(i = 0; cc_ids[i] != cc_id; i++)
			{}
			int size = cc_size[i];
			if(size >= candidate[k].size * confidence_thresh)
			{
				representatives.push_back(candidate[k]);
				confidence_tab.push_back((float)size/candidate[k].size);
			}
			flags[cc_id] = false;
		}
	}	
	
	/*���ص��Ĵ��ڽ����޳�*/
	i32 nRepresentatives = representatives.size();
	flags.resize(nRepresentatives);

	for(i = 0; i < nRepresentatives; i++)
	{
		flags[i] = true;
	}

	for(i = 0; i < nRepresentatives; i++)
	{
		for(j = i + 1; j < nRepresentatives; j++)
		{
			if(flags[j] && is_inside(representatives[i], representatives[j]))
			{
				if(confidence_tab[i] > confidence_tab[j])
				{
					flags[j] = false;
				}
				else
				{
					flags[i] = false;
					break;
				}
			}
		}
	}

	/*�ռ�ʣ��ļ�ⴰ*/
	candidate.resize(0);
	for(i = 0; i < nRepresentatives; i++)
	{
		if(flags[i])
		{
			candidate.push_back(representatives[i]);
		}
	}

	free(in_img);
	free(out_img);
}


/*
���ܣ��ͷ�ģ��ռ�õ��ڴ�
*/
void free_model(Model *model, i32 is_load_model)
{
	if(NULL == model)
	{
		return;
	}
	if(1 == is_load_model)/*�Ǽ��ص�ģ��*/
	{
		i32 i;
		Stage *stage = model->head_stage;
		for(i = 0; i < model->stage_num; i++)
		{
			free(stage->head_stump);
			stage = stage->next_stage;
		}
		free(model->head_stage);
	}
	else/*�Ǽ��ص�ģ��*/
	{
		Stage *cur_stage = model->head_stage;
		Stage *next_stage = cur_stage->next_stage;
		while(cur_stage)
		{
			Stump *cur_stump = cur_stage->head_stump;
			Stump *next_stump = cur_stump->next_stump;
			while(cur_stump)
			{
				free(cur_stump);
				cur_stump = next_stump;
				if(cur_stump != NULL)
				next_stump = cur_stump->next_stump;
			}
			free(cur_stage);
			cur_stage = next_stage;
			if(cur_stage != NULL)
			next_stage = cur_stage->next_stage;
		}
	}
	free(model);
}


/*
���ܣ����һ��ͼ�񣬲����ϼ���
������im-�����ͼ��
     model-����õ���ģ��
     skin_test_flag-�Ƿ���з�ɫ���
����ֵ�����˼�ⴰ��ͼ��
��ע����ͼ���е�Ŀ����м�⣬����ͼ���ϻ�������
*/
image run_detection(image im, Model *model, i32 skin_test_flag)
{
	image im_gray;
	i32 is_gray_image = 0;
	i32 i;
	i32 count = 0;
	vector<Sub_wnd> candidate;
	if(NULL == model)
	{
		return im;
	}
	i32 wnd_size = get_detect_wnd_size(model);

	scale_image(im, 255.0);

	/*���ͼ����󣨳��Ϳ�������512���ڣ���֤һ���ĺ��ݱȣ�����ͼ�����resize*/
 	im = constrain_image_size(im, 512);


	/*��ɫͼ��ת��Ϊ�Ҷ�ͼ���ٽ��м��*/
	if(3 == im.c)
	{
		im_gray = rgb_to_grayscale(im);
	}
	else
	{
		is_gray_image = 1;
		im_gray = copy_image(im);
	}
	
	times("scan_image begin ");
	/*ɨ������ͼ�񣬲�����ѡ��*/
 	scan_image_for_testing(candidate, model, im_gray, wnd_size, 1.5, 1);
	
	/*������һ���޳�false positive*/
	//post_processing(candidate, im_gray.w, im_gray.h, 3.0 / wnd_size);
	//postprocess(candidate);

	/*�ڱ����ͼ���ϻ�������*/
	for(i = 0; i < candidate.size(); i++)
	{
		if(skin_test_flag && !skin_test(im, candidate[i]))/*����δ��ͨ��Ƥ�����Եļ���*/
		{
			continue;
		}
		count++;
		draw_box(im, candidate[i].pos_i, candidate[i].pos_j, candidate[i].size, candidate[i].size, 0, 255, 0);
	}

	printf("detection count:%d\n", count);
	times("scan_image end ");

	free_image(im_gray);
	
	scale_image(im, 1.0/255);

	return im;
}


/*
���ܣ�ɨ��ͼ�񣨷�ѵ��ʱʹ�ã�
��ע����Ӧ����An Analysis of the Viola-Jones Face Detection Algorithm�е��㷨7
     �ú���ֻ���ڶ�ͼ��ļ�⣬ѵ��ʱ����Ҫʹ��
*/
void scan_image_for_testing(vector<Sub_wnd> &candidate, Model *model, image im, i32 wnd_size, float scale_size, i32 step_size)
{
	i32 ij;
	i32 w = im.w, h = im.h;
	i32 possibleI = (h - wnd_size) / step_size + 1;
	i32 possibleJ = (w - wnd_size) / step_size + 1;
	i32 possibleConers = possibleI * possibleJ;
		
	#pragma omp parallel for num_threads(16) schedule(static)
	for(ij = 0; ij < possibleConers; ij++)
	{
		int i = ij / possibleJ * step_size;
		int j = ij % possibleJ * step_size;
		float scale = 1;
		Sub_wnd wnd;
		wnd.pos_i = i;
		wnd.pos_j = j ;
		wnd.size = wnd_size;
		while(i + wnd.size <= h && j + wnd.size <= w)
		{
			float mean, var;
			image cropped = crop_image_extend(im, j, i, wnd.size, wnd.size);
	    	mean = calc_im_mean(cropped);
	    	var = calc_im_var(cropped, mean);
	    	if(var >= 1)
	   		{
				image integ = normalize_integral_image(cropped, mean, var);
				if(1 == model_func(model, integ))
				{
					#pragma omp critical
					{
 						candidate.push_back(wnd);
					}
				}
   				free_image(integ);
   			}
   			free_image(cropped);

   			scale *= scale_size;
			wnd.size = NEAREST_INTEGER(wnd_size * scale);
		}
	}
}



/*
���ܣ�ɨ��ͼ���Ի�ȡָ����С�ļ�������������ѵ��
��ע����Ӧ����An Analysis of the Viola-Jones Face Detection Algorithm�е��㷨7��8��9��ϵ�һ��
	 �˴�ɨ��ͼ���ռ����Ӵ�����ѵ�������Ӵ������˽�����ʹ��С����ѵ��Ҫ��
*/
void scan_image_for_training(vector<Sub_wnd> &candidate, Model *model, image im, i32 wnd_size, float scale_size, i32 step_size)
{
	i32 ij;
	i32 w = im.w, h = im.h;
	i32 possibleI = (h - wnd_size) / step_size + 1;
	i32 possibleJ = (w - wnd_size) / step_size + 1;
	i32 possibleConers = possibleI * possibleJ;

	#pragma omp parallel for num_threads(16) schedule(static)
	for(ij = 0; ij < possibleConers; ij++)
	{
		int i = ij / possibleJ * step_size;
		int j = ij % possibleJ * step_size;
		float scale = 1;
		Sub_wnd wnd;
		wnd.pos_i = i;
		wnd.pos_j = j;
		wnd.size = wnd_size;
		while(i + wnd.size <= h && j + wnd.size <= w)
		{
			float mean, var;
			image cropped = crop_image_extend(im, j, i, wnd.size, wnd.size);
	    	mean = calc_im_mean(cropped);
	    	var = calc_im_var(cropped, mean);
	    	if(var >= 1)
	   		{
				image integ = normalize_integral_image(cropped, mean, var);
				if(1 == model_func(model, integ))
				{
					if(wnd.size > wnd_size)
		            {
		                image cropped_small = down_sample(cropped, wnd_size);
		                free_image(integ);
		                integ = make_intergral_image(cropped_small);
		                free_image(cropped_small);
		                if(1 == model_func(model, integ))/*���������ٴμ�⣬�������ȻΪ�����ռ�������*/
		                {
		                    wnd.integ = integ;
			                #pragma omp critical
			                {
								candidate.push_back(wnd);
			                }
		                }
		                else
		                {
		                    free_image(integ);
		                }
		            }
		            else
		            {
		                wnd.integ = integ;
		                #pragma omp critical
		                {
							candidate.push_back(wnd);
		                }
		            }
				}
   				else
   				{
   					free_image(integ);
   				}
   			}
   			free_image(cropped);

   			scale *= scale_size;
			wnd.size = NEAREST_INTEGER(wnd_size * scale);
		}
	}
}


/*
���ܣ�ѵ������ע����ģ��
��ע����Ӧ����An Analysis of the Viola-Jones Face Detection Algorithm�е��㷨10
	 ���Գ���ѵ��
*/
Model *attentional_cascade(char *save_path, Model *model, Data t_pos_data, Data v_pos_data, Data t_neg_data, Data v_neg_data, i32 wnd_size, double fpr_overall, double fpr_perstage, double fnr_perstage)
{
	i32 l = 0;
	Stage *tail_stage = NULL;
	Stage *new_stage = NULL;
	i32 retrain_flag = 0;

	if(model != NULL)/*����ѵ��*/
	{	
		i32 stage_num = model->stage_num;
		tail_stage = model->head_stage;
		while(--stage_num > 0)
		{
			tail_stage = tail_stage->next_stage;
		}
		l = model->stage_num;
		retrain_flag = 1;
	}
	else
	{
		model = (Model*)malloc(sizeof(Model));
		model->stage_num = 0;
	}
	double fpr = 1;
	double u, s_l, T_l;
	i32 opt_case = 0;/*���ڴ���������ת*/
	i32 example_num;
	double fpr_e, fpr_g;/*�ֱ��Ӧѵ��������֤���ϵļ�������*/
	double fnr_e, fnr_g;/*�ֱ��Ӧѵ��������֤���ϵļ�������*/
	double fpr_r, fnr_r;
	Train_example *t_neg, *v_neg, *examples;
	i32 t_pos_num = t_pos_data.im_num;
	i32 v_pos_num = v_pos_data.im_num;
	i32 t_neg_num = t_pos_num;
	i32 v_neg_num = v_pos_num;
	i32 s_obesrver[2];/*���ڼ�¼tweak�Ƿ���*/
	i32 tweak_counter = 0;
	s_obesrver[0] = 0; s_obesrver[1] = 0;
	i32 count;
	i32 feat_num;
	i32 i;

	/*��ȡ��ѵ������*/
	Train_example *t_pos = make_pos_example(t_pos_data);
	Train_example *v_pos = make_pos_example(v_pos_data);

	if(retrain_flag)/*����ѵ������Ҫ����fpr���Ҹ���ѵ�����õ��ĸ�����*/
	{
		/*��ǰģ�͵�fpr*/
		fpr = model->fpr;
		/*������ǰģ���״��õ��ļ���������*/
		times("replenish examples beg\n");
		t_neg = make_neg_example(t_neg_data, 0, t_neg_num, wnd_size, model, fpr, 1.5, 1);
		v_neg = make_neg_example(v_neg_data, 0, v_neg_num, wnd_size, model, fpr, 1.5, 1);
		examples = merge_pos_neg(t_pos, t_pos_num, t_neg, t_neg_num);
		example_num = t_neg_num + t_pos_num;
		times("replenish examples end\n");
	}
	else
	{
		/*�ռ���ʼ��ѵ������֤���õĸ�����*/
		t_neg = make_neg_example(t_neg_data, 1, t_neg_num, wnd_size, NULL, 0, 0, 0);
		v_neg = make_neg_example(v_neg_data, 1, v_neg_num, wnd_size, NULL, 0, 0, 0);
		examples = merge_pos_neg(t_pos, t_pos_num, t_neg, t_neg_num);/*�����������ϵ�һ��������ȥ*/
		example_num = t_pos_num + t_neg_num;/*��������*/
	}

	/*����������Ϣ*/
	Haar_feat *feat_array = make_haar_features(wnd_size, &feat_num);
	printf("make haar features feat_num:%d\n", feat_num);

	/*����������ѵ�����������д����ڴ�ռ�*/
	Feat_info **parallel_examples = make_parallel_examples(t_pos_num + t_neg_num, feat_num);/*ѵ�����������͸���������һ��*/	
	
	times("attentional_cascade beg\n");
	while(fpr > fpr_overall)/*�������ʻ�δ���*/
	{
		switch(opt_case)
		{
			case 0:/*ѵ����ʼ����*/
				u = 0.01;
				l++;
				s_l = 0;
				T_l = 1;
			case 1:/*ѵ��adaboost*/
				new_stage = adaboost(parallel_examples, examples, example_num, t_pos_num, t_neg_num, feat_array, feat_num, T_l);
				model->stage_num++;
				if(tail_stage == NULL)
				{
					model->head_stage = new_stage;
					tail_stage = new_stage;
				}
				else
				{
					tail_stage->next_stage = new_stage;
					tail_stage = new_stage;
				}

			case 2:/*���Դ�ƫ�Ƶľ��飨ѵ�������ͷ�������֤�����������ʺͼ�������*/
				new_stage->shift = s_l;
				fpr_e = 1 - test_stage(new_stage, t_neg, t_neg_num);/*ѵ�����ļ�������*/
				fpr_g = 1 - test_stage(new_stage, v_neg, v_neg_num);/*��֤���ļ�������*/
				fnr_e = 1 - test_stage(new_stage, t_pos, t_pos_num);/*ѵ�����ļ�������*/
				fnr_g = 1 - test_stage(new_stage, v_pos, v_pos_num);/*��֤���ļ�������*/
			default:
				break;
		}
		fpr_r = MAX(fpr_e, fpr_g);
		//fpr_r = (fpr_e + fpr_g) / 2;
		fnr_r = MAX(fnr_e, fnr_g);
		//fnr_r = (fnr_e + fnr_g) / 2;

		//printf("layer:%d stump_num:%d s_l:%f u:%f\n", l, new_stage->stump_num, s_l, u);
		//printf("training set: fnr:%f fpr:%f\nvalidationset fnr:%f fpr:%f\n", fnr_e, fpr_e, fnr_g, fpr_g);

		if(fpr_r <= fpr_perstage && fnr_r <= fnr_perstage)
		{
			fpr = fpr * fpr_r;
		}
		else if(fpr_r <= fpr_perstage && fnr_r > fnr_perstage && u > 10e-5)
		{	
			s_l = s_l + u;
			s_obesrver[tweak_counter % 2] = 1;
			tweak_counter++;
			if(s_obesrver[0] + s_obesrver[1] == 0)/*�ǵ���*/
			{
				u = u * UNIT_DECAY_RATE;
				s_l = s_l - u;
				s_obesrver[tweak_counter % 2] = -1;
				tweak_counter++;
			}
			opt_case = 2;
			continue;
		}
		else if(fpr_r > fpr_perstage && fnr_r <= fnr_perstage && u > 10e-5)
		{	
			s_l = s_l - u;
			s_obesrver[tweak_counter % 2] = -1;
			tweak_counter++;
			if(s_obesrver[0] + s_obesrver[1] == 0)/*�ǵ���*/
			{
				u = u * UNIT_DECAY_RATE;
				s_l = s_l + u;
				s_obesrver[tweak_counter % 2] = 1;
				tweak_counter++;
			}
			opt_case = 2;
			continue;
		}
		else
		{
			if(T_l > MAX_DEPTH(l))/*��ȴﵽ����*/
			{
				s_l = -1;
				while(1 - fnr_r < 0.99)
				{
					s_l = s_l + u;
					new_stage->shift = s_l;
					fpr_e = 1 - test_stage(new_stage, t_neg, t_neg_num);
					fpr_g = 1 - test_stage(new_stage, v_neg, v_neg_num);
					fnr_e = 1 - test_stage(new_stage, t_pos, t_pos_num);
					fnr_g = 1 - test_stage(new_stage, v_pos, v_pos_num);
					fpr_r = MAX(fpr_e, fpr_g);
					//fpr_r = (fpr_e + fpr_g) / 2;
					fnr_r = MAX(fnr_e, fnr_g);
					//fnr_r = (fnr_e + fnr_g) / 2;
				}
				fpr = fpr * fpr_r;
				//printf("fnr:%f fpr:%f\n", fnr_r, fpr_r);
			}
			else
			{
				times("add one stump\n");
				T_l++;
				add_stump_2_stage(new_stage, parallel_examples, examples, example_num, feat_array, feat_num);
				s_l = 0;
				opt_case = 2;
				u = 0.01;
				continue;
			}
		}
		opt_case = 0;
		printf("add one stage\n");
		printf("stage:%d stump_num:%d shift:%lf unit:%lf\n", l, new_stage->stump_num, s_l, u);
		printf("current model: fpr:%.10lf\n", fpr);
		printf("current stage: fpr:%.6lf and fnr:%.6lf\n", fpr_r, fnr_r);

		if(l % 1 == 0)/*ÿһ�㱣��һ��ģ��*/
		{
			char buf[100];
			i32 len = sprintf(buf, "%s/attention_cascade_%d.cfg", save_path, l);
			buf[len] = 0;
			printf("save model stage:%d\n", l);
			model->fpr = fpr;
			save_model(model, buf);
		}
	
		times("replenish examples beg\n");
		/*�����ռ���������ѵ��������֤��*/
		for(i = 0; i < t_neg_num; i++)
		{
			free_image(t_neg[i].integ);
		}
		for(i = 0; i < v_neg_num; i++)
		{
			free_image(v_neg[i].integ);
		}
		free(t_neg);
		free(v_neg);
		free(examples);
		t_neg = make_neg_example(t_neg_data, 0, t_neg_num, wnd_size, model, fpr, 1.5, 1);
		v_neg = make_neg_example(v_neg_data, 0, v_neg_num, wnd_size, model, fpr, 1.5, 1);
		examples = merge_pos_neg(t_pos, t_pos_num, t_neg, t_neg_num);
		example_num = t_neg_num + t_pos_num;
		times("replenish examples end\n");
	}
	times("attentional_cascade end\n");
	
	/*�ͷŲ���ѵ���ڴ�ռ�*/
	free_parallel_examples(parallel_examples, feat_num);
	
	/*�ͷ�������Ϣ*/
	free(t_pos);
	free(v_pos);
	free(t_neg);
	free(v_neg);
	free(examples);
	
	return model;
}


/*
���ܣ����������Ƿ�����Ƥ��
������r-��ɫͨ������ֵ
     g-��ɫͨ������ֵ
     b-��ɫͨ������ֵ
����ֵ��1-��
       0-��
*/
bool is_skin_pixel(float r, float g, float b)
{
	int diff = max(r, max(g, b))- min(r, min(g, b));
	return r > 95 && g > 40 && b > 20 && r > g && r > b && r - g > 15 && diff > 15;
}


/*
���ܣ�������ɫ���
����ֵ��1-ͨ�����
	   0-δͨ�����
��ע��ֻ���ڴ����ɫͼƬʱ��ʹ��
*/
i8 skin_test(image src, Sub_wnd wnd)
{
	i32 counter = 0;
	i32 i_origin = wnd.pos_i;
	i32 j_origin = wnd.pos_j;
	i32 i, j;

	for(i = 0; i < wnd.size; i++)
	{
		for(j = 0; j < wnd.size; j++)
		{
			i32 i_embed = i_origin + i;
			i32 j_embed = j_origin + j;
			i32 pixel_idx = i_embed * src.w + j_embed;
			if(is_skin_pixel(src.data[pixel_idx], src.data[pixel_idx + src.w * src.h], src.data[pixel_idx + 2 * src.w * src.h]))
			{
				counter++;
			}
		}
	}
	return (float)counter / (wnd.size * wnd.size) > 0.4;
}


/*
���ܣ���ȡģ�͵ļ�ⴰ��С
��ע��ģ��ѵ��ʱ���õ�����������С�ǹ̶��ģ��˴�ֱ�ӻ�ȡģ�͵ĵ�һ������׮�е�wnd_size
*/
i32 get_detect_wnd_size(Model *model)
{
	if(NULL == model)
	{
		return 0;
	}
	return model->head_stage->head_stump->feat.src_wnd_size;
}


/*
���ܣ����м��
*/
void detect(Model *model, i32 skin_test_flag, char *infile, char *save_path)
{
	i32 web_cam = 0;
	image im;
	image im_detected;
	if(0 == strcmp(infile, "webcam"))/*����ͼ����������ͷ*/
	{
		web_cam = 1;
	}
	else
	{
		im = load_image(infile);
	}
#ifdef OPENCV
	if(web_cam)
	{
		VideoCapture camera;
		camera.open(0);
		if(!camera.isOpened())
   		{
   			fprintf(stderr, "ERROR: Could not access the camera or video!\n");
   			exit(0);
   		}
   		camera.set(CAP_PROP_FRAME_WIDTH, 512);
    	camera.set(CAP_PROP_FRAME_HEIGHT, 512);
   		while(1)
   		{
   			image im = get_image_from_stream(&camera);
			im_detected = run_detection(im, model, skin_test_flag);
			i32 key = show_image(im_detected, "webcam", 10);
			if(27 == key)/*�˳�*/
			{
				break;
			}
			free_image(im_detected);
   		}
		
	}
	else
	{	
		im_detected = run_detection(im, model, skin_test_flag);
		show_image(im_detected, "test", 100000);
	}
#else
	im_detected = run_detection(im, model, skin_test_flag);
#endif
	
	if(NULL != save_path && web_cam == 0)
	{
		save_image(im_detected, save_path);
	}
	free_image(im_detected);
}
