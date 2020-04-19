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
功能：模型加载
备注：通过文件加载模型
*/
Model *load_model(const char* path)
{
	char magic = 0xcc;
	Model *m = (Model*)malloc(sizeof(Model));
	FILE *fp = fopen(path, "rb");
	char ch;
	i32 i, j;

	if(NULL == fp)/*错误文件*/
	{
		printf("modelfile doesn't exist!\n");
		return NULL;
	}
	fread(&ch, sizeof(char), 1, fp);
	if(magic != ch)
	{
		printf("%s is not a valid cfg file!\n", path);
		return NULL;
	}

	fread(&m->stage_num, sizeof(i32), 1, fp);
	Stage *stage = (Stage*)malloc(sizeof(Stage) * m->stage_num);
	m->head_stage = stage;

	for(i = 0; i < m->stage_num; i++)
	{
		fread(&stage[i], sizeof(Stage), 1, fp);
		Stump *stump = (Stump*)malloc(sizeof(Stump) * stage[i].stump_num);
		stage[i].head_stump = stump;
		for(j = 0; j < stage[i].stump_num; j++)
		{
			fread(&stump[j], sizeof(Stump), 1, fp);
			if(j != (stage[i].stump_num - 1))
			{
				stump[j].next_stump = &stump[j + 1];
			}
		}
		/*最后一个决策桩*/
		stage[i].tail_stump = &stump[j];
		stump[j].next_stump = NULL;
		if(i != (m->stage_num - 1))
		{
			stage[i].next_stage = &stage[i + 1];
		}
	}
	stage[i].next_stage = NULL;

	fclose(fp);
	
	return m;
}


/*
功能：保存模型
参数：m-待保存的模型
	 path-模型保存的路径
备注：将模型保存至文件系统
*/
i8 save_model(Model *m, const char* path)
{
	char magic = 0xcc;
	FILE *fp = fopen(path, "wb");
	if(NULL == fp)/*错误文件*/
	{
		printf("save model error!\n");
		return -1;
	}
	/*写入magic No*/
	fwrite(&magic, sizeof(char), 1, fp);

	/*将模型写入文件*/
	i32 stage_count = 0;
	Stage *stage = m->head_stage;

	fwrite(&m->stage_num, sizeof(i32), 1, fp);
	while(stage_count < m->stage_num)
	{
		//printf("stage_count:%d stage_num：%d\n", stage_count, m->stage_num);
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
功能：模型推断函数
参数：m-模型
	 integ-模型输入（积分图像）
返回值：1-正样本
      -1-负样本
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
功能：模型测试
返回值：正确预测的样本数占样本总数的比率
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
功能：判断窗口1的中心是否位于窗口2中
参数：
返回值：1-是 0-否
*/
i32 is_inside(Sub_wnd w1, Sub_wnd w2)
{
	i32 ret = 0;
	float center_i = w1.pos_i + w1.size / 2.0;
	float center_j = w1.pos_j + w1.size / 2.0;

	if(center_i > w2.pos_i && center_i < w2.pos_i + w2.size &&
		center_i > w2.pos_i && center_i < w2.pos_i + w2.size)
	{
		ret = 1;
	}
	return ret;
}


/*
功能：检测后处理，对检测窗进行进一步筛选，剔除掉虚警和重复检测
参数：candidate-通过模型检测出来的图像中的候选窗口
     confidence_thresh-置信度门限，即 连通分量的数量 / 窗口大小
备注：对应论文An Analysis of the Viola-Jones Face Detection Algorithm中的算法11
     此处用到了connected component算法（Ali Rahimi提供）
*/
void post_processing(vector<Sub_wnd> &candidate, i32 w, i32 h, float confidence_thresh)
{
	i32 component_num = 0;
	i32 wnd_num = candidate.size();
	i32 *in_img = (i32*)calloc(w * h, sizeof(i32));
	i32 i, j;

	/*用检测窗大小对输入的图像进行初始化*/
	for(i = 0; i < wnd_num; i++)
	{
		in_img[candidate[i].pos_i * w + candidate[i].pos_j] = candidate[i].size;
	}

	/*执行connected component算法*/
	i32 *out_img = (i32*)calloc(w * h, sizeof(i32));
	ConnectedComponents cc(30);
	component_num = cc.connected(in_img, out_img, w, h, equal_to<int>(), true);
	
	/*建立连通分量id，坐标和窗口大小的对应关系，有算法生成的连通分量ID是从0开始的连续整数*/
	vector<int> cc_labels(out_img, out_img + w * h);

	/*对标记过的窗口进行排序，并确定每个连通分量的大小和对应的ID*/
	sort(cc_labels.begin(), cc_labels.end());

	vector<int> cc_ids;/*每种连通分量的ID*/
	vector<int> cc_size;/*每种连通分量的大小*/
	cc_ids.push_back(cc_labels[0]);
	cc_size.push_back(1);
	i32 cur_cc_id = cc_labels[0];
	i32 cc_id_idx = 0;
	/*每个连通分量仅保留一个窗口作为代表*/
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

	/*去掉置信度较低的连通分量*/
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
	
	/*对重叠的窗口进行剔除*/
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

	/*收集剩余的检测窗*/
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
功能：释放模型占用的内存
*/
void free_model(Model *model, i32 is_load_model)
{
	if(NULL == model)
	{
		return;
	}
	if(1 == is_load_model)/*是加载的模型*/
	{
		i32 i;
		Stage *stage = model->head_stage;
		for(i = 0; i < model->stage_num; i++)
		{
			free(stage->head_stump);
			stage = stage->next_stage;
		}
		free(stage);
	}
	else/*非加载的模型*/
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
功能：检测一副图像，并画上检测框
参数：im-待检测图像
     model-检测用到的模型
     skin_test_flag-是否进行肤色检测
     savepath-图像检测后的保存路径
返回值：检测框数目
备注：对图像中的目标进行检测，并在图像上画出检测框
*/
i32 run_detection(image im, Model *model, i32 skin_test_flag, char *savepath)
{
	image im_gray;
	i32 is_gray_image = 0;
	i32 i;
	i32 count = 0;
	vector<Sub_wnd> candidate;
	if(NULL == model)
	{
		return 0;
	}
	i32 wnd_size = get_detect_wnd_size(model);

	/*如果图像过大（长和宽限制在512以内，保证一定的横纵比），对图像进行resize*/
 	im = constrain_image_size(im, 512);

	/*彩色图像转换为灰度图像再进行检测*/
	if(3 == im.c)
	{
		im_gray = rgb_to_grayscale(im);
	}
	else
	{
		is_gray_image = 1;
		im_gray = im;
	}
	
	times("scan_image begin ");
	/*扫描整个图像，产生候选窗*/
 	scan_image_for_testing(candidate, model, im_gray, wnd_size, 1.5, 1);
	
	/*后处理，进一步剔除false positive*/
	post_processing(candidate, im_gray.w, im_gray.h, 3.0 / wnd_size);

	/*在被检测图像上画出检测框*/
	for(i = 0; i < candidate.size(); i++)
	{
		if(skin_test_flag && !skin_test(im, candidate[i]))/*舍弃未能通过皮肤测试的检测框*/
		{
			continue;
		}
		count++;
		draw_box(im, candidate[i].pos_i, candidate[i].pos_j, candidate[i].size, candidate[i].size, 0, 255, 0);
	}

	printf("detection count:%d\n", count);
	times("scan_image end ");
	

	scale_image(im, 1.0/255);
#ifdef OPENCV
	show_image(im, "test", 500000);
#endif

	if(NULL != savepath)
	{
		save_image(im, savepath);
	}

	free_image(im_gray);
	if(!is_gray_image)
	{
		free_image(im);
	}
	return count;
}


/*
功能：扫描图像（非训练时使用）
备注：对应论文An Analysis of the Viola-Jones Face Detection Algorithm中的算法7
     该函数只用于对图像的检测，训练时不需要使用
*/
void scan_image_for_testing(vector<Sub_wnd> &candidate, Model *model, image im, i32 wnd_size, float scale_size, i32 step_size)
{
	i32 ij;
	i32 w = im.w, h = im.h;
	i32 possibleI = (h - wnd_size) / step_size + 1;
	i32 possibleJ = (w - wnd_size) / step_size + 1;
	i32 possibleConers = possibleI * possibleJ;
		
	#pragma omp parallel for 
	for(ij = 0; ij < possibleConers; ij++)
	{
		int i = ij / possibleJ;
		int j = ij % possibleJ;
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
功能：扫描图像以获取指定大小的假阳性样本用于训练
备注：对应论文An Analysis of the Viola-Jones Face Detection Algorithm中的算法7，8，9结合到一起
	 此处扫描图像收集的子窗用于训练，对子窗进行了降采样使大小满足训练要求。
*/
void scan_image_for_training(vector<Sub_wnd> &candidate, Model *model, image im, i32 wnd_size, float scale_size, i32 step_size)
{
	i32 ij;
	i32 w = im.w, h = im.h;
	i32 possibleI = (h - wnd_size) / step_size + 1;
	i32 possibleJ = (w - wnd_size) / step_size + 1;
	i32 possibleConers = possibleI * possibleJ;

	#pragma omp parallel for 
	for(ij = 0; ij < possibleConers; ij++)
	{
		int i = ij / possibleJ;
		int j = ij % possibleJ;
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
		                if(1 == model_func(model, integ))/*降采样后再次检测，若检测依然为正则收集该样本*/
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
功能：训练级联注意力模型
备注：对应论文An Analysis of the Viola-Jones Face Detection Algorithm中的算法10
	 可以持续训练
*/
Model *attentional_cascade(char *save_path, Model *model, Data t_pos_data, Data v_pos_data, Data t_neg_data, Data v_neg_data, i32 wnd_size, float fpr_overall, float fpr_perstage, float fnr_perstage)
{
	i32 l = 0;
	Stage *tail_stage = NULL;
	Stage *new_stage = NULL;
	i32 retrain_flag = 0;

	if(model != NULL)/*继续训练*/
	{
		tail_stage = model->head_stage;
		while(tail_stage->next_stage != NULL)
		{
			tail_stage = tail_stage->next_stage;
		}
		l = model->stage_num;
		retrain_flag = 1;
	}
	else
	{
		model = (Model*)malloc(sizeof(Model));
	}
	float fpr = 1;
	float u, s_l, T_l;
	i32 opt_case = 0;/*用于代码间进行跳转*/
	i32 example_num;
	float fpr_e, fpr_g;/*分别对应训练集和验证集上的假阳性率*/
	float fnr_e, fnr_g;/*分别对应训练集和验证集上的假阴性率*/
	float fpr_r, fnr_r;
	i32 t_pos_num = t_pos_data.im_num;
	i32 v_pos_num = v_pos_data.im_num;
	i32 t_neg_num = t_pos_num;
	i32 v_neg_num = v_pos_num;
	i32 s_obesrver[2];/*用于记录tweak是否震荡*/
	i32 tweak_counter = 0;
	s_obesrver[0] = 0; s_obesrver[1] = 0;
	i32 count;
	i32 feat_num;
	i32 i;
	

	/*获取正训练样本*/
	Train_example *t_pos = make_pos_example(t_pos_data);
	//printf("make train pos example finish\n");	
	Train_example *v_pos = make_pos_example(v_pos_data);
	//printf("make valid pos example finish\n");	

	/*收集初始的训练和验证所用的负样本*/
	Train_example *t_neg = make_neg_example(t_neg_data, 1, t_neg_num, wnd_size, NULL, 0, 0, 0);
	Train_example *v_neg = make_neg_example(v_neg_data, 1, v_neg_num, wnd_size, NULL, 0, 0, 0);
	Train_example *examples = merge_pos_neg(t_pos, t_pos_num, t_neg, t_neg_num);/*把正负样本合到一个数组中去*/
	example_num = t_pos_num + t_neg_num;/*所有样本*/
	

	if(retrain_flag)/*继续训练，需要评估fpr并且更换训练中用到的负样本*/
	{
		float fpr_t = 1 - test_model(model, t_neg, t_neg_num);/*训练集的假阳性率*/
		float fpr_v = 1 - test_model(model, v_neg, v_neg_num);/*训练集的假阳性率*/
		/*评估当前模型的fpr*/
		fpr = max(fpr_t, fpr_v);
		printf("initial model: stage_num:%d fpr:%f\n", model->stage_num, fpr);
		/*更换当前模型首次用到的假阳性样本*/
		times("replenish examples beg\n");
		/*重新收集负样本的训练集和验证集*/
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

	/*生成特征信息*/
	Haar_feat *feat_array = make_haar_features(wnd_size, &feat_num);
	printf("make haar features feat_num:%d\n", feat_num);

	/*创建可用于训练的样本并行处理内存空间*/
	Feat_info **parallel_examples = make_parallel_examples(t_pos_num + t_neg_num, feat_num);/*训练中正样本和负样本数量一致*/	

	model->stage_num = 0;
	
	times("attentional_cascade beg\n");
	while(fpr > fpr_overall)/*假阳性率还未达标*/
	{
		switch(opt_case)
		{
			case 0:/*训练初始参数*/
				u = 0.01;
				l++;
				s_l = 0;
				T_l = 1;
			case 1:/*训练adaboost*/
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

			case 2:/*测试带偏移的经验（训练集）和泛化（验证集）假阳性率和假阴性率*/
				new_stage->shift = s_l;
				fpr_e = 1 - test_stage(new_stage, t_neg, t_neg_num);/*训练集的假阳性率*/
				fpr_g = 1 - test_stage(new_stage, v_neg, v_neg_num);/*验证集的假阳性率*/
				fnr_e = 1 - test_stage(new_stage, t_pos, t_pos_num);/*训练集的假阴性率*/
				fnr_g = 1 - test_stage(new_stage, v_pos, v_pos_num);/*验证集的假阴性率*/
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
			if(s_obesrver[0] + s_obesrver[1] == 0)/*非单调*/
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
			if(s_obesrver[0] + s_obesrver[1] == 0)/*非单调*/
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
			if(T_l > MAX_DEPTH(l))/*深度达到上限*/
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
		printf("stage:%d stump_num:%d shift:%f unit:%f\n", l, new_stage->stump_num, s_l, u);
		printf("current model: fpr:%f\n", fpr);
		printf("current stage: fnr:%f and fpr:%f\n", fpr_r, fnr_r);

		if(l % 1 == 0)/*每一层保存一次模型*/
		{
			char buf[100];
			i32 len = sprintf(buf, "%s/attentional_cascade_%d.cfg", save_path, l);
			buf[len] = 0;
			printf("save model stage:%d\n", l);
			save_model(model, buf);
		}
	
		times("replenish examples beg\n");
		/*重新收集负样本的训练集和验证集*/
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
	
	/*释放并行训练内存空间*/
	free_parallel_examples(parallel_examples, feat_num);
	
	/*释放样本信息*/
	free(t_pos);
	free(v_pos);
	free(t_neg);
	free(v_neg);
	free(examples);
	
	return model;
}


/*
功能：检查该像素是否属于皮肤
参数：r-红色通道像素值
     g-绿色通道像素值
     b-蓝色通道像素值
返回值：1-是
       0-否
*/
bool is_skin_pixel(float r, float g, float b)
{
	int diff = max(r, max(g, b))- min(r, min(g, b));
	return r > 95 && g > 40 && b > 20 && r > g && r > b && r - g > 15 && diff > 15;
}


/*
功能：人脸肤色检查
返回值：1-通过检查
	   0-未通过检查
备注：只能在处理彩色图片时候使用
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
功能：获取模型的检测窗大小
备注：模型训练时所用到的正样本大小是固定的，此处直接获取模型的第一个决策桩中的wnd_size
*/
i32 get_detect_wnd_size(Model *model)
{
	if(NULL == model)
	{
		return 0;
	}
	return model->head_stage->head_stump->feat.src_wnd_size;
}
