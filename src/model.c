#include "const.h"
#include "type.h"
#include "feature.h"
#include "image.h"
#include "classifier.h"
#include "model.h"
#include "data.h"
#include "proto.h"
#include "list.h"


#define MAX_DEPTH(l) (MIN((10 * l) + 10, 200))
//#define MAX_DEPTH(l) (1)

#define UNIT_DECAY_RATE (0.8)

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
		printf("%s doesn't exist!\n", path);
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
功能：收集训练集和验证集中的假阳性样本，用于下一轮的训练
参数：data-样本图像数据
	 examples-用于收集假阳性样本的数组
	 example_num-收集样本的数量
备注：对应论文An Analysis of the Viola-Jones Face Detection Algorithm中的算法7
	 出于内存的限制，实现上有些改动
*/
void collect_fp_example(Data data, Train_example *examples, i32 example_num, i32 wnd_size, i32 scale_mutiplier)
{
	i32 i, j, k, l, m;
	i32 scan_wnd;
	i32 count = 0;
	float mean, var;
	image im_wnd, integ;

	while(1)
	{
		scan_wnd = wnd_size * pow(scale_mutiplier, i);
		for(i = 0; i < data.im_num; i++)
		{
			/*扫描整个图像*/
			for(j = 0; j <= data.im_array[i].w - scan_wnd; j++)
			{
				for(k = 0; k <= data.im_array[i].h - scan_wnd; k++)
				{
					im_wnd = crop_image_extend(data.im_array[i], j, k, scan_wnd, scan_wnd);
					mean = calc_im_mean(im_wnd);
					var = calc_im_var(im_wnd, mean);
					if(var < 1)
					{
						continue;
					}
					integ = normalize_integral_image(im_wnd, mean, var);
				}
			}
		}
		i++;
	}
}


/*
功能：模型推断函数
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
功能：训练级联注意力模型
备注：对应论文An Analysis of the Viola-Jones Face Detection Algorithm中的算法10
*/
Model *attentional_cascade(Data t_pos_data, Data v_pos_data, Data t_neg_data, Data v_neg_data, i32 wnd_size, float fpr_overall, float fpr_perlayer, float fnr_perlayer)
{
	Model *model = (Model*)malloc(sizeof(Model));
	Stage *tail_stage = NULL;
	Stage *new_stage = NULL;
	i32 l = 0;
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
	prepare_pos_examples(t_pos_data);
	prepare_pos_examples(v_pos_data);
	Train_example *t_pos = make_pos_example(t_pos_data);
	printf("make train pos example finish\n");	
	Train_example *v_pos = make_pos_example(v_pos_data);
	printf("make valid pos example finish\n");	

	/*收集初始的训练和验证所用的负样本*/
	Train_example *t_neg = make_neg_example(t_neg_data, 1, t_neg_num, wnd_size, NULL);
	Train_example *v_neg = make_neg_example(v_neg_data, 1, v_neg_num, wnd_size, NULL);
	Train_example *examples = merge_pos_neg(t_pos, t_pos_num, t_neg, t_neg_num);
	example_num = t_pos_num + t_neg_num;/*所有样本*/

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
		//fpr_r = MAX(fpr_e, fpr_g);
		fpr_r = (fpr_e + fpr_g) / 2;
		//fnr_r = MAX(fnr_e, fnr_g);
		fnr_r = (fnr_e + fnr_g) / 2;

		printf("layer:%d stump_num:%d s_l:%f u:%f\n", l, new_stage->stump_num, s_l, u);
		printf("training set: fnr:%f fpr:%f\nvalidationset fnr:%f fpr:%f\n", fnr_e, fpr_e, fnr_g, fpr_g);

		if(fpr_r <= fpr_perlayer && fnr_r <= fnr_perlayer)
		{
			fpr = fpr * fpr_r;
		}
		else if(fpr_r <= fpr_perlayer && fnr_r > fnr_perlayer && u > 10e-5)
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
		else if(fpr_r > fpr_perlayer && fnr_r <= fnr_perlayer && u > 10e-5)
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
			if(T_l >= MAX_DEPTH(l))/*深度达到上限*/
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
					//fpr_r = MAX(fpr_e, fpr_g);
					fpr_r = (fpr_e + fpr_g) / 2;
					//fnr_r = MAX(fnr_e, fnr_g);
					fnr_r = (fnr_e + fnr_g) / 2;
				}
				fpr = fpr * fpr_r;
				printf("fnr:%f fpr:%f\n", fnr_r, fpr_r);
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
		
		printf("fpr:%f fpr_r:%f fnr_r:%f\n", fpr, fpr_r, fnr_r);
		printf("add one stage\n");


		if(l % 2 == 0)/*每两层保存一次模型*/
		{
			char buf[100];
			i32 len = sprintf(buf, "./backup/attentional_cascade_%d.cfg", l);
			buf[len] = 0;
			printf("save model layer:%d fpr:%f\n", l, fpr);
			save_model(model, buf);
		}
	
		printf("layer:%d stump_num:%d s_l:%f u:%f\n", l, new_stage->stump_num, s_l, u);
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
		t_neg = make_neg_example(t_neg_data, 0, t_neg_num, wnd_size, model);
		v_neg = make_neg_example(v_neg_data, 0, v_neg_num, wnd_size, model);
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

	/*释放图片数据*/
	free_image_data(t_pos_data);
	free_image_data(t_neg_data);
	free_image_data(v_pos_data);
	free_image_data(v_neg_data);

	return model;
}


/*
功能：模型测试
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
功能：通过模型对样本进行预测
*/
void make_predictions(Model *m, Train_example *examples, i32 example_num)
{
	i32 i;

	#pragma omp parallel for
	for(i = 0; i < example_num; i++)
	{
		examples[i].predict_label = model_func(m, examples[i].integ);
	}
}


/*
功能：对样本中的真阴性的样本进行剔除
返回值：移除的真阴性样本数
备注：
*/
i32 screen_examples(Train_example *examples, i32 example_num)
{
	i32 count = 0;
	i32 i;

	for(i = example_num - 1; i >=0; i--)
	{
		if(examples[i].predict_label == -1 && examples[i].label == -1)/*真阴性样本*/
		{
			free_image(examples[i].integ);
			free_image(examples[i].integ);
			memcpy(&examples[i], &examples[example_num - 1], sizeof(Train_example));
			example_num--;
			count++;
		}
	}
	return count;
}


/*
功能：扫描整个图像，筛选出
*/
Sub_wnd *scan_image(Model *model, image im, i32 *wnd_num)
{

}


/*
功能：检测后处理，对检测窗进行进一步筛选，剔除掉虚警和重复检测
*/



/*
功能：释放模型
*/
void free_model(Model *model)
{

}
