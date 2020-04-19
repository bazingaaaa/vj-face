#ifndef _PROTO_H_
#define _PROTO_H_


/*feature.c*/
float calc_featA_val(image integ, i32 i, i32 j, i32 w, i32 h);
float calc_featB_val(image integ, i32 i, i32 j, i32 w, i32 h);
float calc_featC_val(image integ, i32 i, i32 j, i32 w, i32 h);
float calc_featD_val(image integ, i32 i, i32 j, i32 w, i32 h);
float calc_featE_val(image integ, i32 i, i32 j, i32 w, i32 h);
Haar_feat *make_haar_features(u16 wndSize, i32 *num);
float calc_haar_feat_val(image integ, Haar_feat *pFeat);


/*classifier.c*/
void calc_example_feat_val(Feat_info *parallel_examples, Train_example *array, i32 example_num, Haar_feat *pFeat);
Stump *find_best_stump(Feat_info **parallel_examples, Train_example *array, i32 example_num, Haar_feat *feat_array, i32 feat_num);
float test_stump(Stump *stmp, Train_example *examples, i32 example_num);
float test_stage(Stage *s, Train_example *examples, i32 example_num);
i8 stump_func(Stump *stmp, image integ, i32 train_flag);
i8 stage_func(Stage *s, image integ, i32 train_flag);
Stage *adaboost(Feat_info **parallel_examples, Train_example *examples, i32 example_num, i32 pos_num, i32 neg_num, Haar_feat *feat_array, i32 feat_num, i32 depth);
void add_stump_2_stage(Stage *s, Feat_info** parallel_examples, Train_example *examples, i32 example_num, Haar_feat *feat_array, i32 feat_num);


/*model.h*/
i8 save_model(Model *m, const char* path);
Model *load_model(const char* path);
float test_model(Model *m, Train_example *examples, i32 example_num);
void make_predictions(Model *m, Train_example *examples, i32 example_num);
i8 model_func(Model *m, image integ);
Model *attentional_cascade(char *save_path, Model *model, Data t_pos_data, Data v_pos_data, Data t_neg_data, Data v_neg_data, i32 wnd_size, float fpr_overall, float fpr_perstage, float fnr_perstage);
i32 run_detection(image im, Model *model, i32 skin_test_flag, char *savepath);
float test_model(Model *m, Train_example *examples, i32 example_num);
void scan_image_for_testing(std::vector<Sub_wnd> &candidate, Model *model, image im, i32 wnd_size, float scale_size, i32 step_size);
void scan_image_for_training(std::vector<Sub_wnd> &candidate, Model *model, image im, i32 wnd_size, float scale_size, i32 step_size);
Train_example *collect_false_positives(Data data, Model *model, i32 example_num, i32 wnd_size, float scale_size, i32 step_size);
i8 skin_test(image src, Sub_wnd wnd);
void free_model(Model *model, i32 is_load_model);
i32 get_detect_wnd_size(Model *model);


/*utils.h*/
void assertion_failure(char *exp, char *file, char *base_file, int line);
int constrain_int(int a, int min, int max);


/*data.h*/
Data load_image_data(char *images);
Train_example * make_pos_example(Data data);
Train_example * make_neg_example(Data data, i32 init_flag, i32 example_num, i32 wnd_size, Model *pModel, float fpr, float scale_size, i32 step_size);
Train_example *merge_pos_neg(Train_example *pos, i32 pos_num, Train_example *neg, i32 neg_num);
void prepare_pos_examples(Data data);
Feat_info **make_parallel_examples(i32 example_num, i32 feat_num);
void free_parallel_examples(Feat_info **parrel_examples, i32 feat_num);
void free_image_data(Data data);


/*utils.h*/
int constrain_int(int a, int min, int max);
void times(const char * which);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
int find_int_arg(int argc, char **argv, char *arg, int def);
char *fgetl(FILE *fp);
void strip(char *s);
char *option_find(List *l, char *key);
char *option_find_str(List *l, char *key, char *def);
int option_find_int(List *l, char *key, int def);
List *read_data_cfg(char *filename);
float option_find_float(List *l, char *key, float def);


#endif