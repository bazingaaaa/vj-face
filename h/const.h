#ifndef _CONST_H_
#define _CONST_H_


#define ASSERT
#ifdef ASSERT
void assertion_failure(char *exp, char *file, char *base_file, int line);
#define assert(exp)  if (exp) ; \
        else assertion_failure(#exp, __FILE__, __BASE_FILE__, __LINE__)
#else
#define assert(exp)
#endif
      
#define PRIVATE static 
#define PUBLIC
#define NEAREST_INTEGER(a) ((int)(a + 0.5))  	


#define MAG_V 0xCC

#define TWOPI 6.2831853

#define SET_RANGE(a, min, max) {\
                                    if(a<min)\
                                        a=min;\
                                    if(a>max)\
                                    a=max;\
                                }


#define MAX_DEPTH(l) (MIN((10 * l) + 10, 200))
//#define MAX_DEPTH(l) (0)

#define UNIT_DECAY_RATE (0.8)
#define FPR_GOAL (10e-7)


/*目前仅实现一种特征模式*/
typedef enum haar_feature_mode
{
	HAAR_BASIC,/*仅包含垂直特征，共5种特征模版*/
	//HAAR_ALL/*包含边缘，线性，中心特征，共14种特征模版*/
}Feat_mode;



/*haar特征类别,共五种,其中
a和c由两个矩形组成，分别为水平和垂直结构
b和d由三个矩形组成，分别为水平和垂直结构
e由四个矩形组成，水平和垂直方向均有两个
*/
typedef enum haar_feature_type
{
	FEAT_A,
	FEAT_B,
	FEAT_C,
	FEAT_D,
	FEAT_E
}Feat_type;


#endif