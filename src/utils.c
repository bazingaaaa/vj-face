#include "type.h"
#include "const.h"
//#include "image.h"
#include <time.h>
#include <string.h>
#include "stdlib.h"
#include "list.h"
#include <vector>
#include "feature.h"
#include "classifier.h"
#include "model.h"
//#include "data.h"
//#include "proto.h"
//#include <stddef.h>

#ifdef _WIN64
#include <windows.h>
int gettimeofday(struct timeval* tp, void* tzp)
{
    time_t clock;
    struct tm tm;
    SYSTEMTIME wtm;
    GetLocalTime(&wtm);
    tm.tm_year = wtm.wYear - 1900;
    tm.tm_mon = wtm.wMonth - 1;
    tm.tm_mday = wtm.wDay;
    tm.tm_hour = wtm.wHour;
    tm.tm_min = wtm.wMinute;
    tm.tm_sec = wtm.wSecond;
    tm.tm_isdst = -1;
    clock = mktime(&tm);
    tp->tv_sec = clock;
    tp->tv_usec = wtm.wMilliseconds * 1000;
    return (0);
}
#else
#include <sys/time.h>
#endif


typedef struct{
    char *key;
    char *val;
    int used;
} kvp;


void option_insert(List *l, char *key, char *val);
int read_option(char *s, List *options);
void strip(char *s);


/*
功能：断言失败
*/
void assertion_failure(char *exp, char *file, int line)
{
	printf("assert(%s) failed:file:%s,ln:%d", exp, file, line);

	while(1)
	{
		;
	}
}


/*
功能：对整形范围进行限制
*/
int constrain_int(int a, int min, int max)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}


/*
功能：时间函数
*/
void times(const char * which)
{
	/* If which is not empty, print the times since the previous call. */
	static double last_wall = 0.0, last_cpu = 0.0;
	double wall, cpu;
	struct timeval tv;
	clock_t stamp = 0;

	wall = last_wall;
	cpu = last_cpu;
	if (gettimeofday(&tv,NULL) != 0 || (stamp = clock()) == (clock_t)-1)
	{
		printf("Unable to get times\n");
	}
	last_wall = tv.tv_sec+1.0e-6*tv.tv_usec;
	last_cpu = stamp/(double)CLOCKS_PER_SEC;
	if (strlen(which) > 0)
	{
	    wall = last_wall-wall;
	    cpu = last_cpu-cpu;
	    printf("\n%s", which);
	    printf("time = %lf seconds, CPU = %lf  seconds\n\n", wall, cpu);
	}
}


/*
功能：删除参数
*/
void del_arg(int argc, char **argv, int index)
{
    int i;
    for(i = index; i < argc-1; ++i) argv[i] = argv[i+1];
    argv[i] = 0;
}


/*
功能：找到并删除参数
*/
int find_arg(int argc, char* argv[], char *arg)
{
    int i;
    for(i = 0; i < argc; ++i) {
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)) {
            del_arg(argc, argv, i);
            return 1;
        }
    }
    return 0;
}


/*
功能：找到指定字符串后的字符串参数
*/
char *find_char_arg(int argc, char **argv, char *arg, char *def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = argv[i+1];
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}


/*
功能：找到指定字符串后的整形参数
*/
int find_int_arg(int argc, char **argv, char *arg, int def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = atoi(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}



/*
功能：从文件中获取一行
*/
char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    i32 size = 512;
    char *line = (char*)malloc(size*sizeof(char));
    if(!fgets(line, size, fp)){
        free(line);
        return 0;
    }

    i32 curr = strlen(line);

    while((line[curr-1] != '\n') && !feof(fp)){
        if(curr == size-1){
            size *= 2;
            line = (char*)realloc(line, size*sizeof(char));
            if(!line) {
                fprintf(stderr, "malloc failed %ld\n", size);
                exit(0);
            }
        }
        i32 readsize = size-curr;
        if(readsize > INT_MAX) readsize = INT_MAX-1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    if(line[curr-1] == '\n') line[curr-1] = '\0';

    return line;
}


/*
功能：读取数据配置文件
*/
List *read_data_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0)
    {
    	printf("cannot open file %s\n", filename);
        return NULL;
    }
    char *line;
    int nu = 0;
    List *options = make_list();
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}


/*
功能：读取选项
*/
int read_option(char *s, List *options)
{
    i32 i;
    i32 len = strlen(s);
    char *val = 0;
    for(i = 0; i < len; ++i){
        if(s[i] == '='){
            s[i] = '\0';
            val = s+i+1;
            break;
        }
    }
    if(i == len-1) return 0;
    char *key = s;
    option_insert(options, key, val);
    return 1;
}


/*
功能：选项插入
*/
void option_insert(List *l, char *key, char *val)
{
    kvp *p = (kvp*)malloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    list_insert(l, p);
}


/*
功能：字符串抽取
*/
void strip(char *s)
{
    i32 i;
    i32 len = strlen(s);
    i32 offset = 0;
    for(i = 0; i < len; ++i){
        char c = s[i];
        if(c==' '||c=='\t'||c=='\n') ++offset;
        else s[i-offset] = c;
    }
    s[len-offset] = '\0';
}


/*
功能：选项寻找
*/
char *option_find(List *l, char *key)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(strcmp(p->key, key) == 0){
            p->used = 1;
            return p->val;
        }
        n = n->next;
    }
    return 0;
}


/*
功能：选项寻找，字符串
*/
char *option_find_str(List *l, char *key, char *def)
{
    char *v = option_find(l, key);
    if(v) return v;
    if(def) fprintf(stderr, "%s: Using default '%s'\n", key, def);
    return def;
}


/*
功能：选项寻找，整型
*/
int option_find_int(List *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    fprintf(stderr, "%s: Using default '%d'\n", key, def);
    return def;
}


/*
功能：选项寻找，浮点型
*/
double option_find_float(List* l, char* key, double def)
{
    char* v = option_find(l, key);
    if (v) return atof(v);
    fprintf(stderr, "%s: Using default '%lf'\n", key, def);
    return def;
}