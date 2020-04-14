OPENCV=1
OPENMP=1
DEBUG=0

OBJ=classifier.o feature.o image.o load_image.o utils.o model.o list.o data.o
EXOBJ=main.o

VPATH=./src/:./
#SLIB=haarcascade.so
#ALIB=haarcascade.a
EXEC=haarcascade
OBJDIR=./obj/

CC=g++
LD=ld
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= -Ih/ -Isrc/ 
CFLAGS=-Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC -std=c++11

ifeq ($(OPENMP), 1) 
CFLAGS+= -Xpreprocessor -fopenmp
LDFLAGS+= -lomp
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
COMMON= -Ih/ -Isrc/ 
else
CFLAGS+= -flto
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV -I/usr/local/include/opencv4
LDFLAGS+= `pkg-config --libs opencv4` 
COMMON+= `pkg-config --cflags opencv4`  
endif

EXOBJS = $(addprefix $(OBJDIR), $(EXOBJ))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard h/*.h) makefile 

all: obj $(SLIB) $(ALIB) $(EXEC)
#all: obj $(EXEC)


$(EXEC): $(EXOBJS) $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) 

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXOBJS) $(OBJDIR)/*

