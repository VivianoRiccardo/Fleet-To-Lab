DIR:= src/


SRCS = $(wildcard $(DIR)*.c)

PROGS = $(patsubst %.c,%,$(SRCS))
PROGTESTS = $(patsubst %.c,%,$(SRCSTEST))

CC:= gcc

CFLAGS:= -O3 -mavx2
LDLIBS:= -lm -lpthread -fPIC
LABLIB:= -lezsparkc

all: $(PROGS)

%: %.c
	$(CC) -c $< -o $@.o $(CFLAGS) $(LDLIBS) 

create: $(DIR)
	ar r libezsparkc.lib $(DIR)*.o
	rm $(DIR)*.o

shared: $(DIR)
	gcc $(DIR)*.o -shared -o libezsparkc.so 
	rm $(DIR)*.o


shared_windows: $(DIR)
	gcc $(DIR)*.o -shared -o libezsparkc.dll 
	rm $(DIR)*.o
