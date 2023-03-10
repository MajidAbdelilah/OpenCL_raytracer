MKDIR   := mkdir
RM   := rm
CC      := gcc
BIN     := ./bin
OBJ     := ./obj
INCLUDE := ./include
SRC     := ./src
SRCS    := $(wildcard $(SRC)/*.c)
OBJS    := $(patsubst $(SRC)/%.c,$(OBJ)/%.o,$(SRCS))
BINARY     := ocl_game_engine
CFLAGS := -I$(SRC) -msse -msse2 -mavx -mavx2 -mtune=native -std=gnu99 
CDFLAGS  :=  -Wall -g
COFLAGS  :=  -Wall -Ofast
PGO_GEN_FLAGS = -fprofile-generate 
PGO_USE_FLAGS = -fprofile-use 
LDLIBS  := -lm -lOpenCL  
LDFLAGS := #-Ofast -Wall
flags := 
#-Ofast -Wall
.PHONY: all run clean

all: debug

debug: CFLAGS += $(CDFLAGS)
debug: LDFLAGS += $(CDFLAGS)
debug: $(BIN)/$(BINARY)

release: CFLAGS += $(COFLAGS)
release: LDFLAGS += $(COFLAGS)
release: $(BIN)/$(BINARY)

pgogen: CFLAGS += $(COFLAGS) $(PGO_GEN_FLAGS)
pgogen: LDFLAGS += $(COFLAGS) $(PGO_GEN_FLAGS)
pgogen: $(BIN)/$(BINARY)

pgouse: CFLAGS += $(COFLAGS) $(PGO_USE_FLAGS)
pgouse: LDFLAGS += $(COFLAGS) $(PGO_USE_FLAGS)
pgouse: $(BIN)/$(BINARY)

win_release: CC = x86_64-w64-mingw32-gcc
win_release: CFLAGS += -O0
win_release: LDFLAGS += -O0
win_release: LDLIBS  = -lm 
win_release: $(OBJS) | $(BIN)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(BIN)/$(BINARY): $(OBJS) | $(BIN)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(OBJ)/%.o: $(SRC)/%.c | $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN) $(OBJ):
	$(MKDIR) $@

run: $(BIN)/$(BINARY)
	cd $(BIN) && ./$(BINARY) $(flags) > ../image.ppm

clean:
	$(RM) $(OBJ)/*.o $(BIN)/$(BINARY)

ddd: debug
	ddd $(BIN)/$(BINARY)
valgrind: debug
	valgrind $(BIN)/$(BINARY)
