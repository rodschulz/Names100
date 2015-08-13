IS_WINDOWS	:= n
IS_DEBUG	:= y
USE_PKGCONFIG	:= y

CXXFLAGS	:= -Wall -Wextra -pedantic -Wno-unused-function -std=gnu++11 -fopenmp
LDFLAGS		:= -lpthread -lm

ifeq ($(IS_WINDOWS),y)
	SUFFIX_EXE  := .exe

	ifeq ($(IS_DEBUG),y)
		CXXFLAGS += -O0 -ggdb
	else
		CXXFLAGS += -O2
	endif
else
	SUFFIX_EXE  :=

	ifeq ($(IS_DEBUG),y)
		CXXFLAGS += -O0 -ggdb
	else
		CXXFLAGS += -O3
	endif
endif

########## pkg-config ##########

ifeq ($(USE_PKGCONFIG),y)
	ifneq ($(shell pkg-config --exists opencv && echo ok),ok)
		$(warning cant find opencv)
	endif

	CXXFLAGS  += $(shell pkg-config --cflags opencv)
	LDFLAGS += $(shell pkg-config --libs   opencv)
else
	OPENCV_HEADERS  := C:\...\opencv-2.4.9\include
	OPENCV_LIBS     := C:\...\opencv-2.4.9\...\bin
	CXXFLAGS  += -I$(OPENCV_HEADERS)
	LDFLAGS += -L$(OPENCV_LIBS) -lopencv_core249 -lopencv_highgui249 -lopencv_imgproc249
endif

########## rules ##########

SOURCE_DIR	:= src
BUILD_DIR	:= build
TARGET		:= $(BUILD_DIR)/Names$(SUFFIX_EXE)
CPP_FILES	:= $(wildcard $(SOURCE_DIR)/*.cpp)
OBJS		:= $(addprefix $(BUILD_DIR)/, $(notdir $(CPP_FILES:.cpp=.o)))

.PHONY: all clean

all: $(TARGET)

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cpp $(SOURCE_DIR)/%.h
	mkdir -p $(BUILD_DIR)
	$(CXX) -o $@ -c $< $(CXXFLAGS)

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) -o $@ -c $< $(CXXFLAGS)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -rf ./$(BUILD_DIR)/*
	rm -rf ./$(BUILD_DIR)
