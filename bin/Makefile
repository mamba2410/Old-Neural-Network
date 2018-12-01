
########################################################
# CHANGE ME IF SUBDIRECTORIES WERE ADDED
# CHANGE MY FILE EXTENSION
########################################################

# File extension
FE=cpp

# Set compiler to gcc, give it gcc flags, linker flags, set name
CC=g++
CFLAGS=
LFLAGS=
EXE_NAME=nno.exe

# Tells make what directories to look in for things
PROGRAM_SRC=../src
PROGRAM_HEAD=../include
OBJ_DIR=objects

# Makes an 'array' of files in source and head
PROGRAMSOURCES=$(wildcard $(PROGRAM_SRC)/*.$(FE))
PROGRAMHEADS=$(wildcard $(PROGRAM_HEAD)/*.h)

# Tells make what all the object files are called
OBJECTS=$(patsubst $(PROGRAM_SRC)/%.$(FE), $(OBJ_DIR)/%.o, $(PROGRAMSOURCES))

# For any %.o, compile it with this command
$(OBJ_DIR)/%.o: $(PROGRAM_SRC)/%.$(FE)
	$(CC) $(CFLAGS) -I$(PROGRAM_SRC) -I$(PROGRAM_HEAD) -c $< -o $@ 

# Links all of the objects together, recompiles if objects/headers changed
build: $(OBJECTS) $(PROGRAMHEADS)
	$(CC) -o $(EXE_NAME) $(LFLAGS) $(OBJECTS)

run: build
	@./$(EXE_NAME)

clean:
	rm -rf $(OBJ_DIR)
	rm $(EXE_NAME)
